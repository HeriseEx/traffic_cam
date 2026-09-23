import hashlib
import json
import secrets
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app, Event
from config import Settings
from inference import Detector, Tracker, clip_windows, bind_plates
from store import Store
from worker import process_one
from schemas import AnalysisConfig
from plates import PlateReader, consensus, ctc_decode, onnxocr_decode, ONNXOCR_CHARS
from rules import candidates, red_approach, red_approaches, red_light_candidate, lane_changes, red_during_laterals, restricted_park, turned_from_green
from signals import SignalMachine, ThroughLamp, detect_lights, observe, color_of
import cv2
import numpy as np


class BackendTest(unittest.TestCase):
    def test_legacy_device_session_migrates_without_resurrection(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data=Path(directory), reserve_bytes=0)
            store = Store(settings)
            session = 'legacy-client-session'
            digest = hashlib.sha256(session.encode()).hexdigest()
            with store.connection() as db:
                db.execute('INSERT INTO clients VALUES(?,?,?,?,?,?,?,?,?)',
                    ('client-old', 'device-old', 'android', 'phone', '2', '127.0.0.1', digest, time.time(), time.time()))
            migrated = Store(settings)
            self.assertTrue(migrated.client_ok(session))
            newer = migrated.hello('device-old', 'android', 'phone', '2', '127.0.0.2')
            self.assertEqual(newer['client_id'], 'client-old')
            self.assertEqual(migrated.overview()['client_count'], 1)
            self.assertTrue(migrated.client_ok(session))
            migrated.revoke_session(session)
            restarted = Store(settings)
            self.assertFalse(restarted.client_ok(session))
            self.assertTrue(restarted.client_ok(newer['session']))
            with restarted.connection() as db:
                db.execute('UPDATE client_sessions SET expires_at=?', (time.time()-1,))
            self.assertFalse(Store(settings).client_ok(newer['session']))

    def test_capture_proxy_preserves_media_headers(self):
        import http.client
        import http.server
        import threading
        from unittest.mock import patch
        from urllib.parse import urlparse
        import webserve

        seen = []
        class Upstream(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(dict(self.headers))
                partial = self.headers.get('Range') == 'bytes=0-3'
                self.send_response(206 if partial else 200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Content-Length', '4' if partial else '8')
                self.send_header('Set-Cookie', 'traffic_client=test; HttpOnly; SameSite=Lax')
                self.send_header('Set-Cookie', 'traffic_device=device; HttpOnly; SameSite=Lax')
                if partial:
                    self.send_header('Content-Range', 'bytes 0-3/8')
                self.end_headers()
                if self.command != 'HEAD':
                    self.wfile.write(b'abcd' if partial else b'abcdefgh')
            do_HEAD = do_GET
            def log_message(self, *args):
                pass

        class Proxy(webserve.Handler):
            def log_message(self, *args):
                pass

        with http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream) as upstream, \
             http.server.ThreadingHTTPServer(('127.0.0.1', 0), Proxy) as proxy:
            threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (upstream, proxy)]
            for thread in threads:
                thread.start()
            try:
                with patch.object(webserve, 'UPSTREAM', urlparse(f'http://127.0.0.1:{upstream.server_port}')):
                    for method, ranged, expected_size in [('GET', False, '8'), ('GET', True, '4'), ('HEAD', False, '8')]:
                        conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=5)
                        try:
                            headers = {'Cookie': 'traffic_client=test'}
                            if ranged:
                                headers['Range'] = 'bytes=0-3'
                            conn.request(method, '/v1/tasks/example/video', headers=headers)
                            response = conn.getresponse()
                            self.assertEqual(response.status, 206 if ranged else 200)
                            self.assertEqual(response.headers.get_all('Content-Length'), [expected_size])
                            self.assertEqual(len(response.headers.get_all('Set-Cookie')), 2)
                            self.assertEqual(response.read(), b'' if method == 'HEAD' else b'abcd' if ranged else b'abcdefgh')
                            self.assertEqual(seen[-1]['Cookie'], 'traffic_client=test')
                            if ranged:
                                self.assertEqual(response.headers['Content-Range'], 'bytes 0-3/8')
                        finally:
                            conn.close()
            finally:
                for server in (proxy, upstream):
                    server.shutdown()
                for thread in threads:
                    thread.join(timeout=5)

    def test_browser_media_cookies_and_persistent_device_identity(self):
        from concurrent.futures import ThreadPoolExecutor
        from hello import code as hello_code
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data=Path(directory), token='browser-regression-token-long-enough', reserve_bytes=0)
            app = create_app(settings)
            store = app.state.store

            def hello(client, device):
                stamp, nonce = int(time.time()), secrets.token_hex(16)
                return client.post('/v1/hello', json={'device_id': device, 'platform': 'web', 'model': 'Same browser model',
                    'ts': stamp, 'nonce': nonce, 'code': hello_code(device, 'web', stamp, nonce)})

            with TestClient(app, base_url='https://testserver') as browser, TestClient(app, base_url='https://testserver') as other:
                initial = hello(browser, 'persistent-browser-a')
                self.assertEqual(initial.status_code, 200)
                device = initial.json()
                self.assertTrue(all('HttpOnly' in value and 'Secure' in value for value in initial.headers.get_list('set-cookie')))
                # Reloads and a cleared/changed localStorage ID still resolve to the browser's signed device cookie.
                for _ in range(3):
                    resumed = hello(browser, str(uuid.uuid4())).json()
                    self.assertEqual(resumed['client_id'], device['client_id'])
                    self.assertEqual(resumed['device_id'], 'persistent-browser-a')
                    self.assertEqual(resumed['session'], device['session'])
                browser.cookies.delete('traffic_client')
                self.assertEqual(hello(browser, 'recreated-storage-id').json()['client_id'], device['client_id'])
                # Equal model and IP do not identify a device; two real browsers must remain distinct.
                self.assertNotEqual(hello(other, 'persistent-browser-b').json()['client_id'], device['client_id'])
                self.assertEqual(browser.get('/v1/overview').json()['client_count'], 2)
                # Concurrent sessions on the same device remain valid without adding clients.
                with ThreadPoolExecutor(max_workers=4) as pool:
                    sessions = list(pool.map(lambda _: store.hello('persistent-browser-a', 'web', 'Same browser model', '2', '127.0.0.1'), range(6)))
                self.assertTrue(all(row['client_id'] == device['client_id'] and store.client_ok(row['session']) for row in sessions))
                self.assertTrue(store.client_ok(device['session']))
                self.assertEqual(browser.get('/v1/overview').json()['client_count'], 2)
                content = (root/'tests/traffic.mp4').read_bytes()
                headers = {'X-Requested-With': 'traffic-console', 'X-Event-Metadata': json.dumps({'event_id': str(uuid.uuid4())}),
                           'X-Video-SHA256': hashlib.sha256(content).hexdigest()}
                task = browser.post('/v1/tasks', content=content, headers=headers).json()
                video = f"/v1/tasks/{task['task_id']}/video"
                preview = browser.get(video)
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(preview.headers['content-type'], 'video/mp4')
                fragment = browser.get(video, headers={'Range': 'bytes=0-31'})
                self.assertEqual(fragment.status_code, 206)
                self.assertEqual(fragment.content, preview.content[:32])
                self.assertEqual(browser.head(video).headers['content-length'], str(len(preview.content)))
                original = browser.get(video+'?original=true')
                self.assertEqual(original.content, content)
                self.assertIn('attachment', original.headers['content-disposition'])
                clips = store.clips_dir(task['task_id']); clips.mkdir(parents=True)
                (clips/'0.mp4').write_bytes(content)
                self.assertEqual(browser.get(f"/v1/tasks/{task['task_id']}/clips/0", headers={'Range': 'bytes=0-31'}).status_code, 206)
                self.assertEqual(browser.post('/v1/archive').status_code, 403)
                self.assertEqual(browser.delete('/v1/session', headers={'X-Requested-With':'traffic-console'}).status_code, 200)
                self.assertEqual(browser.get(video).status_code, 401)
                self.assertEqual(browser.get(video+'?original=true').status_code, 401)
                self.assertEqual(hello(browser, 'after-logout-device').json()['client_id'], device['client_id'])
                self.assertEqual(browser.get('/v1/overview').json()['client_count'], 2)

    def test_mobile_capture_intervals_and_legacy_retries(self):
        from pydantic import ValidationError
        capture = {'camera_mode': 'moving', 'captured_at': 1700000000, 'duration_ms': 2000,
                   'recording_gaps_ms': 20, 'incidents': [{'track_id': 1, 'kind': 'RED_LIGHT',
                   'start_ms': 200, 'end_ms': 1400, 'plate': '川A12345', 'plate_confirmed': True}]}
        event = {'event_id': str(uuid.uuid4()), 'capture': capture}
        self.assertEqual(Event.model_validate(event).capture.incidents[0].track_id, 1)
        for broken in ({**capture, 'duration_ms': 100}, {**capture, 'captured_at': float('nan')},
                       {**capture, 'incidents': [{**capture['incidents'][0], 'start_ms': 1900}]}):
            with self.assertRaises(ValidationError):
                Event.model_validate({**event, 'capture': broken})

    def test_review_model_switch_changes_video_and_submission_lock(self):
        from settings_security import SettingsPassword
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            settings=Settings(data=Path(directory),model=root/'models/yolox_s.onnx',token='test-console-token-not-production',reserve_bytes=0)
            app=create_app(settings);store=app.state.store
            SettingsPassword(store).set_password('test-only-settings-passphrase')
            content=(root/'tests/traffic.mp4').read_bytes()
            auth={'Authorization':f'Bearer {settings.token}'}
            with TestClient(app) as client:
                self.assertEqual(client.get('/v1/settings').status_code,401)
                login=client.post('/v1/session',headers=auth)
                self.assertEqual(login.status_code,200)
                config=client.get('/v1/settings').json()
                changed={**config['config'],'vehicle_model':'yolox_tiny','vehicle_threshold':.3,'plate_enabled':False}
                body={'expected_revision':config['revision'],'config':changed,'password':'test-only-settings-passphrase'}
                self.assertEqual(client.put('/v1/settings',json=body).status_code,403)
                self.assertEqual(client.put('/v1/settings',headers={'X-Requested-With':'traffic-console'},json=body).status_code,200)
                self.assertEqual(client.put('/v1/settings',headers=auth,json=body).status_code,409)
                task=client.post('/v1/tasks',content=content,headers={**auth,'X-Event-Metadata':json.dumps({'event_id':str(uuid.uuid4()),'trigger':'voice','trigger_text':'开始标记'},ensure_ascii=True),'X-Video-SHA256':hashlib.sha256(content).hexdigest()}).json()
                self.assertEqual(task['analysis_config']['vehicle_model'],'yolox_tiny')
                lamp=np.zeros((180,320,3),np.uint8); cv2.circle(lamp,(160,40),5,(0,0,255),-1)
                ok,jpeg=cv2.imencode('.jpg',lamp)
                self.assertTrue(ok)
                live=client.post('/v1/recognize-frame',content=bytes(jpeg),headers={**auth,'Content-Type':'image/jpeg'}).json()
                self.assertEqual(live['signal_observed'],'RED')
                self.assertIn('lights',live)
                self.assertIn('vehicles',live)
                process_one(store,None,'new-worker',{})
                task=store.get(task['task_id']);task_id=task['task_id']
                self.assertEqual(task['status'],'ANALYZED')
                self.assertEqual(task['result']['model']['input_size'],416)
                ai=task['result']
                cursor=client.get('/v1/changes').json()['cursor']
                review={'expected_revision':task['revision'],'decision':'INVALID','reviewer':'测试员','note':'回传验证：误检','plate':'苏ED51712','violation_type':'NONE'}
                corrected=client.post(f'/v1/tasks/{task_id}/review',json=review,headers=auth).json()
                self.assertEqual(corrected['result'],ai)
                self.assertEqual(corrected['effective_result']['decision'],'REJECTED')
                self.assertEqual(corrected['effective_result']['plate'],'苏ED51712')
                changes=client.get(f'/v1/changes?after={cursor}').json()
                self.assertGreater(changes['cursor'],cursor)
                self.assertEqual(changes['tasks'][0]['review']['decision'],'INVALID')
                self.assertEqual(client.post(f'/v1/tasks/{task_id}/review',json=review,headers=auth).status_code,409)
                self.assertEqual(len(client.get(f'/v1/tasks/{task_id}/audit').json()['history']),1)
                ranged=client.get(f'/v1/tasks/{task_id}/video',headers={'Range':'bytes=0-127'})
                self.assertEqual(ranged.status_code,206)
                self.assertEqual(len(ranged.content),128)
                self.assertNotIn('frames',changes['tasks'][0]['result'])
                self.assertIn('frames',client.get(f'/v1/tasks/{task_id}').json()['result'])
                reset=client.post(f'/v1/tasks/{task_id}/review',headers=auth,json={
                    'expected_revision':corrected['revision'],'decision':'RESET','reviewer':'测试员'}).json()
                self.assertIsNone(reset['review'])
                self.assertEqual(reset['effective_result']['source'],'AI')
                self.assertEqual(reset['result'],ai)
                corrected=reset
                # Reanalysis preserves the prior AI/review snapshot in the audit.
                rerun=client.post(f'/v1/tasks/{task_id}/reanalyze',headers=auth,
                    json={'expected_revision':corrected['revision']}).json()
                self.assertEqual(rerun['status'],'QUEUED')
                self.assertIsNone(rerun['review'])
                process_one(store,None,'second-worker',{})
                corrected=store.get(task_id)
                saved=client.get(f'/v1/tasks/{task_id}/audit').json()['history'][0]['payload']
                self.assertEqual(saved['result'],ai)
                review['expected_revision']=corrected['revision']
                corrected=client.post(f'/v1/tasks/{task_id}/review',headers=auth,json=review).json()
                self.assertEqual(client.post(f'/v1/tasks/{task_id}/submission-receipt',headers=auth,
                    json={'expected_revision':corrected['revision'],'receipt':'test-external-receipt'}).status_code,200)
                locked=store.get(task_id)
                review['expected_revision']=locked['revision']
                self.assertEqual(client.post(f'/v1/tasks/{task_id}/review',json=review,headers=auth).status_code,409)
                self.assertEqual(client.post(f'/v1/tasks/{task_id}/reanalyze',headers=auth,
                    json={'expected_revision':locked['revision']}).status_code,409)
                # A restarted API still sees the human result and immutable submission state.
                self.assertEqual(Store(settings).get(task_id)['review']['decision'],'INVALID')

    def test_real_chinese_plate_and_rule_gates(self):
        root=Path(__file__).resolve().parent
        reader=PlateReader(root/'models/plate')
        plates=reader.read(cv2.imread(str(root/'tests/plate.jpg')))
        self.assertIn('苏ED51712',[p['text'] for p in plates])
        preds=np.zeros((4,3),np.float32); preds[(0,1,2),(1,1,2)]=1
        self.assertEqual(ctc_decode(preds,['blank','A','B'])[0],'AB')
        ppocr=root/'models/plate/ppocrv5_server_rec.onnx'
        if ppocr.is_file():
            newer=PlateReader(root/'models/plate',2,'ppocrv5_server')
            self.assertTrue(any(p['text']=='苏ED51712' for p in newer.read(cv2.imread(str(root/'tests/plate.jpg')))))
        onnx=root/'models/plate/onnxocr_plate_rec.onnx'
        if onnx.is_file():
            dedicated=PlateReader(root/'models/plate',2,'onnxocr_plate')
            self.assertTrue(any(p['text']=='苏ED51712' for p in dedicated.read(cv2.imread(str(root/'tests/plate.jpg')))))
        logits=np.full((3,len(ONNXOCR_CHARS)), -8, np.float32)
        logits[(0,2),(11,56)]=8
        self.assertEqual(onnxocr_decode(logits)[0],'苏E')
        votes=consensus([{'time_seconds':0,'plates':plates},{'time_seconds':.5,'plates':plates}])
        self.assertTrue(votes[0]['stable'])
        self.assertFalse(consensus([{'time_seconds':0,'plates':plates}])[0]['stable'])
        # Synthetic geometry validates the rule implementation, not road-scene accuracy.
        frames=[{'time_seconds':i*.5,'vehicles':[{'track_id':1,'score':.9,'box_normalized':[x-.04,.3,x+.04,.5]}]}
                for i,x in enumerate([.7,.6,.4,.3])]
        scene={'fixed_camera':True,'solid_line':[[.5,.1],[.5,.9]],'allowed_direction':[[.1,.5],[.9,.5]]}
        found,status=candidates(frames,scene,'STATIONARY')
        self.assertEqual({v['type'] for v in found},{'SOLID_LINE','WRONG_WAY'})
        self.assertEqual(candidates(frames,scene,'MOVING_CAMERA'),([], 'MOVING_CAMERA'))
        self.assertEqual(candidates(frames,None,'STATIONARY'),([], 'NEEDS_CALIBRATION'))
        red=np.zeros((180,320,3),np.uint8); cv2.circle(red,(160,40),5,(0,0,255),-1)
        green=np.zeros((180,320,3),np.uint8); cv2.circle(green,(160,40),5,(0,255,0),-1)
        self.assertEqual(observe(detect_lights(red)),'RED')
        self.assertEqual(observe(detect_lights(green)),'GREEN')
        self.assertEqual(observe(detect_lights(np.zeros((180,320,3),np.uint8))),'OFF')
        daylight=np.full((180,320,3),180,np.uint8); cv2.circle(daylight,(160,40),5,(0,0,255),-1)
        self.assertEqual(observe(detect_lights(daylight)),'OFF')  # 白天不走夜景色块
        self.assertEqual(color_of(daylight,(150,30,170,50)),'RED')
        self.assertEqual(observe(detect_lights(daylight,[(150,30,170,50)])),'RED')
        machine=SignalMachine(hold=3)
        self.assertEqual(machine.update('RED',0)['color'],'UNKNOWN')
        self.assertEqual(machine.update('RED',.5)['color'],'UNKNOWN')
        entered=machine.update('RED',1)
        self.assertEqual(entered['color'],'RED')
        self.assertTrue(entered['stable'])
        self.assertEqual(entered['since_seconds'],0)
        self.assertEqual(machine.update('GREEN',1.5)['color'],'RED')  # 单帧不得翻转
        self.assertEqual(machine.update('GREEN',2)['color'],'RED')
        flipped=machine.update('GREEN',2.5)
        self.assertEqual(flipped['color'],'GREEN')
        self.assertEqual(machine.update('OFF',3)['color'],'GREEN')
        self.assertEqual(machine.update('OFF',3.5)['color'],'GREEN')
        ended=machine.update('OFF',4)
        self.assertEqual(ended['color'],'GREEN')
        self.assertEqual(ended['at_end'],'OFF')
        waiting=[{'time_seconds':i*.5,'signal_observed':'RED',
                  'vehicles':[{'track_id':1,'score':.9,'box_normalized':[.4,.4,.6,.62]}]} for i in range(6)]
        self.assertFalse(red_approach(waiting)['approaching'])
        closing=[{'time_seconds':i*.5,'signal_observed':'RED',
                  'vehicles':[{'track_id':1,'score':.9,'box_normalized':[.4,.4,.6,.5+i*.04]}]} for i in range(6)]
        found=red_approach(closing)
        self.assertTrue(found['approaching'])
        self.assertTrue(found['proceeding'])
        self.assertEqual(found['track_id'],1)
        self.assertIsNone(red_approach(waiting[:2]))
        leaving=[{'time_seconds':i*.5,'signal_observed':'RED',
                  'vehicles':[{'track_id':1,'score':.9,'box_normalized':[.4,.4,.6,.7-i*.04]}]} for i in range(6)]
        self.assertTrue(red_approach(leaving)['proceeding'])
        self.assertFalse(red_approach(leaving)['approaching'])
        plates=[{'text':'川A10001','stable':True,'hits':4,'confidence':.99}]
        hit=red_light_candidate({'color':'RED','stable':True,'since_seconds':0},found,plates)
        self.assertEqual(hit['type'],'RED_LIGHT')
        self.assertEqual(hit['plate'],'川A10001')
        self.assertEqual(red_light_candidate({'color':'RED','stable':True},red_approach(leaving),plates)['type'],'RED_LIGHT')
        self.assertIsNone(red_light_candidate({'color':'RED','stable':True},red_approach(waiting),plates))
        self.assertIsNone(red_light_candidate({'color':'RED','stable':True},found,[]))
        # Hood-sized boxes must not beat the car ahead; plate stitches tracker fragments.
        hood=[{'time_seconds':i*.5,'signal_observed':'RED',
               'vehicles':[{'track_id':1,'score':.9,'box_normalized':[.4,.42,.58,.55+i*.04]},
                           {'track_id':99,'score':.4,'box_normalized':[.15,.74,.92,.99]}],
               'plates':[{'text':'川A10001','track_id':10+i,'box_normalized':[.46,.5,.52,.54]}]} for i in range(6)]
        self.assertEqual(red_approach(hood)['track_id'],1)
        self.assertTrue(red_approach(hood,'川A10001')['approaching'])
        late=[{'time_seconds':12+i*.5,'signal_observed':'RED',
               'vehicles':[{'track_id':83,'score':.5,'box_normalized':[.2,.55,.85,.99]}]} for i in range(6)]
        self.assertTrue(red_approach(closing+late)['approaching'])
        self.assertEqual(red_approach(closing+late)['track_id'],1)
        mixed=np.zeros((180,320,3),np.uint8)
        cv2.circle(mixed,(160,40),5,(0,255,0),-1)
        cv2.circle(mixed,(300,40),5,(0,0,255),-1)
        self.assertEqual(observe(detect_lights(mixed)),'GREEN')
        self.assertEqual(observe([
            {'color':'GREEN','center':[.14,.45],'score':1,'area':10,'source':'BOX'},
            {'color':'RED','center':[.46,.46],'score':1,'area':10,'source':'BOX'},
        ]),'RED')
        self.assertEqual(observe([
            {'color':'GREEN','center':[.5,.4],'score':1,'area':10,'source':'BOX'},
            {'color':'RED','center':[.9,.4],'score':1,'area':10,'source':'BOX'},
        ]),'GREEN')
        self.assertEqual(observe([
            {'color':'RED','center':[.91,.4],'score':10000,'area':10,'source':'BOX'},
        ]),'OFF')
        self.assertEqual(observe([
            {'color':'YELLOW','center':[.5,.4],'score':10000,'area':10,'source':'BOX'},
        ]),'OFF')
        self.assertEqual(observe([
            {'color':'GREEN','center':[.50,.476],'score':1,'area':10,'source':'HSV_GLOW'},
            {'color':'RED','center':[.42,.456],'score':1,'area':10,'source':'HSV_GLOW'},
        ]),'RED')
        split=[{'time_seconds':i*.5,'plates':(
            [{'text':'川A10001','confidence':.99,'box_normalized':[.44,.58,.48,.61]}]
            + ([{'text':'冀A10001','confidence':.995,'box_normalized':[.44,.58,.48,.61]}] if i<5 else [])
        )} for i in range(7)]
        ranked=consensus(split)
        self.assertEqual(ranked[0]['text'],'川A10001')
        self.assertEqual(ranked[0]['hits'],12)
        self.assertTrue(all(p['text']!='冀A10001' for p in ranked))
        slide=[{'time_seconds':i*.5,'vehicles':[{'track_id':7,'score':.9,
            'box_normalized':[.52-i*.03,.45,.66-i*.03,.62]}]} for i in range(6)]
        self.assertEqual(lane_changes(slide)[0]['type'],'LATERAL_MOVEMENT')
        self.assertEqual(lane_changes(slide)[0]['time_seconds'],1.5)
        two=[]
        for t0 in (0,20):
            two+=[{'time_seconds':t0+i*.5,'vehicles':[{'track_id':7,'score':.9,
                'box_normalized':[.52-i*.03,.45,.66-i*.03,.62]}]} for i in range(6)]
        self.assertEqual([x['time_seconds'] for x in lane_changes(two)],[1.5,21.5])
        both=[]
        for i in range(6):
            both.append({'time_seconds':i*.5,'signal_observed':'RED','vehicles':[
                {'track_id':1,'score':.9,'box_normalized':[.4,.4,.6,.5+i*.04]},
                {'track_id':2,'score':.9,'box_normalized':[.22,.4,.38,.52+i*.04]}],
                'plates':[{'text':'川A10001','track_id':1,'box_normalized':[.46,.5,.52,.54]},
                          {'text':'川B12345','track_id':2,'box_normalized':[.28,.48,.34,.52]}]})
        self.assertEqual({item['plate'] for item in red_approaches(both,['川A10001','川B12345'])},
                         {'川A10001','川B12345'})
        laterals=[{'type':'SOLID_LINE','track_id':2,'time_seconds':5.0},
                  {'type':'SOLID_LINE','track_id':3,'time_seconds':5.5}]
        red_frames=[{'time_seconds':t,'signal_observed':'RED'} for t in (4.5,5.0,5.5)]
        self.assertEqual({item['track_id'] for item in red_during_laterals(red_frames,laterals)},{2,3})
        self.assertEqual(red_during_laterals(red_frames,laterals,{2,3}),[])
        turning=[{'time_seconds':t,'signal_observed':('GREEN' if t<5 else 'RED'),
                  'vehicles':[{'track_id':4,'box_normalized':[.4+t*.03,.4,.55+t*.03,.6]}]}
                 for t in (2.0,3.0,4.0,4.5,5.0,5.5,6.0)]
        self.assertTrue(turned_from_green(turning,4,5.0))
        self.assertEqual(red_during_laterals(turning,[{'type':'SOLID_LINE','track_id':4,'time_seconds':5.0}]),[])
        plated={'plate':'川A10001','type':'RED_LIGHT','track_id':1,'time_seconds':12}
        red=[{'time_seconds':t,'signal_observed':'RED',
              'vehicles':[{'track_id':1,'box_normalized':[.4,.4,.6,.5]}],
              'plates':[{'text':'川A10001','track_id':1}]} for t in (8,9,10,11,12)]
        self.assertEqual(clip_windows([{**plated,'time_seconds':10},{**plated,'time_seconds':12,'type':'SOLID_LINE'}],20),[])
        self.assertEqual(clip_windows([{'time_seconds':78,'clip_until':111}],180),[])
        span=clip_windows([plated],20,red)
        self.assertEqual(len(span),1)
        self.assertAlmostEqual(span[0]['start'],7.5,places=1)
        self.assertAlmostEqual(span[0]['end'],12.5,places=1)
        cluster=lambda t0: [{'time_seconds':t0+i,'signal_observed':'RED',
                             'vehicles':[{'track_id':1,'box_normalized':[.4,.4,.6,.5]}],
                             'plates':[{'text':'川A10001','track_id':1}]} for i in range(3)]
        split=clip_windows([{**plated,'time_seconds':78},{**plated,'time_seconds':103,'type':'SOLID_LINE'}],180,
                           cluster(76)+cluster(102))
        self.assertEqual([(round(w['start'],1),round(w['end'],1)) for w in split],[(75.5,78.5),(101.5,104.5)])
        self.assertEqual(clip_windows([plated],20,red),span)
        self.assertEqual(clip_windows([{**plated,'time_seconds':40}],80,red),[])
        self.assertEqual([v['plate'] for v in bind_plates(
            [{'type':'SOLID_LINE','track_id':1},{'type':'RED_LIGHT','track_id':2,'plate':'京AM0H772'}],
            [{'plates':[{'text':'冀A10001','track_id':1}]}, {'plates':[{'text':'川A10001','track_id':1}]}],
            [{'text':'川A10001','stable':True}])], ['川A10001'])
        self.assertEqual(bind_plates([{'type':'RED_LIGHT','track_id':2,'plate':'川A10001'}],
            [{'plates':[{'text':'川A10001','track_id':1}]} for _ in range(3)],
            [{'text':'川A10001','stable':True}]), [])
        self.assertEqual(bind_plates([{'type':'SOLID_LINE','track_id':9,'plate':'京AM0H772'}],
            [{'plates':[{'text':'京AM0H772','track_id':9}]}],
            [{'text':'京AM0H772','stable':False}]), [])
        park=[{'time_seconds':i*.5,'vehicles':[
            {'track_id':9,'score':.9,'box_normalized':[.70,.50,.86,.68]},
            {'track_id':1,'score':.9,'box_normalized':[.20,.40,.35,.55+i*.04]}]} for i in range(8)]
        self.assertEqual(restricted_park(park)[0]['type'],'RESTRICTED_LANE')
        frozen=[{'time_seconds':i*.5,'vehicles':[{'track_id':9,'score':.9,
            'box_normalized':[.70,.50,.86,.68]}]} for i in range(8)]
        self.assertEqual(restricted_park(frozen),[])
        drifting=[{'time_seconds':i*.5,'vehicles':[
            {'track_id':9,'score':.9,'box_normalized':[.70,.50,.86,.68+i*.05]},
            {'track_id':1,'score':.9,'box_normalized':[.20,.40,.35,.55]}]} for i in range(8)]
        self.assertEqual(restricted_park(drifting),[])
        bloom=np.zeros((180,320,3),np.uint8)
        cv2.circle(bloom,(160,40),8,(0,255,255),-1)
        cv2.circle(bloom,(160,40),8,(0,0,255),2)
        self.assertEqual(observe(detect_lights(bloom)),'RED')
        lock=ThroughLamp()
        far=[{'color':'RED','center':[.78,.36],'score':1,'area':10,'source':'BOX'}]
        self.assertEqual(lock.update(far),'RED')
        self.assertEqual(lock.update([]),'RED')
        other=[{'color':'RED','center':[.91,.36],'score':1,'area':10,'source':'BOX'}]
        self.assertEqual(ThroughLamp().update(other),'OFF')
        mixed_lock=ThroughLamp()
        self.assertEqual(mixed_lock.update([
            {'color':'RED','center':[.48,.44],'score':1,'area':10,'source':'BOX'},
            {'color':'GREEN','center':[.14,.44],'score':1,'area':10,'source':'BOX'},
        ]),'RED')
        self.assertEqual(mixed_lock.update([
            {'color':'GREEN','center':[.33,.46],'score':1,'area':10,'source':'BOX'},
        ]),'RED')

    def test_upload_inference_recovery_and_cleanup(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(data=Path(temporary), model=root / "models/yolox_s.onnx",
                token="local-test-token-do-not-deploy", reserve_bytes=0)
            app = create_app(settings)
            store = app.state.store
            detector = Detector(settings)
            # This queue/vehicle test deliberately does not run the separate plate reader.
            configuration = store.configuration()
            store.configure(configuration['revision'], {**configuration['config'], 'plate_enabled': False})
            content = (root / "tests/traffic.mp4").read_bytes()
            event = {"event_id": str(uuid.uuid4())}

            def upload(client, payload=content, metadata=event, **headers):
                return client.post("/v1/tasks", content=payload, headers={
                    "Authorization": f"Bearer {settings.token}", "Content-Type": "video/mp4",
                    "X-Event-Metadata": json.dumps(metadata),
                    "X-Video-SHA256": hashlib.sha256(payload).hexdigest(), **headers})

            auth = {"Authorization": f"Bearer {settings.token}"}
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertTrue(client.get("/health").json()["auth"])
                self.assertEqual(client.get("/v1/tasks").status_code, 401)
                self.assertEqual(upload(client, **{"X-Video-SHA256": "0" * 64}).status_code, 422)
                self.assertEqual(upload(client, metadata={"event_id": "../../bad"}).status_code, 422)
                self.assertEqual(upload(client, metadata={**event, "mobile_confidence": -1}).status_code, 422)
                original_limit = settings.max_bytes
                settings.max_bytes = 8
                self.assertEqual(upload(client).status_code, 413)
                settings.max_bytes = original_limit
                response = upload(client)
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["task_id"]
                duplicate = upload(client)
                self.assertEqual(duplicate.status_code, 200)
                self.assertEqual(duplicate.json()["task_id"], task_id)
                # v1 records predate the optional trigger and scene fields.
                with store.connection() as db:
                    old=store.get(task_id)['metadata']
                    for key in ('trigger','trigger_text','scene','capture'): old.pop(key)
                    db.execute('UPDATE tasks SET metadata=? WHERE task_id=?',(json.dumps(old),task_id))
                self.assertEqual(upload(client).status_code,200)
                self.assertEqual(upload(client, metadata={**event, "candidate_type": "RED_LIGHT"}).status_code, 409)
                self.assertTrue(process_one(store, detector, "test-worker"))
                task = client.get(f"/v1/tasks/{task_id}", headers=auth).json()
                self.assertEqual(task["status"], "ANALYZED")
                self.assertEqual(task["result"]["decision"], "UNKNOWN")
                self.assertFalse(task["result"]["submission_allowed"])
                self.assertGreater(task["result"]["vehicle_observations"]["car"], 0)
                self.assertEqual(task["result"]["sampled_frames"], 6)
                self.assertEqual(task["result"]["track_count"], len(task["result"]["frames"][0]["vehicles"]))
                print("REAL_MODEL", {k: v for k, v in task["result"].items() if k != "frames"})
                self.assertEqual(client.post(f"/v1/tasks/{task_id}/retry", headers=auth).status_code, 409)

                pending = upload(client, metadata={"event_id": str(uuid.uuid4())}).json()["task_id"]
                self.assertEqual(store.claim("crashed-worker")["task_id"], pending)
                self.assertEqual(client.delete(f"/v1/tasks/{pending}", headers=auth).status_code, 409)
                with store.connection() as db:
                    db.execute("UPDATE tasks SET lease_until=0 WHERE task_id=?", (pending,))
                recovered = Store(settings)
                self.assertTrue(process_one(recovered, detector, "recovered-worker"))
                self.assertEqual(recovered.get(pending)["attempts"], 2)
                self.assertFalse(recovered.finish(pending, "crashed-worker", "ERROR"))
                self.assertEqual(recovered.get(pending)["status"], "ANALYZED")

                corrupt = upload(client, b"not a video", {"event_id": str(uuid.uuid4())}).json()["task_id"]
                process_one(store, detector, "test-worker")
                self.assertEqual(store.get(corrupt)["status"], "REJECTED")
                self.assertEqual(client.delete(f"/v1/tasks/{task_id}", headers=auth).status_code, 200)
                self.assertFalse(store.video(task_id).exists())
                self.assertEqual(store.get(task_id)["status"], "EXPIRED")
                with store.connection() as db:
                    db.execute("UPDATE tasks SET expires_at=? WHERE task_id=?", (time.time() - 1, pending))
                store.cleanup()
                self.assertFalse(store.video(pending).exists())
                self.assertEqual(store.get(pending)["status"], "EXPIRED")
                self.assertEqual(len(list((settings.data / "videos").glob("*.part"))), 0)

    def test_archive_clears_all_records(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            settings = Settings(data=Path(temporary), model=root / "models/yolox_s.onnx",
                token="local-test-token-do-not-deploy", reserve_bytes=0)
            app = create_app(settings)
            store = app.state.store
            content = (root / "tests/traffic.mp4").read_bytes()
            auth = {"Authorization": f"Bearer {settings.token}"}
            with TestClient(app) as client:
                task_id = client.post("/v1/tasks", content=content, headers={
                    **auth, "Content-Type": "video/mp4",
                    "X-Event-Metadata": json.dumps({"event_id": str(uuid.uuid4())}),
                    "X-Video-SHA256": hashlib.sha256(content).hexdigest()}).json()["task_id"]
                self.assertTrue(store.video(task_id).exists())
                self.assertEqual(client.post("/v1/archive", headers=auth).json()["archived"], 1)
                self.assertEqual(client.get("/v1/tasks", headers=auth).json()["tasks"], [])
                self.assertFalse(store.video(task_id).exists())
                self.assertIsNone(store.get(task_id))
                pair=client.post('/v1/device-pair',headers={**auth,'X-Requested-With':'traffic-console'})
                self.assertEqual(pair.status_code,200)
                code=pair.json()['code']
                self.assertEqual(len(code),8)
                event={"event_id": str(uuid.uuid4())}
                again=client.post("/v1/tasks", content=content, headers={
                    "Authorization": f"Bearer {code}", "Content-Type": "video/mp4",
                    "X-Event-Metadata": json.dumps(event),
                    "X-Video-SHA256": hashlib.sha256(content).hexdigest()})
                self.assertEqual(again.status_code,202)

    def test_hello_handshake_two_clients_record_model_and_ip(self):
        from hello import code as hello_code
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data=Path(directory), model=root / "models/yolox_s.onnx",
                token="unused-open-mode-token-value", open=True, reserve_bytes=0)
            app = create_app(settings)

            def handshake(client, device, platform, model, ip="203.0.113.10"):
                ts, nonce = int(time.time()), secrets.token_hex(16)
                body = {"device_id": device, "platform": platform, "model": model, "app_version": "2.0",
                        "ts": ts, "nonce": nonce, "code": hello_code(device, platform, ts, nonce)}
                return client.post("/v1/hello", json=body, headers={"X-Forwarded-For": ip})

            with TestClient(app) as one, TestClient(app) as two:
                self.assertEqual(one.get("/health").json()["auth"], "hello")
                self.assertEqual(one.get("/v1/settings").status_code, 401)
                self.assertEqual(one.post("/v1/hello", json={
                    "device_id": "device-android-1", "platform": "android", "model": "Xiaomi M2007J3SC",
                    "app_version": "2.0", "ts": int(time.time()), "nonce": secrets.token_hex(16),
                    "code": "0" * 64}).status_code, 401)
                phone = handshake(one, "device-android-1", "android", "Xiaomi M2007J3SC", "198.51.100.20")
                web = handshake(two, "device-web-00002", "web", "Chrome Windows", "198.51.100.30")
                self.assertEqual(phone.status_code, 200)
                self.assertEqual(web.status_code, 200)
                phone_auth = {"Authorization": "Bearer " + phone.json()["session"], "X-Forwarded-For": "198.51.100.20"}
                web_auth = {"Authorization": "Bearer " + web.json()["session"], "X-Forwarded-For": "198.51.100.30"}
                self.assertEqual(one.get("/v1/settings", headers=phone_auth).status_code, 200)
                self.assertEqual(two.get("/v1/overview", headers=web_auth).status_code, 200)
                clients = two.get("/v1/overview", headers=web_auth).json()["clients"]
                models = {row["model"]: row["ip"] for row in clients}
                self.assertEqual(models["Xiaomi M2007J3SC"], "198.51.100.20")
                self.assertEqual(models["Chrome Windows"], "198.51.100.30")
                self.assertEqual(sorted(row["platform"] for row in clients), ["android", "web"])

    def test_web_client_cors_and_hello_salt(self):
        from hello import SALT
        root = Path(__file__).resolve().parent
        self.assertIn(SALT, (root / "web" / "app.js").read_text(encoding="utf-8"))
        self.assertIn("61612", (root / "web" / "index.html").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(data=Path(directory), model=root / "models/yolox_s.onnx",
                token="unused-open-mode-token-value", open=True, reserve_bytes=0)
            app = create_app(settings)
            with TestClient(app) as client:
                preflight = client.options("/v1/hello", headers={
                    "Origin": "https://cam.muqin.ccwu.cc",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertIn(preflight.status_code, (200, 204))
                self.assertEqual(preflight.headers.get("access-control-allow-origin"), "https://cam.muqin.ccwu.cc")
                lan = client.options("/v1/hello", headers={
                    "Origin": "http://192.168.31.155:61612",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertIn(lan.status_code, (200, 204))
                self.assertEqual(lan.headers.get("access-control-allow-origin"), "http://192.168.31.155:61612")
                tls = client.options("/v1/hello", headers={
                    "Origin": "https://192.168.31.53:61612",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertIn(tls.status_code, (200, 204))
                self.assertEqual(tls.headers.get("access-control-allow-origin"), "https://192.168.31.53:61612")
        js = (root / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("function uuid()", js)
        self.assertIn("function sha256sync", js)
        self.assertIn("getUserMedia", js)
        self.assertIn("function clipBlob()", js)
        self.assertIn("function maybeAutoMark()", js)
        self.assertNotIn("chunks.filter", js)
        sample = "device\nweb\n1\nnonce\ntraffic-hello-v1"
        js_path = root / "web" / "app.js"
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "sha256check.js"
            runner.write_text(
                "const fs=require('fs');\n"
                "const src=fs.readFileSync(process.argv[2],'utf8');\n"
                "eval(src.slice(src.indexOf('function sha256sync'), src.indexOf('function toast')));\n"
                "process.stdout.write(sha256sync(Buffer.from(process.argv[3])) + ' ' + sha256sync(Buffer.from('abc')));\n",
                encoding="utf-8")
            digest = subprocess.check_output(
                ["node", str(runner), str(js_path), sample], text=True).strip().split()
        self.assertEqual(digest[0], hashlib.sha256(sample.encode()).hexdigest())
        self.assertEqual(digest[1], hashlib.sha256(b"abc").hexdigest())

    def test_tracker_does_not_assign_one_id_to_two_vehicles(self):
        tracker = Tracker()
        objects = lambda: [{"label": "car", "score": .9, "box": [0, 0, 20, 20]},
                           {"label": "car", "score": .8, "box": [10, 0, 30, 20]}]
        first = tracker.update(objects(), 0)
        second = tracker.update(objects(), 1)
        self.assertEqual([x["track_id"] for x in first], [x["track_id"] for x in second])
        self.assertEqual(len({x["track_id"] for x in second}), 2)
        tracker = Tracker()
        left={'label':'car','score':.9,'box':[200,100,300,200],'plate':'川A10001'}
        right={'label':'car','score':.9,'box':[400,100,500,200],'plate':'冀AEV8180'}
        ids={row['plate']:row['track_id'] for row in tracker.update([dict(left),dict(right)],0)}
        left['box']=[40,100,140,200]
        self.assertEqual({row['plate']:row['track_id'] for row in tracker.update([dict(left),dict(right)],1)}, ids)
        left['box']=[20,100,120,200]
        self.assertEqual({row['plate']:row['track_id'] for row in tracker.update([
            {'label':'car','score':.9,'box':list(left['box']),'plate':'川A10001'},
            {'label':'car','score':.9,'box':list(right['box']),'plate':'冀AEV8180'}],2)}, ids)
        self.assertEqual(len(set(ids.values())), 2)
        tracker = Tracker()
        dark=np.zeros(96,np.float32); dark[0]=1
        light=np.zeros(96,np.float32); light[20]=1
        first=tracker.update([
            {'label':'car','score':.9,'box':[100,80,180,160],'appearance':dark.copy()},
            {'label':'car','score':.9,'box':[200,80,280,160],'appearance':light.copy()}],0)
        dark_id=next(row['track_id'] for row in first if row['box'][0]==100)
        light_id=next(row['track_id'] for row in first if row['box'][0]==200)
        second=tracker.update([
            {'label':'car','score':.9,'box':[190,80,270,160],'appearance':dark.copy()},
            {'label':'car','score':.9,'box':[200,80,280,160],'appearance':light.copy()}],1)
        self.assertEqual({row['track_id'] for row in second}, {dark_id, light_id})

    def test_tracker_coasts_through_occlusion_without_stealing_waiting_car(self):
        tracker = Tracker()
        dark = np.zeros(96, np.float32); dark[0] = 1
        light = np.zeros(96, np.float32); light[20] = 1
        waiter = lambda: {'label': 'car', 'score': .9, 'box': [400, 200, 520, 320], 'appearance': light.copy()}
        passer_at = lambda x: {'label': 'car', 'score': .9, 'box': [x, 190, x + 120, 310], 'appearance': dark.copy()}
        first = tracker.update([waiter(), passer_at(80)], 0)
        waiter_id = next(row['track_id'] for row in first if row['box'][0] == 400)
        passer_id = next(row['track_id'] for row in first if row['box'][0] == 80)
        self.assertNotEqual(waiter_id, passer_id)
        for frame, x in enumerate((160, 240), start=1):
            ids = {row['track_id'] for row in tracker.update([waiter(), passer_at(x)], frame)}
            self.assertEqual(ids, {waiter_id, passer_id})
        for frame, x in enumerate((320, 400, 480, 560, 640), start=3):
            rows = tracker.update([passer_at(x)], frame)
            self.assertEqual([row['track_id'] for row in rows], [passer_id])
        resumed = tracker.update([waiter(), passer_at(720)], 8)
        self.assertEqual(next(row['track_id'] for row in resumed if row['box'][0] == 400), waiter_id)
        self.assertEqual(next(row['track_id'] for row in resumed if row['box'][0] == 720), passer_id)

    def test_tracker_keeps_id_when_view_changes_along_predicted_path(self):
        tracker = Tracker()
        rear = np.zeros(96, np.float32); rear[0] = 1
        side = np.zeros(96, np.float32); side[40] = 1
        first = tracker.update([{'label': 'car', 'score': .9, 'box': [80, 190, 200, 310], 'appearance': rear.copy()}], 0)
        tid = first[0]['track_id']
        for frame in range(1, 6):
            x = 80 + frame * 80
            look = rear if frame < 4 else side
            rows = tracker.update([{'label': 'car', 'score': .9, 'box': [x, 190, x + 120, 310], 'appearance': look.copy()}], frame)
            self.assertEqual(rows[0]['track_id'], tid)

    def test_red_light_follows_overtaking_path_not_waiting_car(self):
        waiter = {'track_id': 1, 'score': .9, 'box_normalized': [.42, .40, .58, .62]}
        frames = []
        for i in range(8):
            x = .10 + i * .06
            y2 = .50 + i * .03
            passer = {'track_id': 2, 'score': .9, 'box_normalized': [x - .08, .38, x + .08, y2]}
            plates = [{'text': '川AWAIT01', 'track_id': 1, 'box_normalized': [.48, .55, .54, .60]}]
            if i < 2:
                plates.append({'text': '川APASS01', 'track_id': 2,
                               'box_normalized': [x - .02, y2 - .04, x + .02, y2]})
            frames.append({'time_seconds': i * .5, 'signal_observed': 'RED',
                           'vehicles': [waiter, passer], 'plates': plates})
        waiting = red_approach(frames, '川AWAIT01')
        passing = red_approach(frames, '川APASS01')
        self.assertEqual(waiting['track_id'], 1)
        self.assertFalse(waiting['proceeding'])
        self.assertEqual(passing['track_id'], 2)
        self.assertTrue(passing['proceeding'])
        hit = red_light_candidate({'color': 'RED', 'stable': True, 'since_seconds': 0}, passing,
                                  [{'text': '川APASS01', 'stable': True}])
        self.assertEqual(hit['track_id'], 2)
        self.assertIsNone(red_light_candidate({'color': 'RED', 'stable': True}, waiting,
                                              [{'text': '川AWAIT01', 'stable': True}]))
        ego = []
        for i in range(8):
            grow = i * .008
            ego.append({'time_seconds': i * .5, 'signal_observed': 'RED', 'vehicles': [
                {'track_id': 1, 'score': .9, 'box_normalized': [.42, .40, .58, .58 + grow]},
                {'track_id': 2, 'score': .9, 'box_normalized': [.20, .42, .34, .56 + grow]},
            ], 'plates': [{'text': '川AWAIT01', 'track_id': 1, 'box_normalized': [.48, .52, .54, .57]}]})
        self.assertFalse(red_approach(ego, '川AWAIT01')['proceeding'])


if __name__ == "__main__":
    unittest.main()
