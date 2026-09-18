"""Exercise actual Docker OCR on the upstream photo and a static video derivative."""
import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from local_api import URL

ROOT=Path(__file__).resolve().parent


def main():
    token=next(line.split('=',1)[1] for line in (ROOT/'.env').read_text().splitlines()
               if line.startswith('TRAFFIC_API_TOKEN='))
    def request(path,body=None,headers=None):
        req=urllib.request.Request(URL+path,data=body,
            headers={'Authorization':'Bearer '+token,**(headers or {})})
        with urllib.request.urlopen(req,timeout=30) as response: return json.load(response)
    photo=ROOT/'tests/plate.jpg'
    frame=request('/v1/recognize-frame',photo.read_bytes(),{'Content-Type':'image/jpeg'})
    assert '苏ED51712' in [p['text'] for p in frame['plates']],frame
    assert 'signal_observed' in frame and 'lights' in frame,frame
    clip=ROOT/'validation/plate-sample.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-loop','1','-i',str(photo),
        '-t','3','-r','10','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(clip)],check=True)
    content=clip.read_bytes()
    task=request('/v1/tasks',content,{'Content-Type':'video/mp4',
        'X-Event-Metadata':json.dumps({'event_id':str(uuid.uuid4()),'trigger':'import','app_version':'plate-smoke'}),
        'X-Video-SHA256':hashlib.sha256(content).hexdigest()})
    for _ in range(90):
        task=request('/v1/tasks/'+task['task_id'])
        if task['status'] not in ('QUEUED','PROCESSING'): break
        time.sleep(1)
    assert task['status']=='ANALYZED',task
    assert task['result']['plate']=='苏ED51712',task['result']['plates']
    assert not task['result']['submission_allowed']
    (ROOT/'validation/plate-result.json').write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'task_id':task['task_id'],'plate':task['result']['plate'],'frames':task['result']['sampled_frames']}))


if __name__=='__main__': main()
