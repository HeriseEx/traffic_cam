import asyncio
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import time
import hmac
import threading
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import Settings
from store import Conflict, Store
from schemas import AnalysisConfig, Scene, Review, Reanalyze, SettingsUpdate, Submission, MobileCapture
from model_catalog import catalog, plate_catalog, plate_installed, model_directory, PLATE_FILES as FILES, PLATE_MODELS
from plates import PlateReader
from signals import detect_lights, observe
from hello import ok as hello_ok


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: UUID
    candidate_type: Literal["UNKNOWN", "RED_LIGHT", "SOLID_LINE", "WRONG_WAY", "RESTRICTED_LANE"] = "UNKNOWN"
    mobile_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    model_version: str = Field(default="manual-test", max_length=100)
    app_version: str = Field(default="1", max_length=40)
    manual_review: bool = True
    trigger: Literal['manual', 'voice', 'automatic', 'import'] = 'import'
    trigger_text: str = Field(default='', max_length=160)
    scene: Scene | None = None
    capture: MobileCapture | None = None


class Hello(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device_id: str = Field(min_length=8, max_length=80)
    platform: Literal["android", "ios", "web"]
    model: str = Field(min_length=1, max_length=120)
    app_version: str = Field(default="2.0", max_length=40)
    ts: int
    nonce: str = Field(min_length=16, max_length=64)
    code: str = Field(min_length=64, max_length=64)


def peer_ip(request: Request):
    for key in ("x-real-ip", "x-forwarded-for"):
        raw = (request.headers.get(key) or "").split(",")[0].strip()
        if raw:
            return raw[:64]
    return (request.client.host if request.client else "")[:64]


def create_app(settings=None):
    settings = settings or Settings()
    store = Store(settings)
    plate_lock = threading.Lock()
    live_reader = None
    live_detector = None
    live_detect_key = None
    previews_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        settings.require_token()
        yield

    app = FastAPI(title="Traffic verification", version="0.2.0", lifespan=lifespan)
    app.state.store = store

    def authorized(request: Request, authorization: str = Header(default="")):
        ip = peer_ip(request)
        got = authorization.encode()
        expected = f"Bearer {settings.token}".encode()
        if settings.token and len(got) == len(expected) and secrets.compare_digest(got, expected):
            return
        offered = authorization[7:].strip() if authorization.startswith('Bearer ') else ''
        if offered and store.client_ok(offered, ip):
            return
        if offered and store.pair_ok(offered):
            return
        if store.client_ok(request.cookies.get('traffic_client', ''), ip):
            if request.method not in ('GET', 'HEAD') and request.headers.get('x-requested-with') != 'traffic-console':
                raise HTTPException(403, '缺少同源请求标识')
            return
        cookie=request.cookies.get('traffic_session','')
        try:
            stamp,nonce,signature=cookie.split('.')
            valid=time.time()<int(stamp) and hmac.compare_digest(signature,
                hmac.new(settings.token.encode(),f'{stamp}.{nonce}'.encode(),'sha256').hexdigest())
        except (ValueError,TypeError):
            valid=False
        if not settings.token or not valid:
            raise HTTPException(401, "请登录或检查访问令牌", headers={"WWW-Authenticate": "Bearer"})
        if request.method not in ('GET','HEAD') and request.headers.get('x-requested-with')!='traffic-console':
            raise HTTPException(403,'缺少同源请求标识')

    @app.exception_handler(Conflict)
    async def conflict(request,error):
        return JSONResponse({'detail':str(error)},status_code=409)

    @app.middleware('http')
    async def response_headers(request,call_next):
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        if request.url.path.startswith('/v1'):
            response.headers['Cache-Control']='no-store'
        return response

    @app.post('/v1/session')
    def login(request:Request,authorization:str=Header(default='')):
        if not settings.token or not secrets.compare_digest(authorization.encode(),f'Bearer {settings.token}'.encode()):
            raise HTTPException(401,'访问令牌不正确')
        value=f'{int(time.time()+8*3600)}.{secrets.token_hex(16)}'
        value+='.'+hmac.new(settings.token.encode(),value.encode(),'sha256').hexdigest()
        response=JSONResponse({'ok':True})
        response.set_cookie('traffic_session',value,httponly=True,samesite='strict',secure=request.url.scheme=='https',max_age=8*3600)
        return response

    @app.delete('/v1/session',dependencies=[Depends(authorized)])
    def logout(request: Request):
        store.revoke_session(request.cookies.get('traffic_client', ''))
        response=JSONResponse({'ok':True})
        response.delete_cookie('traffic_session');response.delete_cookie('traffic_client')
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "traffic-api", "version": "0.2.0", "auth": "hello", "capture_metadata": True}

    @app.post("/v1/hello")
    def hello(request: Request, body: Hello):
        if not hello_ok(body.device_id, body.platform, body.ts, body.nonce, body.code):
            raise HTTPException(401, "握手校验失败")
        device = body.device_id
        previous = request.cookies.get('traffic_client', '') if body.platform == 'web' else ''
        if body.platform == 'web':
            # Device identity survives a new login and localStorage loss. It is not an authentication credential.
            saved = request.cookies.get('traffic_device', '')
            try:
                identity, signature = saved.rsplit('.', 1)
                expected = hmac.new(settings.token.encode(), f'device:{identity}'.encode(), 'sha256').hexdigest()
                if 8 <= len(identity) <= 80 and hmac.compare_digest(signature, expected):
                    device = identity
                else:
                    device = store.session_device(previous) or device
            except ValueError:
                device = store.session_device(previous) or device
        result = store.hello(device, body.platform, body.model.strip(), body.app_version, peer_ip(request), previous)
        response = JSONResponse(result)
        if body.platform == 'web':
            secure = request.url.scheme == 'https'
            response.set_cookie('traffic_client', result['session'], httponly=True, samesite='lax', secure=secure, max_age=30*86400)
            signature = hmac.new(settings.token.encode(), f'device:{device}'.encode(), 'sha256').hexdigest()
            response.set_cookie('traffic_device', f'{device}.{signature}', httponly=True, samesite='lax', secure=secure, max_age=365*86400)
        return response

    @app.post("/v1/tasks", dependencies=[Depends(authorized)])
    async def upload(request: Request, x_event_metadata: str = Header(), x_video_sha256: str = Header()):
        if len(x_event_metadata) > 12288:
            raise HTTPException(400, "Metadata too large")
        try:
            metadata = Event.model_validate_json(x_event_metadata).model_dump(mode="json")
        except ValidationError:
            raise HTTPException(422, "Invalid event metadata") from None
        if len(x_video_sha256) != 64 or any(c not in "0123456789abcdef" for c in x_video_sha256):
            raise HTTPException(400, "X-Video-SHA256 must be a lowercase SHA-256")
        if shutil.disk_usage(settings.data).free < settings.reserve_bytes + settings.max_bytes:
            raise HTTPException(507, "Insufficient free disk space")
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > settings.max_bytes):
            raise HTTPException(413, "Video exceeds upload limit")
        digest = hashlib.sha256()
        total = 0
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=settings.data / "videos", suffix=".part", delete=False) as stream:
                from pathlib import Path
                temporary = Path(stream.name)
                async with asyncio.timeout(600):
                    async for chunk in request.stream():
                        total += len(chunk)
                        if total > settings.max_bytes:
                            raise HTTPException(413, "Video exceeds upload limit")
                        digest.update(chunk)
                        await asyncio.to_thread(stream.write, chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if total == 0:
                raise HTTPException(400, "Empty video")
            if digest.hexdigest() != x_video_sha256:
                raise HTTPException(422, "Video SHA-256 mismatch")
            task, created = await asyncio.to_thread(store.add, metadata, digest.hexdigest(), total, temporary)
            return JSONResponse(task, status_code=202 if created else 200)
        except TimeoutError:
            raise HTTPException(408, "Upload timed out") from None
        except Conflict as error:
            raise HTTPException(409, str(error)) from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @app.get("/v1/tasks", dependencies=[Depends(authorized)])
    def tasks(limit: int = 30, offset:int=0,status:str='',review:str='',query:str=''):
        if not 1 <= limit <= 100 or not 0<=offset<=100000 or len(query)>100:
            raise HTTPException(422, "limit must be between 1 and 100")
        return {"tasks": store.list(limit,offset,status,review,query)}

    @app.get('/v1/overview',dependencies=[Depends(authorized)])
    def overview():
        return store.overview()

    @app.get('/v1/changes',dependencies=[Depends(authorized)])
    def changes(after:int=0):
        if after<0: raise HTTPException(422,'无效同步游标')
        return store.changes(after)

    @app.get('/v1/settings',dependencies=[Depends(authorized)])
    def configuration():
        raw=store.configuration()
        config=AnalysisConfig.model_validate(raw['config']).model_dump()
        return {**raw,'config':config,'models':catalog(settings),'plate_models':plate_catalog(settings)}

    def check_models(config):
        if not next(m for m in catalog(settings) if m['id']==config.vehicle_model)['installed']:
            raise HTTPException(422,'模型尚未下载')
        if config.plate_enabled and not plate_installed(model_directory(settings)/'plate', config.plate_model):
            raise HTTPException(422,'车牌模型尚未下载')
        nonlocal live_reader
        live_reader = None

    @app.put('/v1/settings',dependencies=[Depends(authorized)])
    def configure(body:SettingsUpdate):
        check_models(body.config)
        return store.configure(body.expected_revision,body.config.model_dump())

    @app.post('/v1/recognize-frame',dependencies=[Depends(authorized)])
    async def recognize_frame(request:Request):
        config=AnalysisConfig.model_validate(store.configuration()['config'])
        want_vehicles=request.query_params.get('vehicles','').lower() in ('1','true','yes')
        data=bytearray()
        async with asyncio.timeout(15):
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data)>4*1024*1024: raise HTTPException(413,'图像超过 4 MiB')
        def recognize():
            nonlocal live_reader, live_detector, live_detect_key
            import cv2,numpy as np
            if not plate_lock.acquire(timeout=10): raise HTTPException(429,'车牌识别忙，请稍后重试')
            try:
                image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape[0]*image.shape[1]>3840*2160:
                    raise HTTPException(422,'图像无效或分辨率超限')
                vehicles,lamp_boxes=[],[]
                if want_vehicles:
                    from inference import Detector
                    height,width=image.shape[:2]
                    try:
                        key=(config.vehicle_model,config.vehicle_threshold)
                        if live_detector is None or live_detect_key!=key:
                            live_detector=Detector(settings,config)
                            live_detect_key=key
                        raw=live_detector.detect(image)
                        lamp_boxes=[item['box'] for item in live_detector.lamps]
                        vehicles=[{'label':item['label'],'score':item['score'],
                            'box_normalized':[round(item['box'][0]/width,5),round(item['box'][1]/height,5),
                                              round(item['box'][2]/width,5),round(item['box'][3]/height,5)]}
                            for item in raw]
                    except Exception:
                        vehicles,lamp_boxes=[],[]
                lights=detect_lights(image,lamp_boxes)
                plates=[]
                if config.plate_enabled:
                    if live_reader is None or getattr(live_reader,'rec_id',None)!=config.plate_model:
                        live_reader=PlateReader(model_directory(settings)/'plate',2,config.plate_model)
                    plates=live_reader.read(image,config.plate_threshold)
                return {'plates':plates,'lights':lights,'vehicles':vehicles,'signal_observed':observe(lights),
                        'enabled':config.plate_enabled,'model':PLATE_MODELS[config.plate_model]['name'],'at':time.time()}
            finally: plate_lock.release()
        return await asyncio.to_thread(recognize)

    @app.get("/v1/tasks/{task_id}", dependencies=[Depends(authorized)])
    def task(task_id: UUID):
        record = store.get(str(task_id))
        if not record:
            raise HTTPException(404, "Task not found")
        return record

    @app.get('/v1/tasks/{task_id}/audit',dependencies=[Depends(authorized)])
    def audit(task_id:UUID):
        return {'history':store.audit(str(task_id))}

    @app.post('/v1/tasks/{task_id}/review',dependencies=[Depends(authorized)])
    def review(task_id:UUID,body:Review):
        return store.intervene(str(task_id),body.model_dump())

    @app.post('/v1/tasks/{task_id}/reanalyze',dependencies=[Depends(authorized)])
    def reanalyze(task_id:UUID,body:Reanalyze):
        config=body.config or AnalysisConfig.model_validate(store.configuration()['config'])
        check_models(config)
        previous=store.get(str(task_id))
        scene=body.scene.model_dump() if body.scene else (previous or {}).get('scene')
        return store.reanalyze(str(task_id),body.expected_revision,config.model_dump(),scene)

    @app.post('/v1/tasks/{task_id}/submission-receipt',dependencies=[Depends(authorized)])
    def submitted(task_id:UUID,body:Submission):
        # This records a real external receipt; it does not submit to any reporting service.
        return store.submitted(str(task_id),body.expected_revision,body.receipt)

    @app.api_route('/v1/tasks/{task_id}/video',methods=['GET', 'HEAD'],dependencies=[Depends(authorized)])
    def video(task_id:UUID,original:bool=False):
        record=store.get(str(task_id));path=store.video(str(task_id))
        if not record or record['status']=='EXPIRED' or not path.is_file():
            raise HTTPException(404,'视频已过期或不存在')
        if original: return FileResponse(path,media_type='application/octet-stream',filename=f'{task_id}.mp4')
        preview=settings.data/'previews'/f'{task_id}.mp4'
        with previews_lock:
            if not preview.exists():
                preview.parent.mkdir(exist_ok=True)
                temporary=preview.with_suffix('.part.mp4')
                try:
                    duration=min(float((record.get('result') or {}).get('video',{}).get('duration_seconds') or 600),600)
                    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-protocol_whitelist','file,pipe',
                        '-i',str(path),'-map','0:v:0','-t',f'{duration:.3f}','-vf',
                        'scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2',
                        '-r','20','-an','-sn','-dn','-c:v','libx264','-threads','2','-preset','veryfast',
                        '-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],
                        capture_output=True,timeout=180,check=True)
                    temporary.replace(preview)
                except (subprocess.SubprocessError,OSError):
                    raise HTTPException(422,'无法生成浏览器预览，可下载原始视频') from None
                finally: temporary.unlink(missing_ok=True)
        return FileResponse(preview,media_type='video/mp4')

    @app.post("/v1/tasks/{task_id}/retry", dependencies=[Depends(authorized)])
    def retry(task_id: UUID):
        if not store.retry(str(task_id)):
            raise HTTPException(409, "Task is not retryable or retry limit reached")
        return store.get(str(task_id))

    @app.delete("/v1/tasks/{task_id}", dependencies=[Depends(authorized)])
    def delete(task_id: UUID):
        try:
            if not store.expire(str(task_id)):
                raise HTTPException(404, "Task not found")
        except Conflict as error:
            raise HTTPException(409, str(error)) from None
        return {"task_id": str(task_id), "status": "EXPIRED"}

    @app.post("/v1/device-pair", dependencies=[Depends(authorized)])
    def device_pair():
        return {"code": store.issue_pair(), "ttl_seconds": 1800}

    @app.post("/v1/archive", dependencies=[Depends(authorized)])
    def archive():
        return {"archived": store.archive_all()}

    @app.api_route("/v1/tasks/{task_id}/clips/{index}", methods=['GET', 'HEAD'], dependencies=[Depends(authorized)])
    def clip_file(task_id: UUID, index: int):
        if not 0 <= index <= 99:
            raise HTTPException(404, "裁剪片段不存在")
        record = store.get(str(task_id))
        path = store.clips_dir(str(task_id)) / f"{index}.mp4"
        if not record or record['status'] == 'EXPIRED' or not path.is_file():
            raise HTTPException(404, "裁剪片段不存在")
        return FileResponse(path, media_type="video/mp4")

    static=Path(__file__).resolve().parent/'static'
    if static.is_dir():
        app.mount('/static',StaticFiles(directory=static),name='static')
        @app.get('/')
        def panel():
            return FileResponse(static/'index.html',headers={'Content-Security-Policy':
                "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"})
    origins=[item.strip() for item in os.getenv('TRAFFIC_CORS',
        'http://127.0.0.1:61612,https://127.0.0.1:61612,http://localhost:61612,https://localhost:61612,https://cam.muqin.ccwu.cc').split(',') if item.strip()]
    lan=r'https://cam\.muqin\.ccwu\.cc|(https|http)://(localhost|127\.0\.0\.1)(:\d+)?|(https|http)://(192\.168(\.\d{1,3}){2}|10(\.\d{1,3}){3}|172\.(1[6-9]|2\d|3[0-1])(\.\d{1,3}){2})(:\d+)?'
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex='^('+lan+')$',
        allow_methods=['GET','POST','PUT','DELETE','OPTIONS'],
        allow_headers=['Authorization','Content-Type','X-Event-Metadata','X-Video-SHA256','X-Requested-With'], max_age=600)
    return app


app = create_app()
