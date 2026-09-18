import hashlib
import logging
import signal
import threading
import uuid

from config import Settings
from inference import Detector, InvalidVideo, analyze, extract_clips
from store import Store
from schemas import AnalysisConfig
from plates import PlateReader
from model_catalog import model_directory

log = logging.getLogger("traffic-worker")


def process_one(store, detector, owner, models=None):
    task = store.claim(owner)
    if task is None:
        return False
    task_id = task["task_id"]
    try:
        path = store.video(task_id)
        config=AnalysisConfig.model_validate(task.get('analysis_config') or {})
        plate_reader=None
        if models is not None:
            key=(config.vehicle_model,config.vehicle_threshold,config.threads)
            if models.get('key')!=key:
                models['detector']=Detector(store.settings,config)
                models['key']=key
            detector=models['detector']
            plate_key=(config.plate_model,min(config.threads,4))
            if config.plate_enabled and models.get('plate_key')!=plate_key:
                models['plate']=PlateReader(model_directory(store.settings)/'plate',min(config.threads,4),config.plate_model)
                models['plate_key']=plate_key
            plate_reader=models.get('plate') if config.plate_enabled else None
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != task["sha256"]:
                raise InvalidVideo("Stored video checksum mismatch")
        def heartbeat():
            store.worker_heartbeat('处理中 '+task_id)
            return store.heartbeat(task_id,owner)
        result = analyze(path, detector, store.settings, heartbeat,config,plate_reader,task.get('scene'))
        plated=any(p.get('stable') and p.get('text') for p in result.get('plates') or [])
        if config.plate_enabled and not plated:
            result.update(violations=[],clips=[],decision='REJECTED',reason='PLATE_UNCONFIRMED',
                          violation_type=None,plate=None,rule_assessment='PLATE_UNCONFIRMED')
            status='REJECTED'
        else:
            if result.get('violations'):
                try:
                    heartbeat()
                    result['clips'] = extract_clips(
                        path, result['violations'], store.clips_dir(task_id),
                        result.get('video', {}).get('duration_seconds') or 0,
                        result.get('frames'))
                except Exception:
                    log.exception("task=%s clip extract failed", task_id)
                    result['clips'] = []
                result['violations']=[item for item in result['violations'] if item.get('clip_index') is not None]
                if result['violations']:
                    result['plate']=result['violations'][0].get('plate')
                    result['violation_type']=result['violations'][0].get('type')
                    result['decision']='CANDIDATE'
                    result['reason']=('RED_LIGHT_CANDIDATE'
                                      if any(item.get('type')=='RED_LIGHT' for item in result['violations'])
                                      else 'GEOMETRIC_CANDIDATE')
                else:
                    result['clips']=[]
                    result['decision']='UNKNOWN'
                    result['violation_type']=None
            status = "ANALYZED" if result["vehicle_presence"] else "REJECTED"
        store.finish(task_id, owner, status, result)
        log.info("task=%s status=%s elapsed_ms=%s", task_id, status, result["elapsed_ms"])
    except InvalidVideo as error:
        store.finish(task_id, owner, "REJECTED", {
            "decision": "REJECTED", "submission_allowed": False, "reason": "INVALID_VIDEO",
            "detail": str(error), "plate": None, "violation_type": None,
        })
        log.warning("task=%s invalid video", task_id)
    except Exception:
        log.exception("task=%s processing failed", task_id)
        store.finish(task_id, owner, "ERROR", error="Processing failed; check worker logs and retry")
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    store = Store(settings)
    models={}
    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM"):
        signal.signal(getattr(signal, name), lambda *_: stop.set())
    owner = str(uuid.uuid4())
    log.info("Worker ready: automatic analysis / configurable CPU models")
    while not stop.is_set():
        store.cleanup()
        store.worker_heartbeat('等待任务')
        if not process_one(store, None, owner, models):
            stop.wait(2)


if __name__ == "__main__":
    main()
