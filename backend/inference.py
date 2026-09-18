"""YOLOX ONNX decoding follows Megvii's Apache-2.0 demo; see models/LICENSE-YOLOX.txt."""
import hashlib
import json
import subprocess
import tempfile
import threading
import time
from collections import Counter
from model_catalog import MODELS, PLATE_MODELS, model_directory, verify
from schemas import AnalysisConfig
from plates import consensus
import shutil
from pathlib import Path
from rules import MotionGate, candidates, red_approaches, red_light_candidate, lane_changes, restricted_park, red_during_laterals, turned_from_green
from signals import SignalMachine, ThroughLamp, detect_lights, observe

import cv2
import numpy as np
import onnxruntime as ort

MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LAMP = 9
LAMP_SCORE = 0.10  # ponytail: COCO class-9 is not a traffic-light net; lower than vehicle threshold so small night housings still box. Dedicated TL weights if arrows/housings still lose.


class InvalidVideo(Exception):
    pass


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _plate_key(text):
    return text[1:] if text and len(text) > 1 else text or ''


def appearance_of(image, box):
    """HSV hue/sat histogram of the box. Cheap stand-in for a ReID net."""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = image[max(0, y1):max(y2, y1 + 1), max(0, x1):max(x2, x1 + 1)]
    if crop.size < 16:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, None).astype(np.float32).reshape(-1)


def _app_dist(a, b):
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)))


def _predict(track, frame):
    dt = max(0, frame - track['frame'])
    vx, vy = track.get('vel', (0.0, 0.0))
    box = track['box']
    return [box[0] + vx * dt, box[1] + vy * dt, box[2] + vx * dt, box[3] + vy * dt]


def _parked(track, diag):
    vx, vy = track.get('vel', (0.0, 0.0))
    speed = (vx * vx + vy * vy) ** 0.5
    return track.get('hits', 0) >= 3 and speed < max(8.0, 0.04 * diag)


class Tracker:
    """One ID per vehicle. Coast constant-velocity through missed frames so a passer does not inherit a waiting car."""

    def __init__(self, max_age=12):
        # ponytail: 12 frames at 2fps ≈ 6s occlusion coast; embedding ReID if same-color cars still swap.
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age

    def update(self, detections, frame):
        self.tracks = {k: v for k, v in self.tracks.items() if frame - v['frame'] <= self.max_age}
        available = set(self.tracks)
        assigned = set()

        def take(index, track_id):
            detection = detections[index]
            previous = self.tracks[track_id]
            dt = max(1, frame - previous['frame'])
            vx = (detection['box'][0] - previous['box'][0] + detection['box'][2] - previous['box'][2]) / 2 / dt
            vy = (detection['box'][1] - previous['box'][1] + detection['box'][3] - previous['box'][3]) / 2 / dt
            ovx, ovy = previous.get('vel', (vx, vy))
            incoming, prior = detection.get('appearance'), previous.get('appearance')
            appearance = incoming if incoming is not None else prior
            if incoming is not None and prior is not None:
                appearance = 0.7 * prior + 0.3 * incoming
            detection['track_id'] = track_id
            self.tracks[track_id] = {
                'box': list(detection['box']), 'label': detection['label'], 'frame': frame,
                'vel': (0.6 * ovx + 0.4 * vx, 0.6 * ovy + 0.4 * vy),
                'plate': detection.get('plate') or previous.get('plate'),
                'appearance': appearance,
                'hits': previous.get('hits', 1) + 1,
            }
            available.discard(track_id)
            assigned.add(index)

        pairs = []
        for index, detection in enumerate(detections):
            cx, cy = _center(detection['box'])
            suffix = _plate_key(detection.get('plate'))
            for track_id in self.tracks:
                track = self.tracks[track_id]
                if track['label'] != detection['label']:
                    continue
                previous_suffix = _plate_key(track.get('plate'))
                if suffix and previous_suffix and suffix != previous_suffix:
                    continue
                predicted = _predict(track, frame)
                overlap = iou(detection['box'], predicted)
                px, py = _center(predicted)
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                diag = max(predicted[2] - predicted[0], detection['box'][2] - detection['box'][0], 1)
                plate_hit = bool(suffix) and suffix == previous_suffix
                look = _app_dist(track.get('appearance'), detection.get('appearance'))
                det_area = max((detection['box'][2] - detection['box'][0]) * (detection['box'][3] - detection['box'][1]), 1)
                track_area = max((predicted[2] - predicted[0]) * (predicted[3] - predicted[1]), 1)
                ratio = det_area / track_area
                if not plate_hit:
                    if _parked(track, diag) and dist > .35 * diag:
                        continue
                    if _parked(track, diag) and look > .5:
                        continue
                    # Viewpoint/occlusion can flip HSV; predicted position still owns the ID.
                    if look > .85 and dist > .4 * diag:
                        continue
                    # ponytail: HSV hist is not ReID; similar look may jump (left turn) but can still glue two same-color cars. Upgrade: embedding ReID.
                    similar = look <= .35
                    if not similar:
                        if overlap < .15 and dist > .5 * diag:
                            continue
                        if overlap < .15 and not .5 <= ratio <= 2.0:
                            continue
                    elif dist > 2.2 * diag or not .35 <= ratio <= 2.8:
                        continue
                cost = (.1 * (dist / diag) if plate_hit
                        else .55 * (dist / diag) + .25 * (1 - overlap) + .35 * look)
                pairs.append((cost, index, track_id))
        for _, index, track_id in sorted(pairs):
            if index in assigned or track_id not in available:
                continue
            take(index, track_id)
        for index, detection in enumerate(detections):
            if index in assigned:
                continue
            track_id = self.next_id
            self.next_id += 1
            detection['track_id'] = track_id
            self.tracks[track_id] = {
                'box': list(detection['box']), 'label': detection['label'], 'frame': frame,
                'vel': (0, 0), 'plate': detection.get('plate'),
                'appearance': detection.get('appearance'), 'hits': 1,
            }
        return detections


class Detector:
    def __init__(self, settings, config=None):
        config = config or AnalysisConfig(vehicle_threshold=settings.threshold, threads=settings.threads)
        self.spec = MODELS[config.vehicle_model]
        path = model_directory(settings)/self.spec['file']
        verify(path, self.spec['sha256'])
        self.size = self.spec['size']
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        anchors=sum((self.size//s)**2 for s in (8,16,32))
        if self.input.shape != [1, 3, self.size, self.size] or self.session.get_outputs()[0].shape != [1, anchors, 85]:
            raise ValueError("Unsupported YOLOX model contract")
        self.threshold = config.vehicle_threshold
        grids, strides = [], []
        for stride in (8, 16, 32):
            x, y = np.meshgrid(np.arange(self.size // stride), np.arange(self.size // stride))
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)
        self.lamps = []

    def detect(self, image):
        height, width = image.shape[:2]
        ratio = min(self.size / width, self.size / height)
        resized_width, resized_height = int(width * ratio), int(height * ratio)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[:resized_height, :resized_width] = cv2.resize(image, (resized_width, resized_height))
        # Official YOLOX v0.1.1rc0 expects BGR 0..255, no RGB swap or /255 normalization.
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        prediction = self.session.run(None, {self.input.name: tensor})[0][0]
        class_ids = prediction[:, 5:].argmax(axis=1)
        scores = prediction[:, 4] * prediction[np.arange(len(prediction)), class_ids + 5]
        finite = np.isfinite(prediction).all(axis=1)
        self.lamps = self._decode(prediction, scores,
            (class_ids == LAMP) & (scores >= LAMP_SCORE) & finite, width, height, ratio, LAMP_SCORE)
        mask = np.isin(class_ids, list(VEHICLES)) & (scores >= self.threshold) & finite
        vehicles = self._decode(prediction, scores, mask, width, height, ratio, self.threshold)
        labels = class_ids[mask]
        return [{'label': VEHICLES[int(labels[item['index']])], 'score': item['score'], 'box': item['box']}
                for item in vehicles]

    def _decode(self, prediction, scores, mask, width, height, ratio, threshold):
        if not np.any(mask):
            return []
        centers = (prediction[mask, :2] + self.grid[mask]) * self.strides[mask]
        sizes = np.exp(np.clip(prediction[mask, 2:4], -20, 20)) * self.strides[mask]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores[mask].tolist(), threshold, 0.45)
        ids = np.asarray(kept).reshape(-1)[:50]
        kept_scores = scores[mask]
        result = []
        for k in ids:
            if xywh[k, 2] <= 0 or xywh[k, 3] <= 0:
                continue
            result.append({'index': int(k), 'score': round(float(kept_scores[k]), 4),
                           'box': [round(float(v), 2) for v in boxes[k]]})
        return result


def probe(path, settings):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
            "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, timeout=15, check=True)
        info = json.loads(result.stdout)
        video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
        duration = float(info["format"].get("duration", video.get("duration", "nan")))
        width, height = int(video["width"]), int(video["height"])
        formats = set(info["format"]["format_name"].split(","))
        if not formats.intersection({"mov", "mp4", "matroska", "webm", "avi"}):
            raise ValueError("Unsupported container")
        if not 0 < duration <= settings.max_seconds or not 0 < width * height <= 3840 * 2160:
            raise ValueError(f"Video must be at most {int(settings.max_seconds)} seconds and 4K")
        rotation = next((int(x["rotation"]) for x in video.get("side_data_list", []) if "rotation" in x), 0)
        if rotation % 180:
            width, height = height, width
        return {"duration_seconds": duration, "width": width, "height": height, "codec": video["codec_name"]}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyError, ValueError, StopIteration) as error:
        raise InvalidVideo("Unreadable, unsupported or out-of-bounds video") from error


def analyze(path, detector, settings, heartbeat=lambda: True, config=None, plate_reader=None, scene=None):
    config=config or AnalysisConfig(sample_fps=settings.sample_fps)
    started = time.monotonic()
    video = probe(path, settings)
    tracker, counts, frames = Tracker(), Counter(), []
    motion = MotionGate()
    signals = SignalMachine()
    through = ThroughLamp()
    # Preserve source pixels for perspective-corrected OCR crops. Downscaling to
    # 720p can remove a character while still producing a high OCR confidence.
    width,height=video['width'],video['height']
    sample_limit=max(1, min(int(video['duration_seconds']*config.sample_fps)+1,
                            int(settings.max_seconds*config.sample_fps)))
    # Keep plate detail in sampled frames; each model applies its own letterbox.
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-protocol_whitelist", "file,pipe",
        "-i", str(path), "-map", "0:v:0", "-vf",
        f"fps={config.sample_fps},scale={width}:{height}",
        "-frames:v", str(sample_limit),
        "-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        timeout = threading.Timer(900, process.kill)
        timeout.start()
        try:
            for index in range(sample_limit):
                raw = process.stdout.read(width * height * 3)
                if not raw:
                    break
                if len(raw) != width * height * 3:
                    raise InvalidVideo("Truncated decoded video frame")
                if not heartbeat():
                    raise RuntimeError("Worker lease lost")
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                detections = detector.detect(image)
                plates=plate_reader.read(image,config.plate_threshold) if plate_reader and config.plate_enabled else []
                for plate in plates:
                    x1,y1,x2,y2=plate['box_normalized']
                    cx,cy=(x1+x2)/2*width,(y1+y2)/2*height
                    matched=[v for v in detections if v['box'][0]<=cx<=v['box'][2] and v['box'][1]<=cy<=v['box'][3]]
                    if matched:
                        vehicle=min(matched,key=lambda v:(v['box'][2]-v['box'][0])*(v['box'][3]-v['box'][1]))
                        if plate['confidence']>=vehicle.get('plate_confidence',-1):
                            vehicle['plate']=plate['text']
                            vehicle['plate_confidence']=plate['confidence']
                for detection in detections:
                    detection['appearance']=appearance_of(image, detection['box'])
                detections = tracker.update(detections, index)
                for detection in detections:
                    detection.pop('appearance', None)
                lights = detect_lights(image, [lamp['box'] for lamp in detector.lamps])
                for detection in detections:
                    counts[detection["label"]] += 1
                    x1, y1, x2, y2 = detection["box"]
                    detection["box_normalized"] = [round(min(1, max(0, x / d)), 5)
                        for x, d in zip((x1, y1, x2, y2),
                            (width, height, width, height))]
                    # Tracker retains its own copy in image coordinates.
                    detection.pop("box")
                    detection.pop("plate_confidence", None)
                motion.update(image,[d['box_normalized'] for d in detections])
                for plate in plates:
                    x1,y1,x2,y2=plate['box_normalized']; cx,cy=(x1+x2)/2,(y1+y2)/2
                    matched=[v for v in detections if v['box_normalized'][0]<=cx<=v['box_normalized'][2]
                             and v['box_normalized'][1]<=cy<=v['box_normalized'][3]]
                    if matched:
                        vehicle=min(matched,key=lambda v:(v['box_normalized'][2]-v['box_normalized'][0])*(v['box_normalized'][3]-v['box_normalized'][1]))
                        plate.update(track_id=vehicle['track_id'],vehicle_type=vehicle['label'])
                time_seconds = round(index / config.sample_fps, 3)
                seen = through.update(lights)
                snap = signals.update(seen, time_seconds)
                frames.append({"time_seconds": time_seconds, "vehicles": detections,'plates':plates,
                               'lights':lights,'signal_observed':seen,'signal_color':snap['color']})
            code = process.wait(timeout=10)
            if code != 0 or not frames:
                raise InvalidVideo("Video decoding failed or produced no frames")
        finally:
            timeout.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    plates=consensus(frames,config.plate_min_hits)
    plate_ok={p['text'] for p in plates if p.get('stable') and p.get('text')}
    violations,assessment,approach,red=[],'PLATE_UNCONFIRMED',None,None
    if not config.plate_enabled or plate_ok:
        violations,assessment=candidates(frames,scene,motion.status(),config.rules_enabled)
        approaches=red_approaches(frames, list(plate_ok))
        approach=approaches[0] if approaches else None
        proceeding=any(item.get('proceeding') or item.get('approaching') for item in approaches)
        signal=({'color':'RED','stable':True} if proceeding else signals.snapshot())
        reds=[]
        if config.rules_enabled:
            for item in approaches:
                hit=red_light_candidate(signal,item,plates,plate_text=item.get('plate'))
                if hit:
                    reds.append(hit)
            if not (scene or {}).get('fixed_camera'):
                laterals=lane_changes(frames)
                violations.extend(laterals)
                violations.extend(restricted_park(frames))
                reds.extend(red_during_laterals(frames, laterals, {item['track_id'] for item in reds}))
            reds=[item for item in reds if not turned_from_green(frames, item['track_id'], item['time_seconds'])]
        violations=[*reds,*violations]
        violations=bind_plates(violations, frames, plates)
        red=next((item for item in violations if item.get('type')=='RED_LIGHT'), None)
        if red:
            assessment='RED_LIGHT_CANDIDATE'
    presence=bool(counts or plates)
    return {
        "schema_version": 2,
        "vehicle_presence": presence,
        "decision": 'CANDIDATE' if violations else ('UNKNOWN' if presence else 'REJECTED'),
        "submission_allowed": False,
        "reason": ('RED_LIGHT_CANDIDATE' if red else 'GEOMETRIC_CANDIDATE') if violations else ('NO_VEHICLES_DETECTED' if not presence else assessment),
        "violation_type": violations[0]['type'] if violations else None,
        "plate": next((v.get('plate') for v in violations if v.get('plate')), None), "plates":plates,
        "signal_state": signals.snapshot(), "signal_model": "YOLOX9_HSV",
        "signal_approach": approach,
        "violations":violations,'rule_assessment':assessment,'camera_motion':motion.status(),
        "evidence_quality": 'PLATE_MULTI_FRAME' if any(p['stable'] for p in plates) else 'PLATE_UNCONFIRMED',
        "model": {"name": detector.spec['name'], "input_size": detector.size, "sha256":detector.spec['sha256']},
        'plate_model': (PLATE_MODELS.get(getattr(config,'plate_model','hyperlpr3'),{}).get('name') or 'HyperLPR3 20230229') if plate_reader and config.plate_enabled else None,
        "video": video, "sample_fps": config.sample_fps, "sampled_frames": len(frames),
        "vehicle_observations": dict(counts), "track_count": tracker.next_id - 1,
        "elapsed_ms": round((time.monotonic() - started) * 1000), "frames": frames,
    }


def bind_plates(violations, frames, plates):
    """Keep a candidate only when a multi-frame plate is on the same track."""
    ok={p['text'] for p in plates if p.get('stable') and p.get('text')}
    canon={p['text'][1:]:p['text'] for p in plates if p.get('text') in ok and len(p['text'])>1}
    for violation in violations:
        evidence=[p for f in frames for p in f.get('plates') or [] if p.get('track_id')==violation.get('track_id')]
        mapped=[canon[p['text'][1:]] for p in evidence if len(p.get('text') or '')>1 and p['text'][1:] in canon]
        if mapped:
            violation['plate']=Counter(mapped).most_common(1)[0][0]
        elif violation.get('plate') not in ok:
            violation.pop('plate', None)
    return [item for item in violations if item.get('plate') in ok]


def _plate_times(frames, item, start, end):
    plate=item.get('plate') or ''
    if not plate:
        return []
    key=plate[1:] if len(plate)>1 else plate
    tid=item.get('track_id')
    times=[]
    for frame in frames:
        t=frame['time_seconds']
        if t<start or t>end:
            continue
        for p in frame.get('plates') or []:
            text=p.get('text') or ''
            if p.get('track_id')!=tid or not text:
                continue
            if text==plate or (len(text)>1 and text[1:]==key):
                times.append(t)
                break
    return times


def _cluster(times, t, gap=1.5):
    times=sorted(times)
    if not times:
        return []
    runs, current = [], []
    for value in times:
        if current and value - current[-1] > gap:
            runs.append(current)
            current = []
        current.append(value)
    if current:
        runs.append(current)
    hit = [run for run in runs if run[0] - gap <= t <= run[-1] + gap]
    return hit[0] if hit else []


def incident_window(item, frames, duration, margin=0.5):
    """Expand from the violation time across this incident's evidence, not a fixed pad."""
    t = float(item.get('time_seconds') or 0)
    tid = item.get('track_id')
    plates = _plate_times(frames, item, 0, duration)
    if not plates:
        return None
    track = [frame['time_seconds'] for frame in frames
             if any(v.get('track_id') == tid for v in frame.get('vehicles') or [])]
    if item.get('type') == 'RED_LIGHT':
        reds = [frame['time_seconds'] for frame in frames if frame.get('signal_observed') == 'RED']
        core = _cluster(reds, t, 1.5)
    else:
        core = _cluster([value for value in track if abs(value - t) <= 6], t, 1.0)
    if not core:
        core = [t]
    start, end = core[0], max(core[-1], t)
    nearby = [p for p in plates if start - 1 <= p <= end + 2]
    if not nearby:
        return None
    start = min(start, min(nearby))
    end = max(end, max(nearby), t)
    return max(0.0, start - margin), min(float(duration), end + margin)


def clip_windows(violations, duration, frames=None, merge=0.5):
    """One clip per plated incident span. No frames or no plate in view = not a clip."""
    if not frames:
        return []
    windows=[]
    for item in sorted(violations, key=lambda row: float(row.get('time_seconds') or 0)):
        if not item.get('plate') or not item.get('type'):
            continue
        span = incident_window(item, frames, duration)
        if not span:
            continue
        start, end = span
        if windows and start <= windows[-1]['end'] + merge:
            windows[-1]['end'] = max(windows[-1]['end'], end)
            windows[-1]['items'].append(item)
        else:
            windows.append({'start': start, 'end': end, 'items': [item]})
    return windows


def extract_clips(source, violations, dest, duration, frames=None):
    dest=Path(dest)
    if dest.is_dir():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    clips=[]
    limit=50*1024*1024
    for window in clip_windows(violations, duration, frames):
        index=len(clips)
        out=dest/f'{index}.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',f"{window['start']:.3f}",
            '-i',str(source),'-t',f"{window['end']-window['start']:.3f}",'-map','0:v:0',
            '-c:v','libx264','-preset','veryfast','-crf','28','-an','-sn','-dn',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(out)],
            capture_output=True,timeout=90,check=True)
        if not out.is_file() or not 0<out.stat().st_size<=limit:
            out.unlink(missing_ok=True)
            continue
        clip={'index':index,'start_seconds':round(window['start'],3),'end_seconds':round(window['end'],3),
              'plate':window['items'][0].get('plate'),
              'types':sorted({item['type'] for item in window['items']})}
        clips.append(clip)
        for item in window['items']:
            item['clip_index']=index
            item['clip_start']=clip['start_seconds']
            item['clip_end']=clip['end_seconds']
    return clips
