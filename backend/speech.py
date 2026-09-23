"""Transcribe a manual recording and match it to the offence catalogue. Never invents a transcript."""
import json
import subprocess
import wave
from pathlib import Path

PHRASES = (
    ('变道不打灯', 'NO_SIGNAL'),
    ('不打转向灯', 'NO_SIGNAL'),
    ('不打灯', 'NO_SIGNAL'),
    ('应急车道', 'EMERGENCY_LANE'),
    ('越线超车', 'OVERTAKE'),
    ('压实线', 'SOLID_LINE'),
    ('闯红灯', 'RED_LIGHT'),
    ('危险变道', 'DANGEROUS_CHANGE'),
    ('乱停乱放', 'ILLEGAL_PARKING'),
    ('乱停', 'ILLEGAL_PARKING'),
    ('逆行', 'WRONG_WAY'),
    ('加塞', 'CUT_IN'),
    ('实线', 'SOLID_LINE'),
    ('红灯', 'RED_LIGHT'),
)


class SpeechUnavailable(Exception):
    pass


def compact(text):
    return ''.join(ch for ch in (text or '') if not ch.isspace() and ch not in '，。！？、,.')


def match_offence(text):
    folded = compact(text)
    if not folded:
        return None
    for phrase, kind in PHRASES:
        if phrase in folded:
            return kind
    return None


def model_dir():
    dest = Path(__file__).resolve().parent / 'models' / 'vosk-model-small-cn-0.22'
    if (dest / 'conf' / 'model.conf').is_file():
        return dest
    archive = Path(__file__).resolve().parent / 'validation' / 'vosk-model-small-cn-0.22.zip'
    if not archive.is_file():
        raise SpeechUnavailable('NO_MODEL')
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            parts = Path(name).parts[1:]
            if not parts or name.endswith('/'):
                continue
            target = dest.joinpath(*parts).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise SpeechUnavailable('NO_MODEL')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))
    if not (dest / 'conf' / 'model.conf').is_file():
        raise SpeechUnavailable('NO_MODEL')
    return dest


_model = None


def _recognizer():
    global _model
    if _model is None:
        try:
            from vosk import Model, SetLogLevel
        except ImportError as error:
            raise SpeechUnavailable('NO_ENGINE') from error
        SetLogLevel(-1)
        _model = Model(str(model_dir()))
    from vosk import KaldiRecognizer
    return KaldiRecognizer(_model, 16000)


def transcribe(video, wav_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-nostdin', '-v', 'error', '-i', str(video),
             '-vn', '-ac', '1', '-ar', '16000', '-f', 'wav', str(wav_path)],
            check=True, timeout=90, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise SpeechUnavailable('NO_AUDIO') from error
    if not wav_path.is_file() or wav_path.stat().st_size <= 44:
        raise SpeechUnavailable('NO_AUDIO')
    try:
        rec = _recognizer()
    except SpeechUnavailable:
        raise
    except Exception as error:
        raise SpeechUnavailable('NO_ENGINE') from error
    parts = []
    with wave.open(str(wav_path)) as stream:
        if stream.getnchannels() != 1 or stream.getframerate() != 16000:
            raise SpeechUnavailable('NO_AUDIO')
        while True:
            data = stream.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                parts.append(json.loads(rec.Result()).get('text') or '')
        parts.append(json.loads(rec.FinalResult()).get('text') or '')
    return ' '.join(part for part in parts if part).strip()


def attach_speech(store, task, result):
    trigger = (task.get('metadata') or {}).get('trigger')
    if trigger not in ('manual', 'voice'):
        return result
    try:
        text = transcribe(store.video(task['task_id']), store.audio_file(task['task_id']))
    except SpeechUnavailable as error:
        result['transcript'] = ''
        result['transcript_error'] = str(error)
        return result
    return apply_transcript(result, text, task['task_id'])


def apply_transcript(result, text, task_id):
    kind = match_offence(text)
    result['transcript'] = text
    result['spoken_type'] = kind
    result['audio'] = f"{task_id}.wav"
    if kind and not result.get('violation_type'):
        result['violation_type'] = kind
        result['reason'] = 'SPOKEN_MATCH'
        if result.get('decision') in (None, 'UNKNOWN', 'REJECTED'):
            result['decision'] = 'CANDIDATE'
    elif not kind and not result.get('violations') and not result.get('violation_type'):
        result['reason'] = 'SPOKEN_UNMATCHED'
        if result.get('decision') in (None, 'REJECTED'):
            result['decision'] = 'UNKNOWN'
    return result
