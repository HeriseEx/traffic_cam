"""Device hello: SHA-256 code, then a session. Same formula for Android / iOS / web."""
import hashlib
import hmac
import time

SALT = "traffic-hello-v1"


def code(device_id, platform, ts, nonce):
    raw = f"{device_id}\n{platform}\n{int(ts)}\n{nonce}\n{SALT}"
    return hashlib.sha256(raw.encode()).hexdigest()


def ok(device_id, platform, ts, nonce, offered, now=None):
    now = time.time() if now is None else now
    if abs(now - int(ts)) > 300:
        return False
    if not offered or len(offered) != 64:
        return False
    return hmac.compare_digest(code(device_id, platform, ts, nonce), offered.strip().lower())
