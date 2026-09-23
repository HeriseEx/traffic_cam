"""Separate settings-write password; shared, persistent online guessing budget."""
import hashlib
import math
import secrets
import time


MAX_ATTEMPTS = 5
WINDOW_SECONDS = 15 * 60


class PasswordError(Exception):
    def __init__(self, detail, status=403, retry_after=0):
        super().__init__(detail)
        self.status = status
        self.retry_after = retry_after


def hash_password(password, min_len=15):
    if not min_len <= len(password) <= 128:
        if min_len == 15:
            raise ValueError('管理密码必须为 15–128 个字符，可使用长口令。')
        raise ValueError(f'密码必须为 {min_len}–128 个字符。')
    salt = secrets.token_bytes(16)
    # OWASP scrypt option: 32 MiB, r=8, p=3; no extra runtime dependency.
    key = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=32768, r=8, p=3, dklen=32, maxmem=64*1024*1024)
    return f'scrypt-v1${salt.hex()}${key.hex()}'


def password_matches(password, encoded):
    try:
        version, salt, expected = encoded.split('$')
        if version != 'scrypt-v1' or len(salt) != 32 or len(expected) != 64:
            return False
        actual = hashlib.scrypt(password.encode('utf-8'), salt=bytes.fromhex(salt), n=32768,
                                r=8, p=3, dklen=32, maxmem=64*1024*1024)
        return secrets.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, UnicodeError):
        return False


class SettingsPassword:
    def __init__(self, store):
        self.store = store

    def status(self):
        with self.store.connection() as db:
            row = db.execute('SELECT * FROM settings_security WHERE id=1').fetchone()
        wait = max(0, math.ceil(row['window_started'] + WINDOW_SECONDS - time.time())) if row['attempts'] >= MAX_ATTEMPTS else 0
        return {'configured': bool(row['password_hash']), 'retry_after': wait}

    def set_password(self, password):
        encoded = hash_password(password)
        with self.store.connection() as db:
            db.execute('''UPDATE settings_security SET password_hash=?,revision=revision+1,
                          attempts=0,window_started=0 WHERE id=1''', (encoded,))

    def unlock(self):
        with self.store.connection() as db:
            db.execute('UPDATE settings_security SET attempts=0,window_started=0 WHERE id=1')

    def verify(self, password):
        now = time.time()
        # Reserve before hashing. Parallel requests/processes cannot exceed the shared budget.
        # It is intentionally independent of spoofable IPs, device IDs and browser storage.
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM settings_security WHERE id=1').fetchone()
            if not row['password_hash']:
                raise PasswordError('尚未配置管理密码，请先在服务器初始化。', 503)
            started = row['window_started']
            attempts = row['attempts']
            if now >= started + WINDOW_SECONDS:
                attempts, started = 0, now
            wait = max(1, math.ceil(started + WINDOW_SECONDS - now))
            if attempts >= MAX_ATTEMPTS:
                raise PasswordError('密码尝试过于频繁，请稍后重试。', 429, wait)
            if not password or len(password) > 128:
                raise PasswordError('请输入有效的管理密码。')
            attempts += 1
            if attempts == MAX_ATTEMPTS:
                started, wait = now, WINDOW_SECONDS
            db.execute('UPDATE settings_security SET attempts=?,window_started=? WHERE id=1', (attempts, started))
        if not password_matches(password, row['password_hash']):
            if attempts >= MAX_ATTEMPTS:
                raise PasswordError('密码尝试过于频繁，请稍后重试。', 429, wait)
            raise PasswordError('管理密码不正确。')
        # The writer rechecks this version atomically: password reset revokes in-flight verification.
        return row['revision']
