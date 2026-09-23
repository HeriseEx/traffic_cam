import json
import secrets
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app
from config import Settings
from hello import code as hello_code
from settings_security import SettingsPassword, PasswordError, password_matches, MAX_ATTEMPTS, WINDOW_SECONDS
from store import Store, Conflict
import settings_password


PASSWORD = 'settings-test-only-长口令-2026'


class SettingsSecurityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = Settings(data=Path(self.directory.name),
            model=Path(__file__).resolve().parent/'models/yolox_s.onnx',
            token='settings-test-api-token-long-enough', reserve_bytes=0)
        self.app = create_app(self.settings)
        self.store = self.app.state.store
        self.password = SettingsPassword(self.store)
        self.auth = {'Authorization': 'Bearer '+self.settings.token}
        self.client = self.enterContext(TestClient(self.app, base_url='https://testserver'))

    def body(self, password=PASSWORD):
        current = self.store.configuration()
        return {'expected_revision': current['revision'],
                'config': {**current['config'], 'vehicle_threshold': .4, 'plate_enabled': False},
                'password': password}

    def save(self, body=None, headers=None):
        return self.client.put('/v1/settings', json=self.body() if body is None else body,
                               headers=self.auth if headers is None else headers)

    def test_unconfigured_is_read_only_including_admin_token(self):
        response = self.client.get('/v1/settings', headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['settings_security'], {'configured': False, 'retry_after': 0})
        self.assertEqual(self.save(headers={}).status_code, 401)
        self.assertEqual(self.save().status_code, 503)
        self.assertEqual(self.store.configuration()['revision'], 1)

    def test_password_required_for_each_save_and_never_returned(self):
        self.password.set_password(PASSWORD)
        without = self.body(); without.pop('password')
        self.assertEqual(self.save(without).status_code, 403)
        self.assertEqual(self.save(self.body('wrong password')).status_code, 403)
        self.assertEqual(self.store.configuration()['revision'], 1)
        saved = self.save()
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()['config']['vehicle_threshold'], .4)
        self.assertEqual(self.save(without).status_code, 403)
        self.assertEqual(self.save().status_code, 200)
        raw = json.dumps(self.client.get('/v1/settings', headers=self.auth).json(), ensure_ascii=False)
        self.assertNotIn(PASSWORD, raw)
        self.assertNotIn('scrypt-v1', raw)
        with self.store.connection() as db:
            security = dict(db.execute('SELECT * FROM settings_security').fetchone())
            self.assertEqual(security['attempts'], 0)
            self.assertNotIn(PASSWORD, json.dumps(security, ensure_ascii=False))
            self.assertNotIn(PASSWORD, db.execute('SELECT config FROM runtime_settings').fetchone()[0])

    def test_lockout_survives_new_device_spoofed_ip_and_restart(self):
        self.password.set_password(PASSWORD)
        for attempt in range(MAX_ATTEMPTS):
            device = f'attacking-browser-{attempt}'
            stamp, nonce = int(time.time()), secrets.token_hex(16)
            hello = self.client.post('/v1/hello', json={'device_id': device, 'platform': 'android',
                'model': 'test', 'ts': stamp, 'nonce': nonce, 'code': hello_code(device, 'android', stamp, nonce)})
            auth = {'Authorization': 'Bearer '+hello.json()['session'], 'X-Forwarded-For': f'203.0.113.{attempt}'}
            response = self.save(self.body('incorrect'), headers=auth)
            self.assertEqual(response.status_code, 429 if attempt == MAX_ATTEMPTS-1 else 403)
        self.assertEqual(int(response.headers['Retry-After']), WINDOW_SECONDS)
        self.assertEqual(self.save().status_code, 429)  # No password comparison while locked, even if correct.
        restarted = create_app(self.settings)
        with TestClient(restarted) as client:
            response = client.put('/v1/settings', headers=self.auth, json=self.body())
            self.assertEqual(response.status_code, 429)
            self.assertGreater(response.json()['retry_after'], 0)
        self.assertEqual(self.store.configuration()['revision'], 1)
        future = time.time()+WINDOW_SECONDS+1
        with patch('settings_security.time.time', return_value=future):
            self.assertEqual(self.save().status_code, 200)
        self.assertEqual(self.password.status()['retry_after'], 0)

    def test_concurrent_workers_cannot_race_past_guess_budget(self):
        self.password.set_password(PASSWORD)
        gates = [SettingsPassword(Store(self.settings)) for _ in range(3)]
        def attempt(index):
            try:
                gates[index % len(gates)].verify('incorrect')
            except PasswordError as error:
                return error.status
            self.fail('Wrong password accepted')
        with patch('settings_security.password_matches', wraps=password_matches) as compare:
            with ThreadPoolExecutor(max_workers=12) as pool:
                statuses = list(pool.map(attempt, range(12)))
            self.assertEqual(compare.call_count, MAX_ATTEMPTS)
        self.assertEqual(statuses.count(403), MAX_ATTEMPTS-1)
        self.assertEqual(statuses.count(429), 12-MAX_ATTEMPTS+1)
        self.assertGreater(self.password.status()['retry_after'], 0)

    def test_reset_revokes_old_password_and_inflight_verification(self):
        self.password.set_password(PASSWORD)
        verified_revision = self.password.verify(PASSWORD)
        current = self.store.configuration()
        self.password.set_password('replacement-password-for-test')
        with self.assertRaises(Conflict):
            self.store.configure(current['revision'], current['config'], password_revision=verified_revision)
        self.assertEqual(self.save().status_code, 403)
        self.assertEqual(self.save(self.body('replacement-password-for-test')).status_code, 200)

    def test_cli_initialization_reset_and_unlock(self):
        args = ['settings_password.py', 'set', '--data-dir', self.directory.name]
        with patch('sys.argv', args), patch('getpass.getpass', side_effect=[PASSWORD, PASSWORD]), patch('builtins.print'):
            settings_password.main()
        self.assertTrue(self.password.status()['configured'])
        self.assertEqual(self.save().status_code, 200)
        with self.store.connection() as db:
            first_hash = db.execute('SELECT password_hash FROM settings_security').fetchone()[0]
        with patch('sys.argv', args), patch('getpass.getpass', side_effect=[PASSWORD, PASSWORD]), patch('builtins.print'):
            settings_password.main()
        with self.store.connection() as db:
            self.assertNotEqual(first_hash, db.execute('SELECT password_hash FROM settings_security').fetchone()[0])
            db.execute('UPDATE settings_security SET attempts=?,window_started=?', (MAX_ATTEMPTS, time.time()))
        with patch('sys.argv', ['settings_password.py', 'unlock', '--data-dir', self.directory.name]), patch('builtins.print'):
            settings_password.main()
        self.assertEqual(self.password.status()['retry_after'], 0)
        self.assertEqual(self.save().status_code, 200)
        with self.assertRaises(ValueError):
            self.password.set_password('123456')
        self.assertEqual(self.save().status_code, 200)  # A rejected reset leaves the existing password intact.

    def test_cookie_csrf_and_invalid_payload_cannot_echo_password(self):
        self.password.set_password(PASSWORD)
        self.client.post('/v1/session', headers=self.auth)
        self.assertEqual(self.save(headers={}).status_code, 403)
        self.assertEqual(self.save(headers={'X-Requested-With': 'traffic-console'}).status_code, 200)
        for body in ({**self.body(), 'extra': PASSWORD}, {**self.body(), 'password': PASSWORD*10},
                     {**self.body(), 'config': {'vehicle_model': PASSWORD}}):
            response = self.save(body)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(PASSWORD, response.text)
        stale = self.body(); stale['expected_revision'] = 1
        self.assertEqual(self.save(stale).status_code, 409)


if __name__ == '__main__':
    unittest.main()
