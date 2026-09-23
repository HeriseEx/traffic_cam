import json
import hashlib
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager

from config import Settings
from schemas import AnalysisConfig


class Conflict(Exception):
    pass


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.data / "tasks.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
                    metadata TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
                    status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until REAL, owner TEXT, result TEXT, error TEXT
                );
                CREATE INDEX IF NOT EXISTS queue_index ON tasks(status, created_at);
                CREATE TABLE IF NOT EXISTS runtime_settings (id INTEGER PRIMARY KEY CHECK(id=1),
                    revision INTEGER NOT NULL, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS changes (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS worker_state (id INTEGER PRIMARY KEY CHECK(id=1),
                    heartbeat REAL NOT NULL, message TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS device_pairs (
                    code TEXT PRIMARY KEY, expires_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY, device_id TEXT UNIQUE NOT NULL,
                    platform TEXT NOT NULL, model TEXT NOT NULL,
                    app_version TEXT NOT NULL DEFAULT '', ip TEXT NOT NULL DEFAULT '',
                    session_hash TEXT NOT NULL, created_at REAL NOT NULL, last_seen REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS client_session ON clients(session_hash);
                CREATE TABLE IF NOT EXISTS client_sessions (
                    session_hash TEXT PRIMARY KEY, client_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_client ON client_sessions(client_id);
                CREATE TABLE IF NOT EXISTS settings_security (
                    id INTEGER PRIMARY KEY CHECK(id=1), password_hash TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                    window_started REAL NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO settings_security(id) VALUES(1);
            """)
            # Migrate existing credentials once; browser tabs may each hold a valid session for one device.
            db.execute('INSERT OR IGNORE INTO client_sessions SELECT session_hash,client_id,? FROM clients WHERE session_hash<>?',
                       (time.time()+30*86400, ''))
            db.execute("UPDATE clients SET session_hash='' WHERE session_hash<>''")
            columns = {row['name'] for row in db.execute('PRAGMA table_info(tasks)')}
            for name, declaration in {'revision': 'INTEGER NOT NULL DEFAULT 1',
                    'analysis_config': 'TEXT', 'scene': 'TEXT', 'review': 'TEXT',
                    'submission_status': "TEXT NOT NULL DEFAULT 'NOT_SUBMITTED'", 'receipt': 'TEXT'}.items():
                if name not in columns:
                    db.execute(f'ALTER TABLE tasks ADD COLUMN {name} {declaration}')
            db.execute('INSERT OR IGNORE INTO runtime_settings VALUES(1,1,?)',
                       (AnalysisConfig().model_dump_json(),))
            db.execute("""INSERT INTO changes(task_id,created_at) SELECT task_id,updated_at FROM tasks
                WHERE task_id NOT IN (SELECT task_id FROM changes)""")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        for key in ('analysis_config', 'scene', 'review'):
            result[key] = json.loads(result[key]) if result.get(key) else None
        ai, review = result['result'] or {}, result['review']
        effective = {key: ai.get(key) for key in ('decision', 'plate', 'violation_type', 'reason')}
        effective.update(source='AI', review_status='AUTOMATIC', submission_allowed=False)
        if review:
            effective.update(source='HUMAN', review_status=review['decision'])
            if review['decision'] == 'INVALID':
                effective.update(decision='REJECTED', reason='HUMAN_INVALIDATED')
            elif review['decision'] == 'UNCERTAIN':
                effective.update(decision='UNKNOWN', reason='HUMAN_UNCERTAIN')
            else:
                effective.update(decision='CONFIRMED', reason='HUMAN_CONFIRMED')
            for key in ('plate', 'violation_type'):
                if review.get(key) is not None:
                    effective[key] = review[key]
        result['effective_result'] = effective
        result.pop("owner", None)
        result.pop("lease_until", None)
        return result

    def get(self, task_id):
        with self.connection() as db:
            return self.decode(db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())

    def list(self, limit=30, offset=0, status='', review='', query=''):
        with self.connection() as db:
            filters, args = ['1=1'], []
            if status:
                filters.append('status=?'); args.append(status)
            if review == 'UNSUBMITTED':
                filters.append("submission_status='NOT_SUBMITTED'")
            elif review == 'INTERVENED':
                filters.append('review IS NOT NULL')
            elif review == 'SUBMITTED':
                filters.append("submission_status='SUBMITTED'")
            if query:
                filters.append('(event_id LIKE ? OR result LIKE ? OR review LIKE ?)')
                args.extend(['%'+query+'%']*3)
            return [self.decode(row) for row in db.execute(
                'SELECT * FROM tasks WHERE '+ ' AND '.join(filters) + ' ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (*args, limit, offset))]

    @staticmethod
    def changed(db, task_id):
        db.execute('UPDATE tasks SET revision=revision+1,updated_at=? WHERE task_id=?', (time.time(), task_id))
        db.execute('INSERT INTO changes(task_id,created_at) VALUES(?,?)', (task_id,time.time()))

    def changes(self, after, limit=50):
        with self.connection() as db:
            rows = db.execute('SELECT id,task_id FROM changes WHERE id>? ORDER BY id LIMIT ?', (after,limit)).fetchall()
            tasks = [self.decode(db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone())
                     for task_id in dict.fromkeys(row['task_id'] for row in rows)]
            # Mobile catch-up needs judgments, not every detection box from every video.
            # Full evidence remains available through GET /v1/tasks/{task_id}.
            for task in tasks:
                if task.get('result'):
                    task['result'].pop('frames', None)
            return {'cursor': rows[-1]['id'] if rows else after, 'tasks': tasks, 'has_more': len(rows)==limit}

    def configuration(self):
        with self.connection() as db:
            row = db.execute('SELECT * FROM runtime_settings WHERE id=1').fetchone()
            return {'revision': row['revision'], 'config': json.loads(row['config'])}

    def configure(self, expected, config, password_revision=None):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if password_revision is not None:
                security = db.execute('SELECT revision FROM settings_security WHERE id=1').fetchone()
                if security['revision'] != password_revision:
                    raise Conflict('管理密码已更新，请输入新密码后重试。')
            if not db.execute('UPDATE runtime_settings SET config=?,revision=revision+1 WHERE id=1 AND revision=?',
                              (json.dumps(config), expected)).rowcount:
                raise Conflict('配置已更新，请刷新后重试')
            if password_revision is not None:
                db.execute('UPDATE settings_security SET attempts=0,window_started=0 WHERE id=1')
        return self.configuration()

    def audit(self, task_id):
        with self.connection() as db:
            return [{**dict(row), 'payload': json.loads(row['payload'])} for row in db.execute(
                'SELECT * FROM audit WHERE task_id=? ORDER BY id DESC LIMIT 100', (task_id,))]

    def intervene(self, task_id, review):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.editable(db, task_id, review['expected_revision'])
            if row['status'] not in ('ANALYZED','REJECTED','ERROR'):
                raise Conflict('请等待分析完成后再干预')
            review = {**review, 'at': time.time()}
            payload = json.dumps(review,ensure_ascii=False)
            db.execute('UPDATE tasks SET review=? WHERE task_id=?',
                       (None if review['decision']=='RESET' else payload, task_id))
            db.execute('INSERT INTO audit(task_id,kind,payload,created_at) VALUES(?,?,?,?)',
                       (task_id,'REVIEW',payload,time.time()))
            self.changed(db,task_id)
        return self.get(task_id)

    @staticmethod
    def editable(db, task_id, expected):
        row = db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
        if not row:
            raise Conflict('任务不存在')
        if row['submission_status'] != 'NOT_SUBMITTED':
            raise Conflict('已提交的任务不能修改判定')
        if row['revision'] != expected:
            raise Conflict('任务已更新，请刷新后重试')
        return row

    def reanalyze(self, task_id, expected, config, scene):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.editable(db,task_id,expected)
            if row['status'] in ('QUEUED','PROCESSING','EXPIRED') or row['expires_at'] <= time.time():
                raise Conflict('任务正在处理或视频已过期')
            if not self.video(task_id).exists():
                raise Conflict('原始视频不存在')
            db.execute('INSERT INTO audit(task_id,kind,payload,created_at) VALUES(?,?,?,?)',
                       (task_id,'REANALYZE',json.dumps(self.decode(row),ensure_ascii=False),time.time()))
            db.execute("""UPDATE tasks SET status='QUEUED',result=NULL,review=NULL,error=NULL,attempts=0,
                analysis_config=?,scene=? WHERE task_id=?""", (json.dumps(config),json.dumps(scene),task_id))
            self.changed(db,task_id)
        return self.get(task_id)

    def submitted(self, task_id, expected, receipt):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=self.editable(db,task_id,expected)
            if row['status'] not in ('ANALYZED','REJECTED'):
                raise Conflict('分析尚未完成')
            db.execute("UPDATE tasks SET submission_status='SUBMITTED',receipt=? WHERE task_id=?", (receipt,task_id))
            db.execute('INSERT INTO audit(task_id,kind,payload,created_at) VALUES(?,?,?,?)',
                       (task_id,'SUBMITTED',json.dumps({'receipt':receipt}),time.time()))
            self.changed(db,task_id)
        return self.get(task_id)

    def worker_heartbeat(self, message):
        with self.connection() as db:
            db.execute('INSERT OR REPLACE INTO worker_state VALUES(1,?,?)',(time.time(),message))

    def overview(self):
        with self.connection() as db:
            counts = dict(db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status').fetchall())
            row = db.execute('SELECT * FROM worker_state WHERE id=1').fetchone()
            return {'counts':counts,'total':sum(counts.values()),
                    'intervened':db.execute('SELECT COUNT(*) FROM tasks WHERE review IS NOT NULL').fetchone()[0],
                    'worker':dict(row) if row else None,
                    'client_count':db.execute('SELECT COUNT(*) FROM clients').fetchone()[0],
                    'clients':[dict(item) for item in db.execute(
                        'SELECT client_id,platform,model,app_version,ip,created_at,last_seen FROM clients ORDER BY last_seen DESC LIMIT 50')]}

    def video(self, task_id):
        # Only validated UUIDs from our database are allowed in paths.
        return self.settings.data / "videos" / f"{uuid.UUID(task_id)}.mp4"

    def clips_dir(self, task_id):
        return self.settings.data / "clips" / str(uuid.UUID(task_id))

    def archive_all(self):
        """Hard-clear records and files so the console is empty. Processing tasks block."""
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            busy = db.execute("SELECT COUNT(*) FROM tasks WHERE status='PROCESSING' AND lease_until>?",
                              (now,)).fetchone()[0]
            if busy:
                raise Conflict('有任务正在分析，请稍后再归档')
            ids = [row[0] for row in db.execute('SELECT task_id FROM tasks')]
            db.execute('DELETE FROM audit')
            db.execute('DELETE FROM changes')
            db.execute('DELETE FROM tasks')
        for task_id in ids:
            self.video(task_id).unlink(missing_ok=True)
            (self.settings.data / 'previews' / f'{task_id}.mp4').unlink(missing_ok=True)
            folder = self.settings.data / 'clips' / task_id
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
        return len(ids)

    def issue_pair(self, ttl=1800):
        code = f'{secrets.randbelow(100_000_000):08d}'
        now = time.time()
        with self.connection() as db:
            db.execute('DELETE FROM device_pairs WHERE expires_at<?', (now,))
            db.execute('INSERT INTO device_pairs(code,expires_at) VALUES(?,?)', (code, now+ttl))
        return code

    def pair_ok(self, token):
        if not token or not token.isdigit() or len(token)!=8:
            return False
        with self.connection() as db:
            row = db.execute('SELECT expires_at FROM device_pairs WHERE code=?', (token,)).fetchone()
        return bool(row and row['expires_at'] > time.time())

    def hello(self, device_id, platform, model, app_version, ip, previous_session=''):
        now = time.time()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM client_sessions WHERE expires_at<=?', (now,))
            row = db.execute('SELECT client_id FROM clients WHERE device_id=?', (device_id,)).fetchone()
            if row:
                db.execute("""UPDATE clients SET platform=?,model=?,app_version=?,ip=?,last_seen=?
                    WHERE device_id=?""", (platform, model, app_version, ip, now, device_id))
                client_id = row['client_id']
            else:
                client_id = str(uuid.uuid4())
                db.execute("""INSERT INTO clients(client_id,device_id,platform,model,app_version,ip,session_hash,created_at,last_seen)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (client_id, device_id, platform, model, app_version, ip, '', now, now))
            prior_hash = hashlib.sha256(previous_session.encode()).hexdigest()
            reuse = previous_session and db.execute('SELECT 1 FROM client_sessions WHERE session_hash=? AND client_id=?',
                                                    (prior_hash, client_id)).fetchone()
            session = previous_session if reuse else secrets.token_urlsafe(32)
            digest = hashlib.sha256(session.encode()).hexdigest()
            db.execute('INSERT OR REPLACE INTO client_sessions VALUES(?,?,?)', (digest, client_id, now+30*86400))
        return {'client_id': client_id, 'device_id': device_id, 'session': session, 'platform': platform, 'model': model, 'ip': ip}

    def session_device(self, session):
        if not session or len(session)>80:
            return None
        digest = hashlib.sha256(session.encode()).hexdigest()
        with self.connection() as db:
            row = db.execute('''SELECT clients.device_id FROM client_sessions JOIN clients USING(client_id)
                WHERE client_sessions.session_hash=? AND expires_at>?''', (digest, time.time())).fetchone()
            return row['device_id'] if row else None

    def revoke_session(self, session):
        with self.connection() as db:
            db.execute('DELETE FROM client_sessions WHERE session_hash=?', (hashlib.sha256(session.encode()).hexdigest(),))

    def client_ok(self, session, ip=''):
        if not session or len(session) > 80:
            return False
        digest = hashlib.sha256(session.encode()).hexdigest()
        with self.connection() as db:
            row = db.execute('SELECT client_id FROM client_sessions WHERE session_hash=? AND expires_at>?', (digest, time.time())).fetchone()
            if not row:
                return False
            if ip:
                db.execute('UPDATE clients SET last_seen=?,ip=? WHERE client_id=?',
                           (time.time(), ip, row['client_id']))
            else:
                db.execute('UPDATE clients SET last_seen=? WHERE client_id=?',
                           (time.time(), row['client_id']))
            return True

    def add(self, metadata, sha256, size, temporary):
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        task_id = str(uuid.uuid4())
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM tasks WHERE event_id=?", (metadata["event_id"],)).fetchone()
            if existing:
                previous = {'trigger':'import','trigger_text':'','scene':None,'capture':None,**json.loads(existing['metadata'])}
                if existing["sha256"] != sha256 or previous != metadata:
                    raise Conflict("event_id already exists with different content or metadata")
                return self.decode(existing), False
            destination = self.video(task_id)
            temporary.replace(destination)
            try:
                db.execute("""INSERT INTO tasks
                    (task_id,event_id,metadata,sha256,bytes,status,created_at,updated_at,expires_at)
                    VALUES (?,?,?,?,?,'QUEUED',?,?,?)""",
                    (task_id, metadata["event_id"], encoded, sha256, size, now, now,
                     now + self.settings.retention_hours * 3600))
                config = db.execute('SELECT config FROM runtime_settings WHERE id=1').fetchone()[0]
                db.execute('UPDATE tasks SET analysis_config=?,scene=? WHERE task_id=?',
                    (config,json.dumps(metadata.get('scene')),task_id))
                db.execute('INSERT INTO changes(task_id,created_at) VALUES(?,?)',(task_id,now))
                db.commit()
            except BaseException:
                db.rollback()
                destination.unlink(missing_ok=True)
                raise
        return self.get(task_id), True

    def claim(self, owner):
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            expired = [r[0] for r in db.execute("SELECT task_id FROM tasks WHERE status='PROCESSING' AND lease_until<?",(now,))]
            db.execute("""UPDATE tasks SET status=CASE WHEN attempts>=? THEN 'ERROR' ELSE 'QUEUED' END,
                owner=NULL, lease_until=NULL, updated_at=?, error='worker lease expired'
                WHERE status='PROCESSING' AND lease_until<?""", (self.settings.max_attempts, now, now))
            for task_id in expired:
                self.changed(db,task_id)
            row = db.execute("""SELECT * FROM tasks WHERE status='QUEUED' AND expires_at>?
                ORDER BY created_at LIMIT 1""", (now,)).fetchone()
            if not row:
                return None
            db.execute("""UPDATE tasks SET status='PROCESSING', owner=?, lease_until=?,
                attempts=attempts+1, updated_at=?, error=NULL WHERE task_id=?""",
                (owner, now + self.settings.lease_seconds, now, row["task_id"]))
            self.changed(db,row['task_id'])
        return self.get(row["task_id"])

    def heartbeat(self, task_id, owner):
        with self.connection() as db:
            return db.execute("""UPDATE tasks SET lease_until=?
                WHERE task_id=? AND owner=? AND status='PROCESSING'""",
                (time.time() + self.settings.lease_seconds, task_id, owner)).rowcount == 1

    def finish(self, task_id, owner, status, result=None, error=None):
        with self.connection() as db:
            changed = db.execute("""UPDATE tasks SET status=?,result=?,error=?,updated_at=?,owner=NULL,
                lease_until=NULL WHERE task_id=? AND owner=? AND status='PROCESSING'""",
                (status, json.dumps(result, ensure_ascii=False) if result else None, error,
                 time.time(), task_id, owner)).rowcount == 1
            if changed:
                self.changed(db,task_id)
            return changed

    def retry(self, task_id):
        with self.connection() as db:
            changed = db.execute("""UPDATE tasks SET status='QUEUED',error=NULL,updated_at=?
                WHERE task_id=? AND status='ERROR' AND expires_at>? AND attempts<? AND submission_status='NOT_SUBMITTED'""",
                (time.time(), task_id, time.time(), self.settings.max_attempts)).rowcount == 1
            if changed:
                self.changed(db,task_id)
            return changed

    def expire(self, task_id):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if not row:
                return False
            if row["status"] == "PROCESSING":
                raise Conflict("Task is processing; retry deletion when it finishes")
            db.execute("UPDATE tasks SET status='EXPIRED',updated_at=? WHERE task_id=?", (time.time(), task_id))
            self.changed(db,task_id)
        self.video(task_id).unlink(missing_ok=True)
        folder = self.settings.data / 'clips' / task_id
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        return True

    def cleanup(self):
        now = time.time()
        with self.connection() as db:
            expiring = [r[0] for r in db.execute("""SELECT task_id FROM tasks WHERE expires_at<?
                AND status!='EXPIRED' AND (status!='PROCESSING' OR lease_until<?)""",(now,now))]
            db.execute("""UPDATE tasks SET status='EXPIRED',owner=NULL,lease_until=NULL,updated_at=?
                WHERE expires_at<? AND status!='EXPIRED' AND (status!='PROCESSING' OR lease_until<?)""", (now, now, now))
            for task_id in expiring:
                self.changed(db,task_id)
            expired = [row[0] for row in db.execute("SELECT task_id FROM tasks WHERE status='EXPIRED'")]
            referenced = {row[0] for row in db.execute("SELECT task_id FROM tasks")}
        for task_id in expired:
            self.video(task_id).unlink(missing_ok=True)
            (self.settings.data/'previews'/f'{task_id}.mp4').unlink(missing_ok=True)
            folder = self.settings.data/'clips'/task_id
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
        # Crash before the DB commit can leave a file; don't touch recent/in-flight uploads.
        for path in (self.settings.data / "videos").iterdir():
            if path.is_file() and path.stem not in referenced and path.stat().st_mtime < now - 3600:
                path.unlink(missing_ok=True)
