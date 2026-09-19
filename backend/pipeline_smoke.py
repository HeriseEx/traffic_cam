"""Isolated Android end-to-end regression. Does not touch the configured production server."""
import json
from pathlib import Path
import secrets
import subprocess
import tempfile
import threading
import time

import uvicorn

from app import create_app
from config import Settings
from inference import Detector
from worker import process_one

ROOT = Path(__file__).resolve().parent
PACKAGE = 'com.example.illegalcapture'


def main():
    validation = ROOT / 'validation'
    validation.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pipeline-', dir=validation) as directory:
        token = secrets.token_hex(24)
        settings = Settings(data=Path(directory), model=ROOT/'models/yolox_s.onnx', token=token, reserve_bytes=0)
        app = create_app(settings)
        store = app.state.store
        config = store.configuration()
        # Vehicle/transport smoke uses the bundled static video; plate inference has separate real-model tests.
        store.configure(config['revision'], {**config['config'], 'plate_enabled': False})
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=61617, log_level='error'))
        serving = threading.Thread(target=server.run, daemon=True)
        serving.start()
        stop = threading.Event()

        def work():
            detector = Detector(settings)
            while not stop.wait(.25):
                process_one(store, detector, 'android-pipeline-test')

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(.1)
            assert server.started, 'Local test API did not start'
            subprocess.run(['adb', 'reverse', 'tcp:61617', 'tcp:61617'], check=True)
            subprocess.run(['adb', 'shell', 'run-as', PACKAGE, 'sh', '-c',
                            "'mkdir -p files && cat > files/pipeline-test-token'"], input=token.encode(), capture_output=True, check=True)
            tests = ['mp4JoinConcatenatesTwoSilentClips', 'dynamicWindowKeepsDecodeableStartAndReportsRecorderGap',
                     'realCacheSurvivesActivityRecreationAndQueuesVideo',
                     'dynamicEvidenceUploadsIdempotentlyAndReceivesIndependentVerdict']
            names = ','.join(f'{PACKAGE}.VehicleSmokeTest#{test}' for test in tests)
            result = subprocess.run(['adb', 'shell', 'am', 'instrument', '-w', '-e', 'class', names,
                                     f'{PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner'],
                                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
            (validation/'pipeline-android-tests.txt').write_text(result.stdout+result.stderr, encoding='utf-8')
            print(result.stdout, flush=True)
            for name in ['pipeline-landscape.png', 'pipeline-portrait.png', 'pipeline-preview.png', 'pipeline-result.json']:
                payload = subprocess.run(['adb', 'exec-out', 'run-as', PACKAGE, 'cat', 'files/'+name], capture_output=True)
                if payload.returncode == 0:
                    (validation/name).write_bytes(payload.stdout)
            assert result.returncode == 0 and 'OK (4 tests)' in result.stdout, 'Device regression failed'
        finally:
            subprocess.run(['adb', 'shell', 'run-as', PACKAGE, 'rm', '-f', 'files/pipeline-test-token'], capture_output=True)
            subprocess.run(['adb', 'reverse', '--remove', 'tcp:61617'], capture_output=True)
            stop.set(); worker.join(timeout=30)
            server.should_exit = True; serving.join(timeout=10)


if __name__ == '__main__':
    main()
