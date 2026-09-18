"""Run device checks. Pass the dev token via stdin, never command arguments/logs."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = "com.example.illegalcapture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    token = next(line.split("=", 1)[1] for line in (ROOT / ".env").read_text().splitlines()
                 if line.startswith("TRAFFIC_API_TOKEN="))
    apk = ROOT.parent / "IllegalCapture/app/build/outputs/apk"
    for relative in (() if args.skip_install else ("debug/app-debug.apk", "androidTest/debug/app-debug-androidTest.apk")):
        subprocess.run([args.adb, "install", "-r", str(apk / relative)], check=True)
    subprocess.run([args.adb, "reverse", "tcp:61616", "tcp:61616"], check=True)
    subprocess.run([args.adb, "shell", "run-as", PACKAGE, "sh", "-c",
        "'mkdir -p files && cat > files/backend-test-token'"], input=token.encode(), check=True, capture_output=True)
    try:
        result = subprocess.run([args.adb, "shell", "am", "instrument", "-w",
            "-e", "class", f"{PACKAGE}.VehicleSmokeTest", f"{PACKAGE}.test/androidx.test.runner.AndroidJUnitRunner"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        (ROOT / "validation").mkdir(exist_ok=True)
        (ROOT / "validation/android-tests.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0 or "OK (" not in result.stdout or "tests)" not in result.stdout:
            raise SystemExit("Android checks failed; see output above")
        # A skipped token-dependent test is not an end-to-end success.
        saved = subprocess.run([args.adb, "shell", "run-as", PACKAGE, "cat", "files/device-test-result.json"],
            check=True, capture_output=True, text=True, encoding="utf-8")
        task = json.loads(saved.stdout)
        assert task["status"] == "ANALYZED" and task["result"]["submission_allowed"] is False, task
        assert task["review"]["decision"] == "INVALID" and task["effective_result"]["decision"] == "REJECTED"
        (ROOT / "validation/android-result.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
        for name in ('portrait-menu.png', 'landscape-menu.png', 'landscape-recording.png'):
            image = subprocess.run([args.adb, 'exec-out', 'run-as', PACKAGE, 'cat', 'files/'+name],
                capture_output=True, check=True).stdout
            (ROOT / 'validation' / name).write_bytes(image)
    finally:
        subprocess.run([args.adb, "shell", "run-as", PACKAGE, "rm", "-f", "files/backend-test-token"],
            capture_output=True)


if __name__ == "__main__":
    main()
