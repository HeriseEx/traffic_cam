"""Exercise the running Docker services, optionally recreate them with a queued task."""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from local_api import URL

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    token = next(line.split("=", 1)[1] for line in (ROOT / ".env").read_text().splitlines()
                 if line.startswith("TRAFFIC_API_TOKEN="))
    content = (ROOT / "tests/traffic.mp4").read_bytes()

    def request(path, body=None, headers=None):
        req = urllib.request.Request(URL + path, data=body,
            headers={"Authorization": f"Bearer {token}", **(headers or {})})
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    def upload():
        return request("/v1/tasks", content, {"Content-Type": "video/mp4",
            "X-Event-Metadata": json.dumps({"event_id": str(uuid.uuid4())}),
            "X-Video-SHA256": hashlib.sha256(content).hexdigest()})

    def complete(task_id):
        for _ in range(60):
            task = request(f"/v1/tasks/{task_id}")
            if task["status"] not in ("QUEUED", "PROCESSING"):
                assert task["status"] == "ANALYZED", task
                assert task["result"]["submission_allowed"] is False
                return task
            time.sleep(1)
        raise AssertionError("Worker did not finish in 60 seconds")

    first = complete(upload()["task_id"])
    report = {"completed_task": first["task_id"], "result": first["result"], "container_recreation": False}
    if args.restart:
        subprocess.run(["docker", "compose", "stop", "worker"], cwd=ROOT, check=True)
        try:
            pending = upload()
            assert pending["status"] == "QUEUED"
        finally:
            subprocess.run(["docker", "compose", "up", "-d", "--force-recreate", "--wait"], cwd=ROOT, check=True)
        recovered = complete(pending["task_id"])
        assert request(f"/v1/tasks/{first['task_id']}")["result"] == first["result"]
        report.update(container_recreation=True, recovered_task=recovered["task_id"])
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation/smoke-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "result"}))
    print("Server result:", first["status"], first["result"]["vehicle_observations"])


if __name__ == "__main__":
    main()
