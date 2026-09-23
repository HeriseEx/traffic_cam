import os
from dataclasses import dataclass, field
from pathlib import Path


def _open_access():
    return os.getenv("TRAFFIC_OPEN", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data: Path = Path(os.getenv("TRAFFIC_DATA_DIR", "data")).resolve()
    model: Path = Path(os.getenv("TRAFFIC_MODEL", "models/yolox_s.onnx")).resolve()
    token: str = os.getenv("TRAFFIC_API_TOKEN", "")
    open: bool = field(default_factory=_open_access)
    max_bytes: int = int(os.getenv("TRAFFIC_MAX_BYTES", str(200 * 1024 * 1024)))
    evidence_bytes: int = 50 * 1024 * 1024
    retention_hours: float = float(os.getenv("TRAFFIC_RETENTION_HOURS", str(30 * 24)))
    max_seconds: float = float(os.getenv("TRAFFIC_MAX_SECONDS", "600"))
    sample_fps: float = 2
    lease_seconds: int = 300
    max_attempts: int = 3
    threshold: float = float(os.getenv("TRAFFIC_THRESHOLD", "0.35"))
    threads: int = int(os.getenv("TRAFFIC_THREADS", "4"))
    reserve_bytes: int = int(os.getenv("TRAFFIC_RESERVE_BYTES", str(256 * 1024 * 1024)))

    def __post_init__(self):
        if not 0 < self.max_bytes <= 200 * 1024 * 1024:
            raise ValueError("TRAFFIC_MAX_BYTES must be between 1 and 200 MiB")
        if not 0 < self.threshold < 1 or not 1 <= self.threads <= 32:
            raise ValueError("Invalid threshold or thread count")
        if self.retention_hours <= 0 or self.reserve_bytes < 0:
            raise ValueError("Invalid retention or disk reserve")
        self.data.mkdir(parents=True, exist_ok=True)
        (self.data / "videos").mkdir(exist_ok=True)

    def require_token(self):
        if self.open:
            return
        if len(self.token) < 24:
            raise ValueError("Set TRAFFIC_API_TOKEN to a random token of at least 24 characters")
