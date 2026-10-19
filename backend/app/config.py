"""Application settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader so the app runs without extra dependencies."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("GEN_PROVIDER", "mock").lower())
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")).resolve())
    max_concurrent_jobs: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

    fal_key: str = field(default_factory=lambda: os.getenv("FAL_KEY", ""))
    fal_engine: str = field(default_factory=lambda: os.getenv("FAL_ENGINE", "hunyuan3d_v3"))
    fal_text_to_image_model: str = field(
        default_factory=lambda: os.getenv("FAL_TEXT_TO_IMAGE_MODEL", "fal-ai/flux/schnell")
    )
    fal_depth_model: str = field(default_factory=lambda: os.getenv("FAL_DEPTH_MODEL", "fal-ai/imageutils/depth"))
    fal_timeout_s: float = field(default_factory=lambda: float(os.getenv("FAL_TIMEOUT_S", "600")))

    local_worker_url: str = field(default_factory=lambda: os.getenv("LOCAL_WORKER_URL", "http://127.0.0.1:8001"))
    local_worker_token: str = field(default_factory=lambda: os.getenv("LOCAL_WORKER_TOKEN", ""))
    local_timeout_s: float = field(default_factory=lambda: float(os.getenv("LOCAL_TIMEOUT_S", "1800")))

    blender_path: str = field(default_factory=lambda: os.getenv("BLENDER_PATH", ""))

    mock_step_delay: float = field(default_factory=lambda: float(os.getenv("MOCK_STEP_DELAY", "0.6")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"


def get_settings() -> Settings:
    return Settings()
