"""Application settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, replace
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

    # AI agent (chat): empty model = the newest Opus-class model the key can use
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", ""))

    mock_step_delay: float = field(default_factory=lambda: float(os.getenv("MOCK_STEP_DELAY", "0.6")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"


# Settings the user can change from the app (stored in DATA_DIR/settings.json, they win over .env).
USER_KEYS = ("provider", "fal_key", "fal_engine", "local_worker_url", "local_worker_token", "blender_path",
             "anthropic_key")


def user_settings_path(settings: Settings) -> Path:
    return settings.data_dir / "settings.json"


def with_user_settings(settings: Settings) -> Settings:
    path = user_settings_path(settings)
    if not path.is_file():
        return settings
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    return replace(settings, **{k: v for k, v in saved.items() if k in USER_KEYS and isinstance(v, str)})


def save_user_settings(settings: Settings, values: dict) -> Settings:
    updated = replace(settings, **{k: v for k, v in values.items() if k in USER_KEYS})
    path = user_settings_path(updated)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: getattr(updated, k) for k in USER_KEYS}, indent=1), encoding="utf-8")
    return updated


def get_settings() -> Settings:
    return with_user_settings(Settings())
