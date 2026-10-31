"""Central configuration. Everything comes from environment variables (or a local .env file)."""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


def _bool(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


BASE = Path(__file__).resolve().parent
ON_RENDER = bool(os.environ.get("RENDER"))          # Render sets RENDER=true automatically

# ---- storage -------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE / "data"))
DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "visionpro.db"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", DATA_DIR / "uploads"))
RECORD_DIR = Path(os.environ.get("RECORD_DIR", DATA_DIR / "recordings"))
SNAP_DIR = Path(os.environ.get("SNAP_DIR", DATA_DIR / "snapshots"))
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", BASE / "videos"))   # demo videos you ship with the repo
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE / "models"))
PERSISTENT_STORAGE = _bool("PERSISTENT_STORAGE", not ON_RENDER)  # set true ONLY if a Render Disk is mounted
RETENTION_DAYS = _int("RETENTION_DAYS", 30)

# ---- detection -----------------------------------------------------------
ALLOWED_MODELS = [m.strip() for m in os.environ.get(
    "ALLOWED_MODELS", "yolo11n.pt,yolo11s.pt,yolo11m.pt").split(",") if m.strip()]
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", ALLOWED_MODELS[0])
IMGSZ = _int("YOLO_IMGSZ", 480 if ON_RENDER else 640)
PROCESS_WIDTH = _int("PROCESS_WIDTH", 640 if ON_RENDER else 960)
TORCH_THREADS = _int("TORCH_THREADS", os.cpu_count() or 1)

# ---- sources / uploads ---------------------------------------------------
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
ALLOW_SERVER_CAMERA = _bool("ALLOW_SERVER_CAMERA", not ON_RENDER)   # cameras attached to the server
ALLOW_PRIVATE_URLS = _bool("ALLOW_PRIVATE_URLS", not ON_RENDER)     # RTSP/HTTP hosts on private networks
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---- security ------------------------------------------------------------
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
LOGIN_MAX_ATTEMPTS = _int("LOGIN_MAX_ATTEMPTS", 5)
LOGIN_WINDOW_SEC = _int("LOGIN_WINDOW_SEC", 300)


def missing_settings():
    """Return a list of problems with the security-related settings (empty list = all good)."""
    errors = []
    if not ADMIN_USERNAME:
        errors.append("ADMIN_USERNAME is not set.")
    if len(ADMIN_PASSWORD) < 8:
        errors.append("ADMIN_PASSWORD is not set or shorter than 8 characters.")
    if DEFAULT_MODEL not in ALLOWED_MODELS:
        errors.append("DEFAULT_MODEL must be one of ALLOWED_MODELS.")
    return errors


def check_required():
    """Fail fast with a readable message instead of running with insecure defaults."""
    errors = missing_settings()
    if errors:
        sys.stderr.write("\nVisionPro cannot start. Fix your environment variables:\n  - "
                         + "\n  - ".join(errors) + "\n(See .env.example)\n\n")
        raise SystemExit(1)


def ensure_dirs():
    for d in (DATA_DIR, UPLOAD_DIR, RECORD_DIR, SNAP_DIR, VIDEO_DIR, MODEL_DIR, DB_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
