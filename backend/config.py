"""Central config. Everything is overridable via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CALLRADAR_DATA", ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"
META_DIR = DATA_DIR / "metadata"
DB_PATH = Path(os.getenv("CALLRADAR_DB", ROOT / "callradar.db"))
FRONTEND_DIR = ROOT / "frontend"

# Whisper: model size + device. See README for GPU vs CPU.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")          # "cuda" if you have a GPU
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")       # "float16" on GPU

# Analysis backend: "anthropic" if ANTHROPIC_API_KEY is set, else "heuristic".
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "claude-sonnet-4-6")

for d in (AUDIO_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)
