"""Central config. Everything is overridable via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Read .env into the environment if it exists.

    Without this the API keys only reach the process when the shell exported them, so
    `make run` / uvicorn started any other way silently fell back to the heuristic
    backend -- and `source .env` from the README does not exist on Windows at all.
    Real environment variables always win, so an explicit export still overrides the file.
    """
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass                      # a malformed .env must not stop the app from booting


_load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.getenv("CALLRADAR_DATA", ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"
META_DIR = DATA_DIR / "metadata"
DB_PATH = Path(os.getenv("CALLRADAR_DB", ROOT / "callradar.db"))
FRONTEND_DIR = ROOT / "frontend"

# Whisper: model size + device. See README for GPU vs CPU.
# Attention-score thresholds. Defaults are calibrated to the 1,441-call corpus shipped
# with the project (p90 handle time = 85s; lowest line quality present = 3.0). A corpus
# with different call lengths needs LONG_CALL_S recalibrated -- run
# `python -m backend.calibrate` to print the recommended values for whatever is loaded.
LONG_CALL_S = float(os.getenv("CALLRADAR_LONG_CALL_S", 85))
POOR_MOS = float(os.getenv("CALLRADAR_POOR_MOS", 3.0))

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")          # "cuda" if you have a GPU
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")       # "float16" on GPU

# Analysis backend, when backend="auto": gemini > groq > anthropic > heuristic.
# Gemini's free tier has the most generous daily allowance, so it is tried first.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "claude-sonnet-4-6")

for d in (AUDIO_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)
