"""Runtime settings chosen from the dashboard, persisted across restarts.

Env vars stay the source of the *defaults*; anything picked in the Settings tab is written
to a small JSON file next to the database and wins over them. Keeping it out of .env means
the UI never has to touch a file that holds API keys.

API keys themselves are NEVER settable or readable here -- the UI only learns whether a
provider has one, so it can grey out the rest.
"""
import json, os, threading
from . import config

SETTINGS_PATH = config.ROOT / "callradar-settings.json"
# Reentrant on purpose: save() holds the lock and then calls load(). With a plain Lock
# that is a self-deadlock -- the request thread never returns and the server wedges.
_lock = threading.RLock()
_cache = None

# Curated per provider. Every Gemini/Groq id below was exercised against this corpus in
# development; the lists are short on purpose -- a picker of forty models helps nobody.
LLM_CATALOGUE = {
    "gemini": {
        "label": "Google Gemini",
        "env": "GEMINI_API_KEY",
        "signup": "aistudio.google.com",
        "models": [
            {"id": "gemini-3.5-flash-lite", "label": "3.5 Flash Lite",
             "note": "~2 s per call. The workhorse for bulk analysis."},
            {"id": "gemini-3.6-flash", "label": "3.6 Flash",
             "note": "~10 s per call; more reasoning budget, better on ambiguous mood."},
            {"id": "gemini-flash-lite-latest", "label": "Flash Lite (latest)",
             "note": "Alias that tracks the current lite release."},
        ],
    },
    "groq": {
        "label": "Groq",
        "env": "GROQ_API_KEY",
        "signup": "console.groq.com",
        "models": [
            {"id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B",
             "note": "Fast. Free tier caps around 200k tokens/day per model."},
            {"id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B",
             "note": "Stronger judgments, burns the daily cap sooner."},
            {"id": "openai/gpt-oss-safeguard-20b", "label": "GPT-OSS Safeguard 20B",
             "note": "Separate daily quota -- useful once the others are exhausted."},
        ],
    },
    "anthropic": {
        "label": "Anthropic",
        "env": "ANTHROPIC_API_KEY",
        "signup": "console.anthropic.com",
        "models": [
            {"id": "claude-sonnet-5", "label": "Claude Sonnet 5",
             "note": "Balanced quality and speed."},
            {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5",
             "note": "Cheapest and fastest of the three."},
            {"id": "claude-opus-5", "label": "Claude Opus 5",
             "note": "Most capable; slowest and priciest."},
        ],
    },
    "heuristic": {
        "label": "Heuristic (no API key)",
        "env": None,
        "signup": None,
        "models": [
            {"id": "heuristic", "label": "Keyword rules",
             "note": "Deterministic and offline. Measured 48.9% resolution agreement with "
                     "an LLM on this corpus -- a fallback, not a peer."},
        ],
    },
}


def provider_available(name):
    """Whether a provider has a key. Returns the availability only, never the key."""
    if name == "heuristic":
        return True
    env = (LLM_CATALOGUE.get(name) or {}).get("env")
    return bool(env and os.getenv(env))


def _defaults():
    return {
        "whisper_model": config.WHISPER_MODEL,
        "whisper_device": config.WHISPER_DEVICE,
        "whisper_compute": config.WHISPER_COMPUTE,
        "llm_backend": "auto",          # auto = first provider with a key
        "llm_model": None,              # None = that provider's first catalogue entry
    }


def load():
    global _cache
    with _lock:
        if _cache is None:
            data = _defaults()
            try:
                if SETTINGS_PATH.exists():
                    data.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
            except Exception:
                pass                    # a corrupt settings file must not break startup
            _cache = data
        return dict(_cache)


def save(updates: dict):
    """Merge and persist. Only known keys are accepted."""
    global _cache
    allowed = set(_defaults())
    clean = {k: v for k, v in updates.items() if k in allowed}
    with _lock:
        data = load()
        data.update(clean)
        SETTINGS_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
        _cache = data
        return dict(data)


def resolve_llm():
    """The backend + model analysis should use, honouring availability.

    Falls back to heuristic rather than erroring: a key that expired mid-batch should
    degrade, not stop the run.
    """
    s = load()
    backend = s.get("llm_backend") or "auto"
    if backend == "auto" or not provider_available(backend):
        for candidate in ("gemini", "groq", "anthropic"):
            if provider_available(candidate):
                backend = candidate
                break
        else:
            backend = "heuristic"
    model = s.get("llm_model")
    known = [m["id"] for m in LLM_CATALOGUE[backend]["models"]]
    if model not in known:
        model = known[0]
    return backend, (None if backend == "heuristic" else model)
