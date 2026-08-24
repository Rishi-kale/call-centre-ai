"""Batch = transcribe + analyse + store, for every matched (audio, metadata) pair.

`ingest_one` is the single code path used by BOTH the offline batch and the live
/upload endpoint, so a call handed to you on demo day flows through exactly what you
validated on the full corpus.
"""
import time, shutil
from pathlib import Path
from .config import AUDIO_DIR, META_DIR
from . import pipeline, analyze, db


def ingest_one(mp3_path: str, json_path: str, backend="auto", force=False, model_id=None) -> str:
    """Transcribe + analyse + store one call. Returns its sid.

    Already-ingested calls are skipped unless force=True: transcription is the expensive
    step (~30s of CPU each), so re-uploading a zip that overlaps what is already stored
    must not redo that work.
    """
    meta = pipeline.load_metadata(json_path)
    sid = meta["sid"]
    if not force and db.get_call(sid):
        return sid
    turns = pipeline.transcribe_call(mp3_path, model_id=model_id)
    result = analyze.analyze(turns, meta, backend=backend)
    row = {
        "sid": sid,
        "customer_name": meta["customer_name"],
        "agent_name": meta["agent_name"],
        "start_ms": meta["start_ms"],
        "duration_s": pipeline.audio_duration_s(mp3_path),
        "handle_time_s": meta["handle_time_s"],
        "caller_mos": meta["caller_mos"],
        "agent_mos": meta["agent_mos"],
        "transcript": turns,
        "created_ms": int(time.time() * 1000),
        **result,
    }
    db.upsert_call(row)
    return sid


def _pairs():
    """Match audio and metadata by stem; skip unmatched files gracefully."""
    audios = {p.stem: p for p in AUDIO_DIR.glob("*.mp3")}
    metas = {p.stem: p for p in META_DIR.glob("*.json")}
    both = sorted(set(audios) & set(metas))
    missing = (set(audios) ^ set(metas))
    if missing:
        print(f"  (skipping {len(missing)} unmatched files)")
    return [(audios[s], metas[s]) for s in both]


def run_batch(limit=None, backend="auto"):
    db.init_db()
    pairs = _pairs()
    if limit:
        pairs = pairs[:limit]
    total = len(pairs)
    print(f"Processing {total} calls (backend={backend})...")
    for i, (mp3, js) in enumerate(pairs, 1):
        try:
            sid = ingest_one(str(mp3), str(js), backend=backend)
            print(f"  [{i}/{total}] {sid}  ok")
        except Exception as e:
            print(f"  [{i}/{total}] {mp3.stem}  FAILED: {e}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "heuristic", "gemini", "groq", "anthropic"])
    a = ap.parse_args()
    run_batch(limit=a.limit, backend=a.backend)
