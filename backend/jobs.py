"""Background ingest queue for uploads.

Why: transcribing a call costs ~30s of CPU, so a zip of 100 recordings takes ~50 minutes.
Doing that inside the HTTP request guarantees a browser/proxy timeout and loses the work.
Instead the upload endpoint hands the zip to this queue, returns a job id immediately, and
the dashboard polls for progress.

Deliberately dependency-free -- a thread plus a queue, no Redis/Celery -- so the project
still runs from a bare `pip install -r requirements.txt`.

One worker thread on purpose: transcription is CPU-bound and the faster-whisper model is a
shared global that is not documented thread-safe, so calls are processed one at a time.
Parallelism belongs in a separate process pool, not here.
"""
import io, os, shutil, tempfile, threading, queue, time, uuid, zipfile
from pathlib import Path

MAX_JOB_HISTORY = 50

_jobs: dict = {}
_order: list = []
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _snapshot(job: dict) -> dict:
    """Public view of a job (drops internal paths)."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        return _snapshot(job) if job else None


def list_jobs(limit: int = 20):
    with _lock:
        ids = _order[-limit:][::-1]
        return [_snapshot(_jobs[i]) for i in ids if i in _jobs]


def create_job(filename: str, blob: bytes, model_id=None) -> dict:
    """Register an upload and queue it. Returns the job snapshot (with its id)."""
    job_id = uuid.uuid4().hex[:12]
    workdir = tempfile.mkdtemp(prefix="cr_job_")
    zpath = os.path.join(workdir, "upload.zip")
    with open(zpath, "wb") as fh:                 # to disk, not held in memory
        fh.write(blob)
    job = {
        "id": job_id,
        "filename": filename,
        "model": model_id,                        # None = server default
        "status": "queued",                       # queued | running | done | failed
        "total": None,                            # unknown until the zip is opened
        "processed": 0,
        "ingested": [],
        "skipped": [],
        "failed": [],
        "error": None,
        "created_ms": _now_ms(),
        "finished_ms": None,
        "_workdir": workdir,
        "_zip": zpath,
    }
    with _lock:
        _jobs[job_id] = job
        _order.append(job_id)
        # bound memory: forget the oldest finished jobs
        while len(_order) > MAX_JOB_HISTORY:
            old = _order.pop(0)
            _jobs.pop(old, None)
    _queue.put(job_id)
    _ensure_worker()
    return _snapshot(job)


def _process(job_id: str):
    from . import batch, db                       # imported late: API starts without whisper
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return
    try:
        job["status"] = "running"
        extract_dir = os.path.join(job["_workdir"], "x")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(job["_zip"]) as z:
            z.extractall(extract_dir)
        audios = {p.stem: p for p in Path(extract_dir).rglob("*.mp3")}
        metas = {p.stem: p for p in Path(extract_dir).rglob("*.json")}
        sids = sorted(set(audios) & set(metas))
        job["total"] = len(sids)
        if not sids:
            job["error"] = "no matching audio/ + metadata/ pairs found in the zip"
            job["status"] = "failed"
            return

        from .config import AUDIO_DIR
        for sid in sids:
            try:
                already = db.get_call(sid) is not None
                # Ingest reads from the extracted temp copy, so do that FIRST and only
                # publish into data/audio/ once it succeeds. Copying first would leave an
                # orphan mp3 with no DB row behind every failed recording.
                batch.ingest_one(str(audios[sid]), str(metas[sid]),
                                 model_id=job.get("model"))
                shutil.copy(audios[sid], Path(AUDIO_DIR) / f"{sid}.mp3")
                (job["skipped"] if already else job["ingested"]).append(sid)
            except Exception as e:                # one bad recording must not sink the job
                job["failed"].append({"sid": sid, "error": str(e)})
            finally:
                job["processed"] += 1
        job["status"] = "done"
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "failed"
    finally:
        job["finished_ms"] = _now_ms()
        shutil.rmtree(job.get("_workdir", ""), ignore_errors=True)


def _loop():
    while True:
        job_id = _queue.get()
        try:
            _process(job_id)
        finally:
            _queue.task_done()


def _ensure_worker():
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="callradar-ingest", daemon=True)
            _worker.start()
