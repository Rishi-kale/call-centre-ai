"""Read-only API over the pre-computed analysis, plus a live /upload that runs the
same pipeline. Serves the dashboard and the recordings (with range support so the
player can seek to a cited timestamp)."""
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from . import db
from .config import AUDIO_DIR, FRONTEND_DIR

app = FastAPI(title="Call-Centre Radar")


@app.on_event("startup")
def _startup():
    db.init_db()


# ---- dashboard views ------------------------------------------------------
@app.get("/api/customers")
def customers():
    return db.list_customers()


@app.get("/api/customers/{name}/calls")
def customer_calls(name: str):
    return db.customer_calls(name)


@app.get("/api/calls/{sid}")
def call(sid: str):
    c = db.get_call(sid)
    if not c:
        raise HTTPException(404, "call not found")
    return c


@app.get("/api/attention")
def attention(limit: int = 50):
    return db.attention_ranked(limit)


@app.get("/api/trends")
def trends():
    return db.trends()


@app.get("/api/trends/timeline")
def trends_timeline(days: int = 7):
    return db.attention_timeline(days)


@app.get("/api/stats")
def stats():
    return db.overview_stats()


@app.get("/api/agents")
def agents():
    return db.agent_stats()


# ---- audio (range requests handled by FileResponse -> player can seek) -----
@app.get("/audio/{sid}.mp3")
def audio(sid: str):
    p = AUDIO_DIR / f"{sid}.mp3"
    if not p.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(p, media_type="audio/mpeg")


# ---- live upload: same pipeline as the batch ------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), wait: bool = False):
    """Accept a .zip of audio/ + metadata/ and queue it for ingest.

    Returns 202 with a job id immediately -- transcription costs ~30s per call, so a
    100-call zip would run ~50 minutes and time out if processed inside the request.
    Poll /api/jobs/{id} for progress.

    wait=true processes inline and returns the finished job instead. Handy for small
    uploads and scripted tests; do not use it for large batches.
    """
    from . import jobs
    name = (file.filename or "upload").lower()
    if not name.endswith(".zip"):
        raise HTTPException(400, "upload a .zip containing audio/ and metadata/")
    raw = await file.read()
    job = jobs.create_job(file.filename or "upload.zip", raw)
    if not wait:
        return JSONResponse(job, status_code=202)
    while True:                                   # inline mode: drain then report
        cur = jobs.get_job(job["id"])
        if cur is None or cur["status"] in ("done", "failed"):
            return JSONResponse(cur or job)
        await asyncio.sleep(0.4)


@app.get("/api/jobs")
def jobs_list(limit: int = 20):
    from . import jobs
    return jobs.list_jobs(limit)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    from . import jobs
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# ---- frontend -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
