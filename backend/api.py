"""Read-only API over the pre-computed analysis, plus a live /upload that runs the
same pipeline. Serves the dashboard and the recordings (with range support so the
player can seek to a cited timestamp)."""
import io, zipfile, tempfile, os, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from . import db
from .config import AUDIO_DIR, META_DIR, FRONTEND_DIR

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
async def upload(file: UploadFile = File(...)):
    """Accept a .zip (audio/ + metadata/), or a single .mp3 / .json.
    Runs transcribe -> analyse -> insert and returns the new call ids."""
    from . import batch  # imported here so the API starts even without whisper installed
    raw = await file.read()
    name = (file.filename or "upload").lower()
    tmp = tempfile.mkdtemp(prefix="cr_up_")
    new_sids = []
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                z.extractall(tmp)
            audios = {p.stem: p for p in Path(tmp).rglob("*.mp3")}
            metas = {p.stem: p for p in Path(tmp).rglob("*.json")}
            for sid in sorted(set(audios) & set(metas)):
                shutil.copy(audios[sid], AUDIO_DIR / f"{sid}.mp3")
                new_sids.append(batch.ingest_one(str(audios[sid]), str(metas[sid])))
        else:
            # single file: expect its partner alongside via a .zip normally, but
            # allow a lone mp3+json pair dropped into data/ dirs
            raise HTTPException(400, "upload a .zip containing audio/ and metadata/")
        return JSONResponse({"ingested": new_sids, "count": len(new_sids)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- frontend -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text()
