"""Read-only API over the pre-computed analysis, plus a live /upload that runs the
same pipeline. Serves the dashboard and the recordings (with range support so the
player can seek to a cited timestamp)."""
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from . import db
from .config import AUDIO_DIR, FRONTEND_DIR

app = FastAPI(title="Call-Centre Radar")


@app.on_event("startup")
def _startup():
    db.init_db()


# ---- dashboard views ------------------------------------------------------
Page = Query(0, ge=0, description="0-indexed page number")
Size = Query(db.DEFAULT_PAGE_SIZE, ge=1, le=db.MAX_PAGE_SIZE,
             description=f"rows per page (max {db.MAX_PAGE_SIZE})")


Order = Query("asc", pattern="^(asc|desc)$", description="asc | desc")


def _check_sort(sort, allowed):
    """Reject an unknown sort field rather than silently ignoring it -- a typo in a client
    should surface, not quietly return default-ordered data."""
    if sort and sort not in allowed:
        raise HTTPException(400, f"sort must be one of: {', '.join(sorted(allowed))}")
    return sort


@app.get("/api/customers")
def customers(page: int = Page, size: int = Size, q: str | None = None,
              sort: str | None = None, order: str = Order):
    """Paged. Returns {content, page, size, totalElements, totalPages, first, last}.
    Sortable by: name, calls, last_contact, attention."""
    _check_sort(sort, db.CUSTOMER_SORTS)
    return db.list_customers(page, size, q, sort, order)


@app.get("/api/customers/{name}/calls")
def customer_calls(name: str):
    return db.customer_calls(name)


@app.get("/api/calls/{sid}")
def call(sid: str):
    c = db.get_call(sid)
    if not c:
        raise HTTPException(404, "call not found")
    return c


@app.get("/api/calls/{sid}/context")
def call_context(sid: str, limit: int = 12):
    """Sibling calls for the call-detail view: this customer's other calls, and other
    customers hitting the same issue."""
    ctx = db.call_context(sid, limit)
    if not ctx:
        raise HTTPException(404, "call not found")
    return ctx


@app.get("/api/models")
def models():
    """Whisper models, what this machine can run, and what we recommend for it.

    The upload UI reads this so the model choice is made against real hardware rather
    than a guess.
    """
    from . import hardware, pipeline
    from .config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE
    hw = hardware.detect()
    return {
        "catalogue": hardware.MODEL_CATALOGUE,
        "hardware": hw,
        "recommended": hardware.recommend(hw),
        "current": {"model": WHISPER_MODEL, "device": WHISPER_DEVICE,
                    "compute_type": WHISPER_COMPUTE},
        "loaded": pipeline.loaded_models(),
    }


@app.get("/api/settings")
def get_settings():
    """Everything the Settings tab needs: current choices, the options, and what this
    machine can run. API keys are never returned -- only whether a provider has one."""
    from . import hardware, pipeline, settings as st
    cur = st.load()
    hw = hardware.detect()
    backend, model = st.resolve_llm()
    providers = []
    for name, spec in st.LLM_CATALOGUE.items():
        providers.append({
            "id": name, "label": spec["label"], "signup": spec["signup"],
            "env": spec["env"], "available": st.provider_available(name),
            "models": spec["models"],
        })
    return {
        "settings": cur,
        "effective_llm": {"backend": backend, "model": model},
        "providers": providers,
        "whisper": {
            "catalogue": hardware.MODEL_CATALOGUE,
            "hardware": hw,
            "recommended": hardware.recommend(hw),
            "loaded": pipeline.loaded_models(),
        },
    }


@app.put("/api/settings")
def put_settings(payload: dict):
    """Update settings. Validates every value so a bad write cannot wedge the pipeline."""
    from . import hardware, settings as st
    model = payload.get("whisper_model")
    if model is not None and model not in hardware.MODEL_IDS:
        raise HTTPException(400, f"unknown whisper_model. valid: {', '.join(hardware.MODEL_IDS)}")
    backend = payload.get("llm_backend")
    if backend is not None and backend != "auto" and backend not in st.LLM_CATALOGUE:
        raise HTTPException(400, f"unknown llm_backend. valid: auto, {', '.join(st.LLM_CATALOGUE)}")
    if backend and backend != "auto" and not st.provider_available(backend):
        raise HTTPException(400, f"{backend} has no API key configured")
    llm_model = payload.get("llm_model")
    if llm_model is not None and backend and backend in st.LLM_CATALOGUE:
        known = [m["id"] for m in st.LLM_CATALOGUE[backend]["models"]]
        if llm_model not in known:
            raise HTTPException(400, f"unknown llm_model for {backend}. valid: {', '.join(known)}")
    saved = st.save(payload)
    eff_backend, eff_model = st.resolve_llm()
    return {"settings": saved, "effective_llm": {"backend": eff_backend, "model": eff_model}}


@app.get("/api/filters")
def filters():
    """Available filter values with counts, for the Needs Attention checkbox groups.
    Returns {intents:[{value,n,unresolved}], resolutions:[{value,n}]}."""
    return db.filter_facets()


@app.get("/api/attention")
def attention(page: int = Page, size: int = Size, q: str | None = None,
              intent: list[str] | None = Query(None, description="repeatable; OR'd together"),
              resolution: list[str] | None = Query(None, description="repeatable; OR'd together"),
              sort: str | None = None, order: str = Order):
    """Paged, worst score first. Same envelope as /api/customers.

    Filters are repeatable query params: ?intent=app_login&intent=card_issue&resolution=unresolved
    Values are validated against what /api/filters reports, so a stale or misspelled
    filter fails loudly instead of silently returning everything.

    Sortable by: score, customer, agent, mood, resolution. `q` matches customer name,
    agent name, summary and intent.
    """
    _check_sort(sort, db.ATTENTION_SORTS)
    if intent or resolution:
        facets = db.filter_facets()
        for name, got, allowed in (
            ("intent", intent, {f["value"] for f in facets["intents"]}),
            ("resolution", resolution, {f["value"] for f in facets["resolutions"]}),
        ):
            bad = sorted(set(got or []) - allowed)
            if bad:
                raise HTTPException(
                    400, f"unknown {name} value(s): {', '.join(bad)}. "
                         f"valid: {', '.join(sorted(allowed))}")
    return db.attention_ranked(page, size, q, intent, resolution, sort, order)


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
def agents(page: int = Page, size: int = Size, q: str | None = None,
           sort: str | None = None, order: str = Order):
    """Paged, busiest agent first. Same envelope as /api/customers.
    Sortable by: name, calls, score."""
    _check_sort(sort, db.AGENT_SORTS)
    return db.agent_stats(page, size, q, sort, order)


# ---- audio (range requests handled by FileResponse -> player can seek) -----
@app.get("/audio/{sid}.mp3")
def audio(sid: str):
    p = AUDIO_DIR / f"{sid}.mp3"
    if not p.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(p, media_type="audio/mpeg")


# ---- live upload: same pipeline as the batch ------------------------------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), wait: bool = False,
                 model: str | None = None):
    """Accept a .zip of audio/ + metadata/ and queue it for ingest.

    Returns 202 with a job id immediately -- transcription costs ~30s per call, so a
    100-call zip would run ~50 minutes and time out if processed inside the request.
    Poll /api/jobs/{id} for progress.

    wait=true processes inline and returns the finished job instead. Handy for small
    uploads and scripted tests; do not use it for large batches.
    """
    from . import jobs, hardware
    name = (file.filename or "upload").lower()
    if not name.endswith(".zip"):
        raise HTTPException(400, "upload a .zip containing audio/ and metadata/")
    if model and model not in hardware.MODEL_IDS:
        raise HTTPException(400, f"unknown model. valid: {', '.join(hardware.MODEL_IDS)}")
    raw = await file.read()
    job = jobs.create_job(file.filename or "upload.zip", raw, model_id=model)
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
