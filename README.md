# Call-Centre Radar

Conversation-intelligence over raw bank support-call recordings. Point it at the audio,
it transcribes every call, works out who said what, and builds an admin dashboard that
surfaces the calls a manager actually needs to see today — every judgment backed by the
exact moment in the call that justifies it.

## What it does

- **Transcription.** Each recording is stereo: agent on the **left** channel, customer on
  the **right**. We split the channels, transcribe each independently with Whisper, tag
  the speaker, and merge by timestamp. Perfect diarisation, no diarisation model.
- **Per-call intelligence.** Intent, mood + the moment it shifted, resolution status, a
  ≤40-word summary, and a 0–100 needs-attention score.
- **Evidence on everything.** Every judgment cites a timestamp and the verbatim words
  spoken there. Quotes are verified against the transcript, so a hallucinated citation is
  dropped rather than shown.
- **Cross-call views.** Ranked "needs a manager's attention today", trending issues, and
  per-agent volume / handle-time / outcomes.
- **Dashboard.** Customer list → call history → per-call view with the playable recording,
  the transcript, the summary, and a mood timeline. **Click any timestamp or evidence
  quote and the player jumps there and plays it.**

**Developers:** [ARCHITECTURE.md](ARCHITECTURE.md) walks the whole pipeline stage by stage — every
model, every configuration, and the failure each stage exists to prevent.

## Requirements

- Python 3.10+
- **ffmpeg** on your PATH (`apt install ffmpeg` / `brew install ffmpeg`)
- Optional: an NVIDIA GPU for fast batch transcription
- Optional: `ANTHROPIC_API_KEY` for the richer LLM analysis (a deterministic heuristic
  runs without it)

## Quick start — see it working in 60 seconds

No audio, GPU, or API key needed. This seeds synthetic-but-realistic calls (with playable
audio) through the real analysis pipeline.

```bash
pip install -r requirements.txt
make demo          # populate the database + demo audio
make run           # http://127.0.0.1:8000
```

Open the dashboard, click a call, then click a timestamp in the transcript.

## Running on the real recordings (from scratch)

```bash
pip install -r requirements.txt

# 1. Unzip the dataset so files land here:
#      data/audio/<id>.mp3
#      data/metadata/<id>.json
unzip callradar-data.zip -d data/        # adjust if the zip nests differently

# 2. (GPU recommended) configure Whisper for speed
cp .env.example .env      # then uncomment the GPU lines, and API key if you have one
source .env

# 3. Transcribe + analyse everything (writes to callradar.db, once).
make batch                # or: make batch LIMIT=20   to smoke-test first

# 4. Serve the API + dashboard
make run                  # http://127.0.0.1:8000
```

Transcription runs **once** and is stored in SQLite; the API never re-transcribes.

**Compute note.** 1,441 calls is a lot of audio. On a GPU (`WHISPER_DEVICE=cuda`) the full
batch is a few hours — start it early. On CPU it is too slow for the full set; use
`make batch LIMIT=…` to process a subset, or run the batch on a GPU box.

## Live upload (demo day)

The dashboard has an **Upload recordings** button. Hand it a `.zip` containing `audio/` and
`metadata/` folders and it runs the *same* pipeline — transcribe → analyse → store — and the
new calls appear immediately in the customer list and the attention ranking. The upload and
the offline batch share one code path (`backend/batch.ingest_one`), so a call handed to you
on the day flows through exactly what was validated on the full corpus.

## API

| Method | Path | Returns |
|---|---|---|
| GET | `/api/customers` | every customer, call count, worst attention score |
| GET | `/api/customers/{name}/calls` | that customer's full call history |
| GET | `/api/calls/{sid}` | transcript (turns + timings), intent, mood + shift timestamp, resolution, summary, attention score, and every evidence citation |
| GET | `/api/attention?limit=50` | calls ranked by needs-attention score |
| GET | `/api/trends` | volume + resolution rate by issue |
| GET | `/api/trends/timeline?days=7` | average attention score per recording day |
| GET | `/api/agents` | per-agent volume, handle time, outcomes, clarification signal |
| GET | `/api/stats` | corpus totals for the dashboard stat cards |
| GET | `/audio/{sid}.mp3` | the recording (supports range requests → seeking) |
| POST | `/api/upload` | queue a `.zip` of new calls → **202** with a job id |
| GET | `/api/jobs/{id}` | that ingest job's progress |
| GET | `/api/jobs?limit=20` | recent ingest jobs |

### Uploads are asynchronous

Transcribing one call costs ~30s of CPU, so a 100-call zip takes ~50 minutes — far too long
to hold an HTTP request open. `POST /api/upload` therefore stores the zip, queues it, and
returns a job id immediately; the dashboard polls `/api/jobs/{id}` and shows live
`processed / total` progress. A single background worker processes calls one at a time
(the Whisper model is a shared global and CPU is the bottleneck, so parallelism there would
not help).

Add `?wait=true` to process inline and get the finished job back in one response — handy
for small uploads and scripted tests, not for large batches.

Re-uploading calls that are already stored is cheap: they are **skipped**, not
re-transcribed, and reported separately from newly ingested ones. Ingest failures are
per-call — one unreadable recording never sinks the rest of the zip, and leaves no
half-written row or orphan audio file behind.

## How the needs-attention score works

It is a **deterministic formula**, not a model guess, so it is explainable and defensible.
Factors and weights: unresolved (+30) / escalated (+20) / follow-up promised (+12); ends
frustrated (+25) / concerned (+12); a mood shift during the call (+12); high-stakes intent
like fraud or a complaint (+15); long handle time (+8); poor line quality via `caller_mos`
(+6). Each call's per-factor breakdown is returned in `analysis_json.attention.reasons` and
shown in the UI.

**Thresholds are calibrated to the corpus.** The last two factors originally used generic
call-centre numbers (>240s handle time, MOS ≤ 2.5) and *neither could ever fire* on this
data — the longest call is 181s and the worst line quality is 3.0, so 14 points of the
0–100 scale were dead while being advertised as live signals. They now default to
`LONG_CALL_S=85` (p90 of handle time) and `POOR_MOS=3.0` (the worst tier actually present),
overridable via `CALLRADAR_LONG_CALL_S` / `CALLRADAR_POOR_MOS`.

`POOR_MOS` generalises to new data (it is a lower bound), but `LONG_CALL_S` is
corpus-relative — on a corpus of ten-minute calls, 85s would flag nearly everything. After
ingesting different data, run:

```bash
python -m backend.calibrate
```

It prints the recommended thresholds and warns when a factor never fires (dead weight) or
fires on most calls (noise). Thresholds are deliberately *fixed* rather than computed live,
so a call's score never drifts as unrelated calls are ingested.

## Project layout

```
backend/
  pipeline.py     channel split + Whisper transcription + turn splitting + name fixes
  analyze.py      intent / mood / resolution / summary + attention scoring + quote verify
  batch.py        ingest_one() — shared by the offline batch and the live upload
  jobs.py         background ingest queue behind POST /api/upload
  calibrate.py    prints attention thresholds recommended for the loaded corpus
  seed_demo.py    synthetic demo data (no audio/GPU/key needed)
  db.py           SQLite schema + queries
  api.py          FastAPI app, endpoints, audio + frontend serving
frontend/
  index.html      the dashboard (single file, no build step)
data/
  audio/          <id>.mp3
  metadata/       <id>.json
```

## Design notes

- **Storage:** SQLite. One file, trivial to reproduce, ample for 1,441 calls. Transcript
  and full evidence blob live in JSON columns; the judgments used for ranking and
  aggregation are real indexed columns.
- **Analysis backends:** `heuristic` (deterministic, always available), `groq` (free tier,
  console.groq.com), and `anthropic`. Auto-selects `groq` > `anthropic` > `heuristic` based
  on which API key is set. All three pass through the same validation: quotes verified
  against the transcript, summary capped at 40 words, and the attention score computed by
  the fixed formula above — never by the model.
