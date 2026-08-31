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
- **A fixed intent taxonomy.** `intent_cat` is always one of eleven banking categories
  (card replacement, checkbook request, balance inquiry, funds transfer, bill payment,
  password reset, branch info, appointment scheduling, fraud dispute, loan/mortgage,
  general enquiry). The model's own wording is kept in `intent_label`, but the category is
  normalised on the way in — the LLMs produced 194 distinct labels for these same topics,
  and anything that groups or filters needs a stable enum.
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
| GET | `/api/customers` | **paged** — every customer, call count, worst attention score |
| GET | `/api/customers/{name}/calls` | that customer's full call history |
| GET | `/api/calls/{sid}/context` | sibling calls: this customer's history + others with the same intent |
| GET | `/api/calls/{sid}` | transcript (turns + timings), intent, mood + shift timestamp, resolution, summary, attention score, and every evidence citation |
| GET | `/api/attention` | **paged** — calls ranked by needs-attention score |
| GET | `/api/trends` | volume + resolution rate by issue |
| GET | `/api/trends/timeline?days=7` | average attention score per recording day |
| GET | `/api/agents` | **paged** — per-agent volume, handle time, outcomes, clarification signal |
| GET | `/api/stats` | corpus totals for the dashboard stat cards |
| GET | `/api/filters` | available intent / resolution filter values, with counts |
| GET | `/api/models` | Whisper model catalogue, detected hardware, and the recommended model |
| GET | `/api/settings` | current model choices, the options, and which LLM providers have keys |
| PUT | `/api/settings` | change the transcription or analysis model (validated, persisted) |
| GET | `/audio/{sid}.mp3` | the recording (supports range requests → seeking) |
| POST | `/api/upload` | queue a `.zip` of new calls → **202** with a job id. `?model=` overrides the transcription model for that upload |
| GET | `/api/jobs/{id}` | that ingest job's progress |
| GET | `/api/jobs?limit=20` | recent ingest jobs |

### Paged endpoints

`/api/customers`, `/api/attention` and `/api/agents` are paginated server-side and share one
envelope:

```json
{ "content": [ … ], "page": 0, "size": 20, "totalElements": 1440, "totalPages": 72,
  "numberOfElements": 20, "first": true, "last": false, "empty": false }
```

| Param | Default | Notes |
|---|---|---|
| `page` | `0` | 0-indexed; negative values are rejected with 422 |
| `size` | `20` | 1–200; outside that range is rejected with 422 |
| `q` | — | server-side search (name / agent / summary / intent) |
| `sort` | — | `/api/attention`: `score`, `customer`, `agent`, `mood`, `resolution` · `/api/customers`: `name`, `calls`, `last_contact`, `attention` · `/api/agents`: `name`, `calls`, `score`. Unknown fields return 400 |
| `order` | `asc` | `asc` \| `desc`; anything else returns 422 |
| `intent` | — | `/api/attention` only. Repeatable: `?intent=app_login&intent=card_issue` |
| `resolution` | — | `/api/attention` only. Repeatable, e.g. `?resolution=unresolved` |

Filter values are validated against what `/api/filters` reports, so a stale or misspelled
value returns 400 rather than silently returning everything. Within one filter the values
are OR'd (several intents at once); across filters they are AND'd (that intent **and**
unresolved) — what checkbox groups normally imply.

`sort=mood` orders by **severity**, not alphabetically — `desc` surfaces frustrated
callers first, where A–Z would put them behind "calm" and "concerned".

A sort column cannot be a bound SQL parameter, so it has to be concatenated into the
`ORDER BY`. Accepted fields are therefore checked against a fixed allowlist and anything
else is rejected — client input never reaches the SQL.

Sorting is **total**, not just "good enough": each `ORDER BY` ends with a unique tiebreaker
(`customer_name`, `agent_name`, `sid`). Without it, ties in the leading key leave row order
undefined under `LIMIT`/`OFFSET`, so the same row can appear on two pages while another is
skipped — 859 calls here share an attention score of 0, so the corpus would hit that
immediately. Paging the full 1,440 calls returns each row exactly once.

### Settings tab

The dashboard's **Settings** tab is the single place the models are chosen:

- **Transcription (Whisper)** — the catalogue, this machine's hardware, the recommendation
  and its reasoning. Picking one persists to `callradar-settings.json` and applies to every
  later transcription.
- **Analysis (LLM)** — Gemini, Groq and Anthropic with 2–3 models each, plus the
  deterministic heuristic. A provider with no API key is greyed out and names the env var
  it needs. **Automatic** follows whichever provider has a key and falls back to the
  heuristic.

API keys are never sent to the browser or settable through the API — the UI only learns
*whether* a provider has one. Settings are validated server-side: an unknown model, or a
provider without a key, returns 400 rather than being stored.

`.env` is now loaded by `backend/config.py` at import. Previously the keys only reached the
process if the shell had exported them, so `make run` (and `source .env`, which does not
exist on Windows) left the API silently using the heuristic backend.

### Choosing a transcription model

`GET /api/models` reports the model catalogue, what the machine actually has (CUDA device
and VRAM, CPU cores, total and **free** RAM), and a recommendation with its reasoning. The
Bulk Upload page shows this and preselects the recommendation; `?model=` on the upload
overrides it for that batch only, without changing the server default.

Detection needs no extra dependency: CUDA presence and compute types come from
`ctranslate2` (already required by faster-whisper), VRAM from `nvidia-smi` when present,
RAM from the OS.

The recommendation is deliberately conservative on CPU — it weighs **free** RAM, not just
total, because a model that swaps is slower than a smaller one that fits. Measured on this
corpus: `small`/int8 runs ~28 s per one-minute call on 12 cores; `large-v3` is ~6x that on
CPU. `large-v3-turbo` is listed but not recommended — tested on this 8 kHz audio it was
both slower than `small` *and* less stable, garbling one caller's turns into a run-on.

Models are cached per `(model, device, compute_type)` and the cache holds two at most,
evicting the oldest — each resident model is real memory (~1 GB for small, ~4.5 GB for
large-v3).

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
Factors and weights: escalated (+35) / unresolved (+30) / follow-up promised (+12); ends
frustrated (+25) / concerned (+12); a mood shift during the call (+12); high-stakes intent
like fraud or a complaint (+15); long handle time (+8); poor line quality via `caller_mos`
(+6). Each call's per-factor breakdown is returned in `analysis_json.attention.reasons` and
shown in the UI.

**Escalated outranks unresolved.** An escalated call is unresolved *and* handed to another
team without closure, so it must score higher; it previously sat at +20 against
unresolved's +30, which ranked the worse outcome lower.

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
