# How a recording becomes an answer

Every layer between a raw 8 kHz phone recording and a scored, evidence-cited judgment on the
dashboard — what runs, in what order, with which model, and which failure each stage exists to
prevent.

Measured over the 1,440 processed calls in `callradar.db`: 23.3 h of audio, 100% LLM-analysed,
2,894 evidence citations.

---

## The shape of the problem

Input is audio, not text. Each call arrives as `audio/<id>.mp3` plus `metadata/<id>.json`, matched
by filename stem. The output must be a per-call judgment — intent, mood and where it shifted,
resolution, a ≤40-word summary, an attention score — where **every claim cites a timestamp and the
words spoken there**. Unsupported evidence is worse than no evidence, so verification is a stage in
its own right.

Two properties drive most of the design:

1. Transcription costs ~30 s of CPU per call, so it happens **once** and is stored. The API never
   re-transcribes.
2. Judgments are cited against transcript turns, which makes **turn timing** — not word accuracy —
   the thing most worth getting right.

## Stereo is the diarization

The recordings are two-channel: **agent on the left, customer on the right**. Splitting them means
each channel is transcribed independently and the speaker tagged at source, so speaker attribution
is exact by construction. No diarization model is involved, and none would beat ground-truth
channel separation.

```
<id>.mp3            left  → agent.wav  → Whisper pass 1 (speaker=agent)  ┐
8 kHz stereo  ──┬──                                                      ├─→ merge by time
                └── right → caller.wav → Whisper pass 2 (speaker=caller) ┘   = one transcript
```

---

## Stage by stage

Both entry points — the offline batch and the live upload — converge on one function,
`batch.ingest_one()`, so a call handed over on demo day flows through exactly the code validated on
the full corpus.

### 01 · Ingest & queue
`backend/jobs.py` · `backend/batch.py`

Offline, `python -m backend.batch` pairs every `audio/` file with its `metadata/` twin by stem.
Live, `POST /api/upload` stores the zip, queues it, and returns **202** with a job id in ~0.3 s; a
single background worker drains the queue while the dashboard polls `/api/jobs/{id}`.

| | |
|---|---|
| In | `.zip` → `audio/` + `metadata/` |
| Out | job id, then per-call results |
| Guard | sid already stored → skip |

One worker on purpose: transcription is CPU-bound and the Whisper handle is a shared global that is
not documented thread-safe. Parallelism belongs in a process pool, not here.

**Why async.** At ~30 s per call a 100-call zip runs ~50 minutes. Processing inside the request
guarantees a proxy timeout *and* loses the work. Re-uploading stored calls is cheap — they are
skipped, not re-transcribed, and reported separately.

### 02 · Metadata parse
`pipeline.load_metadata()`

Names, timings and line-quality labels come from metadata, never from the transcript — so a misheard
name can never corrupt customer grouping. Note the caller-name key literally contains spaces.

```json
{ "sid": "004860b1ab2e4c88",
  "agent":  { "metadata": { "agent_name": "Robert" } },
  "caller": { "metadata": { "first and last name": "Mary Smith" } },
  "start_time_ms": 1590860609249, "end_time_ms": 1590860654497,
  "labels": { "caller_mos": 3.0, "agent_mos": 3.0, "lhvb_script": 5.0 } }
```

`handle_time_s` is derived as `(end_time_ms − start_time_ms) / 1000`; `caller_mos` is a 1–5
line-quality score that later feeds the attention formula.

### 03 · Channel split
`ffmpeg` · `pipeline.split_channels()`

One ffmpeg call emits both mono channels, upsampled to the 16 kHz Whisper expects.

```bash
ffmpeg -y -v error -i <in.mp3> \
  -filter_complex "[0:a]pan=mono|c0=c0[l];[0:a]pan=mono|c0=c1[r]" \
  -map "[l]" -ar 16000 agent.wav \
  -map "[r]" -ar 16000 caller.wav
```

### 04 · Speech to text
faster-whisper (CTranslate2) · model `small`

Each channel is transcribed separately and its turns tagged with that speaker. The model is
lazy-loaded once and reused for the whole batch.

```python
WhisperModel("small", device="cpu", compute_type="int8")
model.transcribe(wav, language="en",
                 vad_filter=True,
                 vad_parameters={"min_silence_duration_ms": 500},
                 word_timestamps=True, beam_size=5)
```

**Word timestamps are not optional** — two later stages depend on them. Turns from both channels are
then concatenated and sorted by start time into one conversation.

> **Tested and rejected:** `large-v3-turbo` ran 2–3× slower on CPU and was *less* stable on 8 kHz
> audio — on one sample call it collapsed a caller's turns into an unpunctuated run-on that `small`
> transcribed cleanly. Its trimmed 4-layer decoder trades robustness for speed.

### 05 · Turn splitting
`pipeline.split_merged_turns()` · gap > 2.0 s

Whisper's VAD sometimes emits one segment spanning two utterances separated by a long silence. The
worst case in this corpus put two agent sentences **56 s apart** in a single turn labelled
`start=18.56`:

```
18.56s  " What is your address?"        ← turn.start
75.67s  " A new checkbook has been…"    ← same turn, 57s later
```

Because evidence cites `turn["start"]`, a quote from the tail pointed a full minute before the words
were spoken — click the timestamp, hear the wrong moment. The fix splits on the word timings already
captured, so no re-transcription is needed.

| | |
|---|---|
| Threshold | 2.0 s inter-word gap |
| Chosen from | p95 = 1.14 s, p98 = 5.46 s |
| Effect | 3,833 → 2 bad gaps |

Gap sizes are bimodal — natural pauses cluster under ~1.1 s, separate utterances above ~5 s — so
2.0 s sits in the empty valley between them rather than being guessed.

### 06 · Domain-term correction
`pipeline.apply_domain_corrections()`

Whisper has no idea *Harper Valley National Bank* is a proper noun, so the scripted greeting came out
wrong in ~20% of calls: *Harbor, Hopper, Hapa, Upper, Hyper, Harvard, Hartford…* Ground truth is in
the dataset's own metadata — every session is named `"Little Harper Valley N"` and the label key is
`lhvb_script`.

Accuracy went **79.7% → 97.0%**.

Deliberately narrow: an allowlist of ~25 observed variants, substituted only inside the
`"… Valley [National] Bank"` context. A blanket `<word> Valley` rule would corrupt real text — e.g.
*"calling **for** Valley National Bank"*.

### 07 · Judgment
`analyze.analyze()` · gemini > groq > anthropic > heuristic

The transcript is rendered as `[t] speaker: text` lines and sent to whichever LLM backend has a key,
which returns strict JSON for intent, mood, resolution and summary. Enum values are pinned in the
prompt — and resolution is defined by *what happened*, not by whether a magic word was spoken:

- **resolved** — the agent addressed the request on this call and the caller ends satisfied or
  neutral. A plain "thanks, that's all, bye" counts; no confirmation phrase required.
- **unresolved** — left unaddressed or still broken, or the caller ends frustrated.
- **escalated** — transferred, or a manager / complaints team brought in.
- **follow_up_promised** — a callback promised *instead of* resolving now.

**Those definitions are load-bearing.** Without them the model reproduced the keyword heuristic's
bias and marked plainly-resolved calls unresolved — a caller thanking an agent who had just answered
them.

Each call gets up to 3 attempts with backoff. A response is rejected — and retried — if
`_validate_shape()` finds a missing key or an out-of-vocabulary mood, because a malformed-but-
successful reply is not an API error and would otherwise pass straight through. Exhausting retries
falls back to the heuristic and **logs loudly**: a silent fallback is how a whole batch quietly runs
on keywords while everyone believes it used the LLM.

### 08 · Heuristic floor
`analyze.heuristic_analysis()`

A deterministic keyword cascade that always works, with no key and no network. Order matters —
escalation and follow-up are checked first because both mean the issue was *not* closed on the call:

```
escalate cues   → escalated
follow-up cues  → follow_up_promised
resolved cues   → resolved
positive close  → resolved      (caller signs off happy)
otherwise       → unresolved
```

> **Kept as a floor, not a peer.** Measured head-to-head on 883 calls it agreed with the LLM on
> resolution only **48.9%** of the time, and the disagreement ran **434 to 17** in one direction — it
> said *unresolved* where the LLM said *resolved*, because it demands an explicit cue and defaults to
> failure. Its escalation cues never fired once across the corpus. That systematic bias is why it is
> a fallback and not a vote in an ensemble.

### 09 · Evidence verification
`analyze._verify_quote()`

Every citation is checked against the transcript before it is stored. A quote must overlap some real
turn by **≥60% of its tokens**; if it does, the timestamp is snapped to that turn's true start, and
if it does not, the citation is **dropped**.

A judgment may survive with no citation; it may never survive with a fabricated one. Across the
corpus all **2,894** stored citations resolve to a real turn whose words match the quote.

### 10 · Attention score
`analyze._score_attention()` · deterministic

The 0–100 score is a fixed formula over the judgments — **never authored by the model** — so it is
reproducible and every point is explainable. Each call stores its own per-factor breakdown, which the
UI renders as a clickable list.

| Factor | Weight | Fires on |
|---|---:|---:|
| unresolved | +30 | 242 |
| ends frustrated | +25 | 5 |
| escalated | +20 | 0 — none in corpus |
| high-stakes intent | +15 | 1 |
| mood shift | +12 | 27 |
| follow-up promised | +12 | 0 — none in corpus |
| ends concerned | +12 | 3 |
| long handle time `>85s` | +8 | 148 |
| poor line quality `MOS≤3.0` | +6 | 265 |

> **Thresholds must match the corpus.** The last two originally used generic call-centre numbers —
> `>240 s` and `MOS ≤ 2.5` — and *neither could ever fire* here: the longest call is 181 s and the
> worst line quality is 3.0. Fourteen points of the scale were dead while being advertised as live
> signals. `python -m backend.calibrate` now reports the right values for whatever corpus is loaded
> and flags any factor that never fires or fires on nearly everything.

### 11 · Store once
SQLite · `backend/db.py`

One row per call. Fields used for ranking and aggregation are real indexed columns; the transcript
and the full evidence blob live in JSON columns.

```
calls( sid PK, customer_name, agent_name, start_ms,
       duration_s, handle_time_s, caller_mos, agent_mos,
       transcript JSON,          -- turns + word timings
       intent_label, intent_cat, start_mood, end_mood,
       shift_t, shift_quote, resolution, summary,
       attention, clarification_count,
       analysis_json JSON,       -- every citation + score breakdown
       created_ms )
idx: customer_name · agent_name · attention DESC · intent_cat
```

Writes happen only during ingest. `upsert_call` is an `INSERT … ON CONFLICT(sid) DO UPDATE`, so
re-analysis is idempotent and never duplicates a call.

### 12 · Serve
FastAPI · `backend/api.py`

The API is read-only over pre-computed analysis — it never transcribes and never calls an LLM. Audio
is served through `FileResponse`, which honours range requests, so the player can seek straight to a
cited second. Endpoint list is in the [README](README.md#api).

---

## Every model and library in the path

Nothing here is trained or fine-tuned. Transcription uses a pretrained checkpoint as-is; judgment is
prompt-driven. No labelled ground truth exists in this dataset, so no supervised model could be
trained or honestly validated against it.

| Layer | What it is | Configuration | Role |
|---|---|---|---|
| Channel split | ffmpeg `pan` filter | `c0=c0` / `c0=c1`, `-ar 16000` | Speaker separation — replaces a diarization model entirely |
| Speech→text | OpenAI Whisper via `faster-whisper` (CTranslate2) | `small`, cpu, `int8`, VAD on, word timestamps, beam 5 | Transcript + word-level timings. Pretrained, not fine-tuned |
| Judgment — primary | Google Gemini (REST) | `gemini-3.5-flash-lite` & siblings, JSON mime, 4096 out | Intent, mood + shift, resolution, summary. 903 calls |
| Judgment — secondary | Groq (OpenAI-compatible SDK) | `gpt-oss-120b`, `gpt-oss-20b`, `gpt-oss-safeguard-20b`, compound | Same contract; used when Gemini quota is spent. 537 calls |
| Judgment — optional | Anthropic SDK | `ANTHROPIC_API_KEY`, `claude-sonnet-4-6` | Same contract; wired but unused in this run |
| Judgment — floor | Rule-based lexicons | intent / mood / resolution / escalation cue lists | Never-fail fallback. 0 calls in the final data |
| Citation check | Token-overlap matcher | ≥0.6 overlap, snap to turn start | Drops hallucinated evidence before it is stored |
| Scoring | Fixed weighted formula | 9 factors, env-tunable thresholds | Attention 0–100. Deterministic, never model-authored |
| Storage | SQLite | Indexed columns + JSON blobs | Write once at ingest; read-only thereafter |
| Serving | FastAPI + uvicorn | Range-request audio, single-file UI | API + dashboard, no build step |

## Cost model

| | |
|---|---|
| Transcription, per call (CPU) | ~30 s |
| LLM judgment, per call | ~2.5 s |
| Per dashboard request | 0 s |
| Full 1,441-call transcription | ~8 h |

Transcription dominates and happens once. Re-analysis is cheap and independent — the stored
transcript can be re-judged without touching audio, which is how the corpus moved from keyword
analysis to full LLM coverage without an 8-hour rerun. Free LLM tiers cap per model per day, so the
runner rotates providers and models, retiring one only after repeated failures rather than on a single
transient error.

## What this pipeline does not know

Judgments describe **what was said**, not what happened afterwards. When an agent says a replacement
card has been ordered, the call is marked resolved — the system has no way to learn whether the card
ever arrived. *Resolved* means resolved on the call, as far as the conversation shows.

Closing that gap is a cross-call question, not a per-call one: if a customer calls back about the same
issue category days after a call marked resolved, that earlier call was probably a false resolution.
The data needed — full per-customer history and intent categories — is already stored.

Two further limits worth stating plainly:

- **Mood is noisy at the margins.** Of the handful of calls labelled *frustrated*, some do not survive
  inspection, and 96% of calls in this corpus end *calm* — the mood-shift feature has genuinely thin
  material to work with.
- **Word error rate cannot be measured.** There are no reference transcripts in the dataset, so
  accuracy claims here are limited to what metadata can verify, such as the bank and customer names.
