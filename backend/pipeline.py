"""Audio -> turn-by-turn transcript.

The recordings are stereo with agent on the LEFT channel and customer on the RIGHT.
That gives us perfect speaker separation for free: split the channels, transcribe each
independently, tag the speaker, and merge by timestamp. No diarization model needed.

Verified on the sample data: channel correlation ~0.001 (fully independent speakers),
left channel leads the call (agent greeting) and is louder.
"""
import subprocess, json, tempfile, os, shutil, re
from pathlib import Path

_MODEL = None

# ---------------------------------------------------------------------------
# Domain-term correction
# ---------------------------------------------------------------------------
# Whisper has no idea "Harper Valley National Bank" is a proper noun, so the scripted
# greeting comes out wrong in ~20% of calls (Harbor / Hopper / Hapa / Upper / Hyper...).
# The dataset's own metadata is the ground truth here: every session is named
# "Little Harper Valley N" and the label key is `lhvb_script` (Little Harper Valley Bank).
#
# Substitution is deliberately narrow: only these observed misrecognitions, and only when
# followed by the institutional name. A blanket "<word> Valley" rule would corrupt real
# text -- e.g. "...calling for Valley..." where "for" is a legitimate word.
_BANK_VARIANTS = [
    "harbor", "harbour", "hopper", "hapa", "papa", "papua", "upper", "hyper", "halper",
    "halberd", "harvard", "pepper", "copper", "heather", "hartford", "hubbard", "hover",
    "helper", "happy", "parker", "tapper", "arbor", "huffer", "hoover", "harbert",
]
_BANK_RE = re.compile(
    r"\b(" + "|".join(_BANK_VARIANTS) + r")(\s+Valley\s+(?:National\s+)?Bank)",
    re.IGNORECASE,
)


def correct_domain_terms(text: str) -> str:
    """Fix known proper-noun misrecognitions in one piece of transcript text."""
    return _BANK_RE.sub(lambda m: "Harper" + m.group(2), text)


_VARIANT_SET = set(_BANK_VARIANTS)

# ---------------------------------------------------------------------------
# Turn splitting
# ---------------------------------------------------------------------------
# Whisper's VAD sometimes emits one "segment" spanning two utterances separated by a long
# silence -- e.g. "What is your address?" at 18.6s and "A new checkbook has been sent..."
# at 75.7s arrive as a single turn labelled start=18.56s.
#
# That breaks the one thing the product must get right: evidence cites turn["start"], so a
# quote from the tail of a merged turn points at a timestamp ~1 minute before the words are
# actually spoken. Split on the word timings we already have -- no re-transcription needed.
#
# Threshold: inter-word gaps in this corpus are bimodal (p95 = 1.14s for natural pauses,
# p98 = 5.46s), so 2.0s sits in the valley between "pause" and "separate utterance".
TURN_SPLIT_GAP_S = 2.0


def split_merged_turns(turns, max_gap=TURN_SPLIT_GAP_S):
    """Split any turn whose internal word gap exceeds max_gap into separate turns,
    each carrying its own accurate start/end. Turns without word timings pass through."""
    out = []
    for t in turns:
        words = t.get("words") or []
        if len(words) < 2:
            out.append(t)
            continue
        groups, cur = [], [words[0]]
        for prev, nxt in zip(words, words[1:]):
            if nxt["t"] - prev["t"] > max_gap:
                groups.append(cur)
                cur = [nxt]
            else:
                cur.append(nxt)
        groups.append(cur)
        if len(groups) == 1:
            out.append(t)
            continue
        for i, g in enumerate(groups):
            text = "".join(w["w"] for w in g).strip()
            if not text:
                continue
            out.append({
                "speaker": t["speaker"],
                "start": round(g[0]["t"], 2),
                # last fragment keeps the original end; earlier ones end at their last word
                "end": round(t["end"] if i == len(groups) - 1 else g[-1]["t"], 2),
                "text": text,
                "words": g,
            })
    out.sort(key=lambda x: x["start"])
    return out


def apply_domain_corrections(turns):
    """Apply correct_domain_terms across a transcript's turns (text + word tokens).

    Word tokens are handled separately: they hold one word each (" Harbor", " Valley"),
    so the contiguous regex above can never match them. Instead, rewrite a token when it
    is a known variant AND the next token is "Valley" -- the same institutional-name
    context the text-level rule requires.
    """
    for t in turns:
        t["text"] = correct_domain_terms(t["text"])
        words = t.get("words") or []
        for i, w in enumerate(words):
            bare = w["w"].strip().strip(".,!?;:").lower()
            nxt = words[i + 1]["w"].strip().strip(".,!?;:").lower() if i + 1 < len(words) else ""
            if bare in _VARIANT_SET and nxt == "valley":
                w["w"] = w["w"].replace(w["w"].strip().strip(".,!?;:"), "Harper")
    return turns


def _get_model():
    """Lazy-load faster-whisper once and reuse it for the whole batch."""
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        from .config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE
        _MODEL = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    return _MODEL


def split_channels(mp3_path: str, workdir: str):
    """Left -> agent.wav, Right -> caller.wav, upsampled to 16k for Whisper."""
    agent_wav = os.path.join(workdir, "agent.wav")
    caller_wav = os.path.join(workdir, "caller.wav")
    subprocess.run(
        f'ffmpeg -y -v error -i "{mp3_path}" '
        f'-filter_complex "[0:a]pan=mono|c0=c0[l];[0:a]pan=mono|c0=c1[r]" '
        f'-map "[l]" -ar 16000 "{agent_wav}" '
        f'-map "[r]" -ar 16000 "{caller_wav}"',
        shell=True, check=True,
    )
    return agent_wav, caller_wav


def _transcribe_channel(wav_path: str, speaker: str):
    model = _get_model()
    segments, _ = model.transcribe(
        wav_path, language="en",
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True, beam_size=5,
    )
    turns = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        turns.append({
            "speaker": speaker,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
            "words": [{"w": w.word, "t": round(w.start, 2)} for w in (seg.words or [])],
        })
    return turns


def transcribe_call(mp3_path: str):
    """Return the merged, time-ordered list of conversation turns for one recording."""
    workdir = tempfile.mkdtemp(prefix="cr_")
    try:
        agent_wav, caller_wav = split_channels(mp3_path, workdir)
        turns = _transcribe_channel(agent_wav, "agent") + _transcribe_channel(caller_wav, "caller")
        turns.sort(key=lambda t: t["start"])
        return apply_domain_corrections(split_merged_turns(turns))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def audio_duration_s(mp3_path: str) -> float:
    out = subprocess.run(
        f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{mp3_path}"',
        shell=True, capture_output=True, text=True,
    )
    try:
        return round(float(out.stdout.strip()), 2)
    except ValueError:
        return 0.0


def load_metadata(json_path: str) -> dict:
    """Parse one metadata file. Note the caller-name key literally has spaces."""
    m = json.load(open(json_path, encoding="utf-8"))
    labels = m.get("labels", {}) or {}
    start_ms = m.get("start_time_ms")
    end_ms = m.get("end_time_ms")
    handle = round((end_ms - start_ms) / 1000, 1) if (start_ms and end_ms) else None
    return {
        "sid": m.get("sid") or Path(json_path).stem,
        "agent_name": (m.get("agent", {}).get("metadata", {}) or {}).get("agent_name", "Unknown"),
        "customer_name": (m.get("caller", {}).get("metadata", {}) or {}).get(
            "first and last name", "Unknown"),
        "start_ms": start_ms,
        "handle_time_s": handle,
        "caller_mos": labels.get("caller_mos"),
        "agent_mos": labels.get("agent_mos"),
    }
