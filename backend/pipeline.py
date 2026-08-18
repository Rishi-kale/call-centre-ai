"""Audio -> turn-by-turn transcript.

The recordings are stereo with agent on the LEFT channel and customer on the RIGHT.
That gives us perfect speaker separation for free: split the channels, transcribe each
independently, tag the speaker, and merge by timestamp. No diarization model needed.

Verified on the sample data: channel correlation ~0.001 (fully independent speakers),
left channel leads the call (agent greeting) and is louder.
"""
import subprocess, json, tempfile, os, shutil
from pathlib import Path

_MODEL = None


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
        return turns
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
    m = json.load(open(json_path))
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
