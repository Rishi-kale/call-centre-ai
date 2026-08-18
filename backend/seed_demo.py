"""Populate the DB with synthetic-but-realistic calls so the whole product runs
end-to-end with no audio files, no GPU, and no API key.

Each call gets:
  * a turn-by-turn transcript (real timings + word offsets),
  * a matching stereo MP3 (agent tone left, caller tone right) so the player and the
    click-to-seek evidence feature work,
  * genuine analysis via the real analyze.analyze() heuristic path.
"""
import time, subprocess, random
from pathlib import Path
from .config import AUDIO_DIR
from . import analyze, db

random.seed(7)

# (customer, agent, intent-flavoured caller lines, resolution flavour)
SCRIPTS = [
    ("David Wilson", "Patricia", "fraud_dispute", [
        ("agent", "Thanks for calling Northbank, this is Patricia, how can I help?"),
        ("caller", "Hi, there's a charge on my debit card I don't recognise for ninety pounds."),
        ("agent", "I'm sorry to hear that. Can I take the last four digits of your card?"),
        ("caller", "It's four two one one. This is the second time this has happened."),
        ("agent", "I understand. I've blocked the card and reversed the charge, that's all sorted now."),
        ("caller", "Oh brilliant, thank you so much for sorting that."),
    ], "resolved"),
    ("Sarah Ahmed", "Marcus", "app_login", [
        ("agent", "Good afternoon, Northbank, Marcus speaking."),
        ("caller", "I've been locked out of the app and I can't reset my password."),
        ("agent", "Let me help. Can you confirm the email on the account?"),
        ("caller", "This is ridiculous, it's the third time this week I've had to call."),
        ("agent", "I'm sorry about that. Someone from the tech team will call you back within 48 hours."),
        ("caller", "That's not good enough, I need this fixed today."),
    ], "follow_up_promised"),
    ("Emma Thompson", "Patricia", "loan_mortgage", [
        ("agent", "Northbank, Patricia here, how can I help today?"),
        ("caller", "I wanted to ask about my mortgage repayment going up next month."),
        ("agent", "Of course, your rate changed at the end of your fixed term. I can explain the new figure."),
        ("caller", "Okay, that makes sense, thanks for explaining it clearly."),
        ("agent", "You're all set. Is there anything else I can help with?"),
        ("caller", "No that's great, cheers."),
    ], "resolved"),
    ("James Patel", "Marcus", "card_issue", [
        ("agent", "Hello, Northbank, this is Marcus."),
        ("caller", "My card keeps getting declined and it's really frustrating."),
        ("agent", "Let me check. I can see a temporary block from a security flag."),
        ("caller", "This is unacceptable, I was in a shop and it was so embarrassing."),
        ("agent", "I completely understand. I'll transfer you to my manager in the complaints team."),
        ("caller", "Fine, please do."),
    ], "escalated"),
    ("Olivia Brown", "Grace", "payment_transfer", [
        ("agent", "Northbank, Grace speaking, how can I help?"),
        ("caller", "A transfer I made yesterday hasn't arrived and the payment bounced."),
        ("agent", "Let me look into that transfer for you now."),
        ("caller", "I've been waiting on hold for ages, no one seems to know anything."),
        ("agent", "I'm sorry for the wait. I've resubmitted it and it's all sorted now."),
        ("caller", "Okay, thank you, I appreciate you sorting it."),
    ], "resolved"),
    ("Michael Chen", "Grace", "balance_account", [
        ("agent", "Good morning, Northbank, Grace here."),
        ("caller", "I just wanted to check my balance and a statement charge."),
        ("agent", "Happy to help. Your balance is up to date and I can explain the fee."),
        ("caller", "Perfect, that answers it, thanks."),
        ("agent", "Great, have a lovely day."),
    ], "resolved"),
    ("Sophie Green", "Patricia", "complaint", [
        ("agent", "Northbank, Patricia speaking."),
        ("caller", "I want to make a complaint, this is the fourth time I've called about the same thing."),
        ("agent", "I'm very sorry. Let me log this properly for you."),
        ("caller", "It's a complete waste of my time, honestly useless."),
        ("agent", "I hear you. I'm raising a formal complaint case now."),
        ("caller", "Well, we'll see if anything actually happens this time."),
    ], "unresolved"),
    ("Daniel Evans", "Marcus", "app_login", [
        ("agent", "Hello, Northbank, Marcus speaking, how can I help?"),
        ("caller", "The online banking website won't let me log in."),
        ("agent", "Let's get that fixed. I'll walk you through a reset."),
        ("caller", "Oh that worked, brilliant, thank you."),
        ("agent", "You're all set. Anything else?"),
        ("caller", "No, that's perfect, cheers."),
    ], "resolved"),
]


def _make_stereo_mp3(turns, out_path):
    """Agent turns -> tone on LEFT, caller turns -> tone on RIGHT; silence elsewhere.
    Synthesised in numpy (robust) and encoded to a real stereo 8kHz mp3, so the
    player and the click-to-seek evidence feature are demonstrable without real audio."""
    import numpy as np, soundfile as sf, tempfile, os
    sr = 8000
    total = max(t["end"] for t in turns) + 1.0
    buf = np.zeros((int(total * sr), 2), dtype=np.float32)
    for t in turns:
        freq = 320.0 if t["speaker"] == "agent" else 480.0
        ch = 0 if t["speaker"] == "agent" else 1
        i0, i1 = int(t["start"] * sr), int(t["end"] * sr)
        tt = np.arange(i1 - i0) / sr
        # gentle envelope so it isn't a harsh beep
        env = np.minimum(1.0, np.minimum(tt * 8, (tt[-1] - tt) * 8)) if len(tt) else tt
        buf[i0:i1, ch] += (0.25 * env * np.sin(2 * np.pi * freq * tt)).astype(np.float32)
    wav = tempfile.mktemp(suffix=".wav")
    sf.write(wav, buf, sr)
    subprocess.run(f'ffmpeg -y -v error -i "{wav}" -ar 8000 -ac 2 "{out_path}"',
                   shell=True, check=True)
    os.remove(wav)


def _build_turns(script_lines):
    turns, t = [], 1.0
    for speaker, text in script_lines:
        dur = max(1.5, len(text) / 14)             # ~14 chars/sec speaking
        words = text.split()
        step = dur / max(len(words), 1)
        turns.append({
            "speaker": speaker, "start": round(t, 2), "end": round(t + dur, 2),
            "text": text,
            "words": [{"w": " " + w, "t": round(t + k * step, 2)} for k, w in enumerate(words)],
        })
        t += dur + 0.5
    return turns


def seed():
    db.init_db()
    base_ms = int(time.time() * 1000) - 6 * 3600 * 1000
    for i, (cust, agent, _cat, lines, _res) in enumerate(SCRIPTS):
        sid = f"demo{i:04d}"
        turns = _build_turns(lines)
        meta = {
            "sid": sid, "customer_name": cust, "agent_name": agent,
            "start_ms": base_ms + i * 900_000,
            "handle_time_s": round(turns[-1]["end"], 1) + random.randint(20, 200),
            "caller_mos": random.choice([2.0, 3.0, 3.0, 4.0]),
            "agent_mos": 3.0,
        }
        out_mp3 = AUDIO_DIR / f"{sid}.mp3"
        try:
            _make_stereo_mp3(turns, str(out_mp3))
            dur = round(turns[-1]["end"] + 1, 2)
        except Exception as e:
            print(f"  audio gen skipped for {sid}: {e}")
            dur = round(turns[-1]["end"] + 1, 2)
        result = analyze.analyze(turns, meta, backend="heuristic")
        db.upsert_call({
            "sid": sid, "customer_name": cust, "agent_name": agent,
            "start_ms": meta["start_ms"], "duration_s": dur,
            "handle_time_s": meta["handle_time_s"], "caller_mos": meta["caller_mos"],
            "agent_mos": meta["agent_mos"], "transcript": turns,
            "created_ms": int(time.time() * 1000), **result,
        })
        print(f"  seeded {sid}  {cust}  attention={result['attention']}")
    print(f"Seeded {len(SCRIPTS)} demo calls.")


if __name__ == "__main__":
    seed()
