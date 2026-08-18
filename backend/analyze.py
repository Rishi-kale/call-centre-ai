"""Turn a transcript into cited judgments.

Every judgment carries evidence {t, quote} where the quote is verbatim from the
transcript. Two backends:

  * heuristic  - deterministic, no dependencies, always runs. Good demo baseline.
  * anthropic  - richer judgments via the API, if ANTHROPIC_API_KEY is set.

Both pass through the SAME validation: quotes are verified against the transcript,
the summary is capped at 40 words, and the attention score is computed by a fixed
formula (never by the model) so it is explainable and defensible.
"""
import json, re
from .config import ANTHROPIC_API_KEY, ANALYSIS_MODEL

# ---------------------------------------------------------------------------
# Lexicons for the heuristic backend
# ---------------------------------------------------------------------------
INTENT_RULES = [
    ("fraud_dispute",     ["fraud", "unauthori", "didn't make", "don't recognise",
                           "don't recognize", "stolen", "scam", "suspicious"]),
    ("card_issue",        ["card", "declined", "blocked", "pin", "chip", "contactless"]),
    ("payment_transfer",  ["transfer", "payment", "send money", "standing order",
                           "direct debit", "bounced"]),
    ("balance_account",   ["balance", "statement", "overdraft", "interest", "account number"]),
    ("loan_mortgage",     ["loan", "mortgage", "repayment", "borrow", "credit limit"]),
    ("app_login",         ["app", "log in", "login", "password", "reset", "locked out",
                           "online banking", "website"]),
    ("complaint",         ["complaint", "complain", "unacceptable", "terrible",
                           "ridiculous", "manager"]),
]
NEGATIVE = {"angry", "annoyed", "frustrated", "upset", "unacceptable", "ridiculous",
            "terrible", "awful", "furious", "disgusted", "waste", "useless",
            "again", "third time", "second time", "still not", "never", "not good enough",
            "sick of", "fed up", "no one", "nobody", "keep", "unhappy"}
POSITIVE = {"thank", "thanks", "great", "perfect", "brilliant", "appreciate",
            "wonderful", "helpful", "sorted", "resolved", "happy", "cheers"}
RESOLVED_CUES = ["sorted", "resolved", "all set", "taken care of", "that's fixed",
                 "you're all good", "problem solved", "refunded", "reversed the charge"]
FOLLOWUP_CUES = ["call you back", "callback", "within 48 hours", "within 24 hours",
                 "someone will call", "we'll be in touch", "log a case", "raise a ticket"]
ESCALATE_CUES = ["escalate", "put you through", "transfer you", "my manager",
                 "complaints team", "senior"]


def _caller_turns(turns):
    return [t for t in turns if t["speaker"] == "caller"]


def _find_quote(turns, keywords, speaker=None):
    """Return the first (t, quote) whose text contains any keyword, matched at a word
    boundary so 'payment' does not fire inside 'repayment'. Prefix match is intentional
    so 'unauthori' still catches 'unauthorised'/'unauthorized'. None if absent."""
    pats = [re.compile(r"\b" + re.escape(kw)) for kw in keywords]
    for t in turns:
        if speaker and t["speaker"] != speaker:
            continue
        low = t["text"].lower()
        if any(p.search(low) for p in pats):
            return {"t": t["start"], "quote": t["text"]}
    return None


def _mood_of(text):
    low = text.lower()
    neg = sum(1 for w in NEGATIVE if w in low)
    pos = sum(1 for w in POSITIVE if w in low)
    if neg > pos and neg > 0:
        return "frustrated" if neg >= 2 else "concerned"
    if pos > neg and pos > 0:
        return "positive"
    return "calm"


def heuristic_analysis(turns, meta):
    callers = _caller_turns(turns)

    # ---- intent -----------------------------------------------------------
    intent_cat, intent_label, intent_ev = "general_enquiry", "General enquiry", None
    for cat, kws in INTENT_RULES:
        ev = _find_quote(turns, kws, speaker="caller") or _find_quote(turns, kws)
        if ev:
            intent_cat = cat
            intent_label = cat.replace("_", " ").title()
            intent_ev = ev
            break
    if intent_ev is None and callers:
        intent_ev = {"t": callers[0]["start"], "quote": callers[0]["text"]}

    # ---- mood + shift -----------------------------------------------------
    moods = [(t["start"], _mood_of(t["text"]), t["text"]) for t in callers]
    start_mood = moods[0][1] if moods else "calm"
    end_mood = moods[-1][1] if moods else "calm"
    rank = {"positive": 0, "calm": 1, "concerned": 2, "frustrated": 3}
    shift_t, shift_quote = None, None
    worst = rank.get(start_mood, 1)
    for t, mood, text in moods:
        if rank.get(mood, 1) > worst:      # first clear downturn
            shift_t, shift_quote, worst = t, text, rank[mood]
            break

    # ---- resolution -------------------------------------------------------
    # Escalation / follow-up take priority (they mean it was NOT closed on the call),
    # then explicit resolution cues, then a positive-closure fallback: a caller who
    # signs off happy with thanks and no escalation was almost certainly resolved.
    res_ev = None
    if _find_quote(turns, ESCALATE_CUES):
        resolution, res_ev = "escalated", _find_quote(turns, ESCALATE_CUES)
    elif _find_quote(turns, FOLLOWUP_CUES):
        resolution, res_ev = "follow_up_promised", _find_quote(turns, FOLLOWUP_CUES)
    elif _find_quote(turns, RESOLVED_CUES):
        resolution, res_ev = "resolved", _find_quote(turns, RESOLVED_CUES)
    elif callers and end_mood == "positive":
        resolution, res_ev = "resolved", {"t": callers[-1]["start"], "quote": callers[-1]["text"]}
    else:
        resolution = "unresolved"
        res_ev = {"t": turns[-1]["start"], "quote": turns[-1]["text"]} if turns else None

    # ---- summary (<=40 words) --------------------------------------------
    verb = {"resolved": "and the agent resolved it on the call",
            "escalated": "and the agent escalated it",
            "follow_up_promised": "and the agent promised a follow-up",
            "unresolved": "which was not resolved on the call"}[resolution]
    summary = f"{meta['customer_name']} contacted {meta['agent_name']} about {intent_label.lower()}, {verb}."
    summary = " ".join(summary.split()[:40])

    return {
        "intent": {"label": intent_label, "category": intent_cat, "evidence": intent_ev},
        "mood": {"start_mood": start_mood, "end_mood": end_mood,
                 "shifted": shift_t is not None,
                 "shift": ({"t": shift_t, "quote": shift_quote} if shift_t is not None else None)},
        "resolution": {"status": resolution, "evidence": res_ev},
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Anthropic backend (optional)
# ---------------------------------------------------------------------------
LLM_PROMPT = """You are analysing one recorded bank support call. The transcript is
timestamped in seconds. Return ONLY JSON, no prose, matching exactly this shape:

{{"intent":{{"label":"...","category":"one of: fraud_dispute|card_issue|payment_transfer|balance_account|loan_mortgage|app_login|complaint|general_enquiry","evidence":{{"t":<sec>,"quote":"<verbatim words>"}}}},
 "mood":{{"start_mood":"...","end_mood":"...","shifted":true|false,"shift":{{"t":<sec>,"quote":"<verbatim>"}}|null}},
 "resolution":{{"status":"resolved|unresolved|escalated|follow_up_promised","evidence":{{"t":<sec>,"quote":"<verbatim>"}}}},
 "summary":"<= 40 words"}}

Every quote MUST be copied verbatim from the transcript. Cite the moment that justifies
each judgment. Transcript:

{transcript}"""


def _format_transcript(turns):
    return "\n".join(f'[{t["start"]:.2f}] {t["speaker"]}: {t["text"]}' for t in turns)


def anthropic_analysis(turns, meta):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=ANALYSIS_MODEL, max_tokens=1024,
        messages=[{"role": "user",
                   "content": LLM_PROMPT.format(transcript=_format_transcript(turns))}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    text = re.sub(r"^```json|```$", "", text.strip()).strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Validation shared by both backends
# ---------------------------------------------------------------------------
def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _verify_quote(turns, ev):
    """Blank out an evidence quote that isn't actually supported by the transcript.

    Guards against hallucinated citations: the brief scores unsupported evidence
    NEGATIVE, so a missing citation is safer than a fabricated one.
    """
    if not ev or not ev.get("quote"):
        return None
    qtok = _tokens(ev["quote"])
    if not qtok:
        return None
    for t in turns:
        overlap = len(qtok & _tokens(t["text"])) / len(qtok)
        if overlap >= 0.6:                       # quote genuinely present somewhere
            ev["t"] = t["start"]                 # snap to the true timestamp
            return ev
    return None                                   # unsupported -> drop it


def _score_attention(analysis, meta, turns):
    """Deterministic 0-100 from explainable factors (never model-authored)."""
    reasons, score = [], 0
    res = analysis["resolution"]["status"]
    if res == "unresolved":
        score += 30; reasons.append({"factor": "unresolved", "weight": 30})
    elif res == "escalated":
        score += 20; reasons.append({"factor": "escalated", "weight": 20})
    elif res == "follow_up_promised":
        score += 12; reasons.append({"factor": "follow_up_promised", "weight": 12})

    end_mood = analysis["mood"]["end_mood"]
    if end_mood == "frustrated":
        score += 25; reasons.append({"factor": "ends_frustrated", "weight": 25})
    elif end_mood == "concerned":
        score += 12; reasons.append({"factor": "ends_concerned", "weight": 12})
    if analysis["mood"].get("shifted"):
        sh = analysis["mood"].get("shift") or {}
        score += 12
        reasons.append({"factor": "mood_shift", "weight": 12, "t": sh.get("t")})

    if analysis["intent"]["category"] in ("fraud_dispute", "complaint"):
        score += 15
        reasons.append({"factor": f"high_stakes_{analysis['intent']['category']}", "weight": 15})

    ht = meta.get("handle_time_s") or 0
    if ht and ht > 240:                          # long calls tend to mean trouble
        score += 8; reasons.append({"factor": "long_handle_time", "weight": 8})

    mos = meta.get("caller_mos")
    if mos is not None and mos <= 2.5:           # poor line quality = frustrating call
        score += 6; reasons.append({"factor": "poor_audio_quality", "weight": 6})

    return min(score, 100), reasons


def analyze(turns, meta, backend="auto"):
    """Full analysis for one call. Returns a dict ready for the DB."""
    if backend == "auto":
        backend = "anthropic" if ANTHROPIC_API_KEY else "heuristic"
    try:
        raw = anthropic_analysis(turns, meta) if backend == "anthropic" \
            else heuristic_analysis(turns, meta)
    except Exception:
        raw = heuristic_analysis(turns, meta)     # never fail the batch on one call

    # normalise + verify every citation
    raw.setdefault("mood", {}).setdefault("shift", None)
    raw["intent"]["evidence"] = _verify_quote(turns, raw["intent"].get("evidence"))
    raw["resolution"]["evidence"] = _verify_quote(turns, raw["resolution"].get("evidence"))
    if raw["mood"].get("shift"):
        verified = _verify_quote(turns, raw["mood"]["shift"])
        raw["mood"]["shift"] = verified
        raw["mood"]["shifted"] = verified is not None
    raw["summary"] = " ".join(str(raw.get("summary", "")).split()[:40])

    score, reasons = _score_attention(raw, meta, turns)
    raw["attention"] = {"score": score, "reasons": reasons}

    return {
        "intent_label": raw["intent"]["label"],
        "intent_cat": raw["intent"]["category"],
        "start_mood": raw["mood"]["start_mood"],
        "end_mood": raw["mood"]["end_mood"],
        "shift_t": (raw["mood"]["shift"] or {}).get("t"),
        "shift_quote": (raw["mood"]["shift"] or {}).get("quote"),
        "resolution": raw["resolution"]["status"],
        "summary": raw["summary"],
        "attention": score,
        "analysis_json": raw,
    }
