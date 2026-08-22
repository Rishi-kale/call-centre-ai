"""Turn a transcript into cited judgments.

Every judgment carries evidence {t, quote} where the quote is verbatim from the
transcript. Two backends:

  * heuristic  - deterministic, no dependencies, always runs. Good demo baseline.
  * anthropic  - richer judgments via the API, if ANTHROPIC_API_KEY is set.

Both pass through the SAME validation: quotes are verified against the transcript,
the summary is capped at 40 words, and the attention score is computed by a fixed
formula (never by the model) so it is explainable and defensible.
"""
import json, re, time
from .config import (GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL,
                    ANTHROPIC_API_KEY, ANALYSIS_MODEL, LONG_CALL_S, POOR_MOS)

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
                 "you're all good", "problem solved", "refunded", "reversed the charge",
                 "ordered your replacement", "ordered a new", "sent you a new",
                 "should receive it", "on its way", "already taken care of",
                 "have ordered", "have issued", "issued a new", "unblocked your",
                 "reset your", "updated your", "processed the", "cancelled the"]
FOLLOWUP_CUES = ["call you back", "callback", "within 48 hours", "within 24 hours",
                 "someone will call", "we'll be in touch", "log a case", "raise a ticket"]
ESCALATE_CUES = ["escalate", "put you through", "transfer you", "my manager",
                 "complaints team", "senior"]
CLARIFICATION_CUES = ["let me repeat", "i'll repeat", "just to repeat", "say that again",
                      "one more time", "could you repeat", "can you repeat",
                      "just to confirm", "just to clarify", "to clarify",
                      "sorry, what was", "sorry, could you", "sorry, can you",
                      "didn't catch that", "did not catch that", "come again",
                      "could you say that", "can you say that", "what was that"]


def clarification_count(turns):
    """Count agent turns where the agent has to ask the caller to repeat/clarify
    something. A rough proxy for 'had to ask three times' -- real signal, computed
    from the actual transcript, not modelled."""
    pats = [re.compile(r"\b" + re.escape(c)) for c in CLARIFICATION_CUES]
    n = 0
    for t in turns:
        if t["speaker"] != "agent":
            continue
        low = t["text"].lower()
        if any(p.search(low) for p in pats):
            n += 1
    return n


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
 "mood":{{"start_mood":"one of: positive|calm|concerned|frustrated","end_mood":"one of: positive|calm|concerned|frustrated","shifted":true|false,"shift":{{"t":<sec>,"quote":"<verbatim>"}}|null}},
 "resolution":{{"status":"resolved|unresolved|escalated|follow_up_promised","evidence":{{"t":<sec>,"quote":"<verbatim>"}}}},
 "summary":"<= 40 words"}}

Resolution definitions -- judge by what actually happened, not by whether an explicit
confirmation phrase like "resolved" or "sorted" was spoken:
 - resolved: the agent directly addressed the caller's request/question on THIS call and
   the caller ends the call satisfied or neutral (a plain "thanks, that's all, bye" close
   after the agent answered them counts as resolved -- do not require an explicit
   confirmation phrase).
 - unresolved: the caller's issue is left unaddressed, still broken, or the caller ends
   frustrated/unsatisfied.
 - escalated: the agent transfers the caller or brings in a manager/senior/complaints team.
 - follow_up_promised: the agent promises a callback or a future action instead of
   resolving it on this call.

Mood-shift definitions -- "shifted" must be a genuine change in the CALLER's emotional
tone from a negative/neutral state toward frustration or away from it. A routine, matched
pleasantry at the end of a call (e.g. the agent says "have a great day" and the caller
replies "you too, bye") is NOT a mood shift -- only mark shifted=true when something in
the call actually changes how the caller feels.

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


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def gemini_list_models(api_key=None):
    """Model names change between releases -- ask the API what this key can actually use
    rather than assuming. (A wrong model name is a 404 that would otherwise show up as a
    silent heuristic fallback across a whole batch.)"""
    import requests
    key = api_key or GEMINI_API_KEY
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     params={"key": key}, timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("models", []):
        if "generateContent" in (m.get("supportedGenerationMethods") or []):
            out.append(m["name"].removeprefix("models/"))
    return out


def gemini_analysis(turns, meta, model=None):
    """Free-tier Google Gemini backend (aistudio.google.com). Same prompt/JSON contract as
    the other LLM backends; responseMimeType pins the reply to JSON."""
    import requests
    body = {
        "contents": [{"parts": [{"text": LLM_PROMPT.format(transcript=_format_transcript(turns))}]}],
        # Gemini 3.x spends part of the output budget on internal reasoning, so a 1k cap
        # truncates the JSON mid-string. Give it room.
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 4096},
    }
    url = GEMINI_URL.format(model=model or GEMINI_MODEL)
    # The free tier limits requests per MINUTE (~15 RPM), so a 429 usually just means
    # "slow down", not "quota gone". Wait out the window rather than giving up.
    for attempt in range(4):
        r = requests.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=60)
        if r.status_code == 429 and attempt < 3:
            wait = float(r.headers.get("Retry-After") or 0) or (12 * (attempt + 1))
            time.sleep(min(wait, 45))
            continue
        if r.status_code == 503 and attempt < 3:      # transient overload
            time.sleep(4 * (attempt + 1))
            continue
        break
    r.raise_for_status()
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts).strip()
    text = re.sub(r"^```json|```$", "", text).strip()
    return json.loads(text)


def groq_analysis(turns, meta, model=None):
    """Free-tier LLM backend (console.groq.com), same prompt/JSON contract as Anthropic."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    msg = client.chat.completions.create(
        model=model or GROQ_MODEL, max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[{"role": "user",
                   "content": LLM_PROMPT.format(transcript=_format_transcript(turns))}],
    )
    text = msg.choices[0].message.content
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


# Thresholds for the two "signal" factors below live in config (env-overridable).
#
# They were originally generic call-centre numbers (>240s handle time, MOS <= 2.5) and
# BOTH were unreachable on this corpus: the longest call is 181s and the worst line
# quality is 3.0, so neither factor ever fired across all 1,441 calls -- 14 points of the
# 0-100 scale were dead while the README advertised them as live signals.
#
# Calibrated defaults:
#   LONG_CALL_S = 85s  -> p90 of handle time, flags the longest ~10% (148 calls)
#   POOR_MOS    = 3.0  -> the lowest quality tier that actually occurs (265 calls, 18%)
#
# POOR_MOS generalises (it is a lower bound -- genuinely bad lines still trip it), but
# LONG_CALL_S is corpus-relative: on a corpus of 10-minute calls, 85s would flag almost
# everything. Recalibrate it via CALLRADAR_LONG_CALL_S / `python -m backend.calibrate`.


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
    if ht and ht > LONG_CALL_S:                  # long calls tend to mean trouble
        score += 8; reasons.append({"factor": "long_handle_time", "weight": 8})

    mos = meta.get("caller_mos")
    if mos is not None and mos <= POOR_MOS:      # poor line quality = frustrating call
        score += 6; reasons.append({"factor": "poor_audio_quality", "weight": 6})

    return min(score, 100), reasons


def _validate_shape(raw):
    """The LLM backends occasionally return technically-valid JSON that's still missing
    a required key (e.g. no "resolution" object at all). That's not an API exception, so
    it slips past the retry loop unless we check for it explicitly -- raise here so a
    malformed response gets retried/falls back exactly like any other backend failure."""
    if not isinstance(raw, dict):
        raise ValueError("analysis response was not a JSON object")
    for key in ("intent", "mood", "resolution"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"analysis response missing '{key}' object")
    if "label" not in raw["intent"] or "category" not in raw["intent"]:
        raise ValueError("intent missing label/category")
    valid_moods = {"positive", "calm", "concerned", "frustrated"}
    if raw["mood"].get("start_mood") not in valid_moods or raw["mood"].get("end_mood") not in valid_moods:
        raise ValueError(f"mood not in {valid_moods}: {raw['mood']}")
    if "status" not in raw["resolution"]:
        raise ValueError("resolution missing status")
    if not isinstance(raw.get("summary"), str):
        raise ValueError("summary missing or not a string")


def analyze(turns, meta, backend="auto", model=None):
    """Full analysis for one call. Returns a dict ready for the DB."""
    if backend == "auto":
        backend = ("gemini" if GEMINI_API_KEY else "groq" if GROQ_API_KEY
                   else "anthropic" if ANTHROPIC_API_KEY else "heuristic")

    raw = None
    used_backend = "heuristic"
    if backend in ("gemini", "groq", "anthropic"):
        if backend == "gemini":
            fn = lambda t, m: gemini_analysis(t, m, model=model)
        elif backend == "groq":
            fn = lambda t, m: groq_analysis(t, m, model=model)
        else:
            fn = anthropic_analysis
        last_err = None
        for attempt in range(1, 4):            # a few retries survives free-tier rate limits
            try:
                raw = fn(turns, meta)
                _validate_shape(raw)
                used_backend = backend
                break
            except Exception as e:
                raw = None
                last_err = e
                if attempt < 3:
                    time.sleep(min(20, 2 ** attempt))
        if raw is None:
            # Never fail the batch on one call -- but a silent fallback here is exactly how a
            # bad model name/API key can make an entire batch quietly run on the heuristic
            # while everyone believes it used the LLM. Always surface it.
            print(f"  [analyze] {backend} backend failed after retries ({last_err!r}); falling back to heuristic")
    if raw is None:
        raw = heuristic_analysis(turns, meta)
    raw["backend_used"] = used_backend        # lets a re-run skip calls already done by an LLM
    if used_backend == "gemini":
        raw["model_used"] = model or GEMINI_MODEL
    elif used_backend == "groq":
        raw["model_used"] = model or GROQ_MODEL

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
    clarifications = clarification_count(turns)
    raw["clarification_count"] = clarifications

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
        "clarification_count": clarifications,
        "analysis_json": raw,
    }
