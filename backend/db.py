"""SQLite storage. Analysis is computed once by the batch job; the API only reads."""
import sqlite3, json
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  sid            TEXT PRIMARY KEY,
  customer_name  TEXT,
  agent_name     TEXT,
  start_ms       INTEGER,
  duration_s     REAL,       -- length of the recording (from audio or metadata)
  handle_time_s  REAL,       -- call length from metadata timestamps
  caller_mos     REAL,       -- telephone audio quality 1-5 (a frustration signal)
  agent_mos      REAL,
  transcript     TEXT,       -- JSON array of turns {speaker,start,end,text,words}
  intent_label   TEXT,
  intent_cat     TEXT,       -- coarse category, used for trend grouping
  start_mood     TEXT,
  end_mood       TEXT,
  shift_t        REAL,       -- seconds into recording where mood shifted (NULL if none)
  shift_quote    TEXT,
  resolution     TEXT,       -- resolved | unresolved | escalated | follow_up_promised
  summary        TEXT,       -- <= 40 words
  attention      INTEGER,    -- 0-100
  analysis_json  TEXT,       -- full analysis incl. every evidence citation
  created_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_customer  ON calls(customer_name);
CREATE INDEX IF NOT EXISTS idx_agent     ON calls(agent_name);
CREATE INDEX IF NOT EXISTS idx_attention ON calls(attention DESC);
CREATE INDEX IF NOT EXISTS idx_cat       ON calls(intent_cat);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = connect()
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def upsert_call(row: dict):
    """row keys must match the columns above. transcript/analysis_json are dicts here."""
    row = dict(row)
    row["transcript"] = json.dumps(row.get("transcript", []))
    row["analysis_json"] = json.dumps(row.get("analysis_json", {}))
    cols = ",".join(row.keys())
    ph = ",".join("?" for _ in row)
    updates = ",".join(f"{k}=excluded.{k}" for k in row if k != "sid")
    con = connect()
    con.execute(
        f"INSERT INTO calls ({cols}) VALUES ({ph}) "
        f"ON CONFLICT(sid) DO UPDATE SET {updates}",
        list(row.values()),
    )
    con.commit()
    con.close()


def _decode(r: sqlite3.Row) -> dict:
    d = dict(r)
    if "transcript" in d and d["transcript"]:
        d["transcript"] = json.loads(d["transcript"])
    if "analysis_json" in d and d["analysis_json"]:
        d["analysis_json"] = json.loads(d["analysis_json"])
    return d


def get_call(sid: str):
    con = connect()
    r = con.execute("SELECT * FROM calls WHERE sid=?", (sid,)).fetchone()
    con.close()
    return _decode(r) if r else None


def list_customers():
    con = connect()
    rows = con.execute(
        "SELECT customer_name, COUNT(*) n, MAX(start_ms) last_ms, "
        "       MAX(attention) max_attention "
        "FROM calls GROUP BY customer_name ORDER BY max_attention DESC, n DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def customer_calls(name: str):
    con = connect()
    rows = con.execute(
        "SELECT sid, agent_name, start_ms, duration_s, intent_label, "
        "       resolution, attention, summary, start_mood, end_mood "
        "FROM calls WHERE customer_name=? ORDER BY start_ms DESC",
        (name,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def attention_ranked(limit=50):
    con = connect()
    rows = con.execute(
        "SELECT sid, customer_name, agent_name, intent_label, resolution, "
        "       attention, summary, shift_t, start_mood, end_mood, start_ms "
        "FROM calls ORDER BY attention DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def trends():
    con = connect()
    rows = con.execute(
        "SELECT intent_cat, COUNT(*) n, "
        "       SUM(CASE WHEN resolution='resolved' THEN 1 ELSE 0 END) resolved, "
        "       AVG(attention) avg_attention "
        "FROM calls GROUP BY intent_cat ORDER BY n DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def agent_stats():
    con = connect()
    rows = con.execute(
        "SELECT agent_name, COUNT(*) calls, "
        "       AVG(handle_time_s) avg_handle_s, "
        "       AVG(attention) avg_attention, "
        "       SUM(CASE WHEN resolution='resolved' THEN 1 ELSE 0 END) resolved "
        "FROM calls GROUP BY agent_name ORDER BY calls DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
