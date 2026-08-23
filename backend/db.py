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
  clarification_count INTEGER, -- # of agent turns asking the caller to repeat/clarify
  analysis_json  TEXT,       -- full analysis incl. every evidence citation
  created_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_customer  ON calls(customer_name);
CREATE INDEX IF NOT EXISTS idx_agent     ON calls(agent_name);
CREATE INDEX IF NOT EXISTS idx_attention ON calls(attention DESC);
CREATE INDEX IF NOT EXISTS idx_cat       ON calls(intent_cat);
"""


def _migrate(con):
    """Add columns introduced after the initial schema, for DBs created earlier."""
    cols = {r["name"] for r in con.execute("PRAGMA table_info(calls)").fetchall()}
    if "clarification_count" not in cols:
        con.execute("ALTER TABLE calls ADD COLUMN clarification_count INTEGER")


DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 200

# Sortable columns, keyed by the name clients use.
#
# An ORDER BY column cannot be a bound parameter, so it has to be concatenated into the
# SQL -- which makes an allowlist the only safe way to accept a sort field from a client.
# Anything not in these maps is rejected by the API rather than interpolated.
CUSTOMER_SORTS = {
    "name": "customer_name",
    "calls": "n",
    "last_contact": "last_ms",
    "attention": "max_attention",
}
AGENT_SORTS = {
    "name": "agent_name",
    "calls": "calls",
    "score": "avg_attention",
}
# Mood is ranked by severity, not alphabetically -- sorting "desc" has to surface
# frustrated callers, and A-Z would order them calm/concerned/frustrated/positive.
_MOOD_RANK = ("CASE end_mood WHEN 'frustrated' THEN 3 WHEN 'concerned' THEN 2 "
              "WHEN 'calm' THEN 1 WHEN 'positive' THEN 0 ELSE 0 END")
ATTENTION_SORTS = {
    "score": "attention",
    "customer": "customer_name",
    "agent": "agent_name",
    "mood": _MOOD_RANK,
    "resolution": "resolution",
}


def _order_by(sort_map, sort, order, default_sql, tiebreak):
    """Build a total ORDER BY: the requested column, then a unique tiebreaker.

    The tiebreaker is not optional -- without it, ties in the sort column leave row order
    undefined under LIMIT/OFFSET and rows duplicate or vanish between pages.
    """
    if not sort:
        return f"{default_sql}, {tiebreak} ASC"
    col = sort_map[sort]                          # KeyError => caller validated wrongly
    direction = "DESC" if (order or "asc").lower() == "desc" else "ASC"
    if col == tiebreak:
        return f"{col} {direction}"
    return f"{col} {direction}, {tiebreak} ASC"


def _page_envelope(rows, total, page, size):
    """Standard paged response. Mirrors the Spring Data `Page` shape: 0-indexed `page`,
    `totalElements`, and the first/last flags clients use to disable their arrows."""
    total_pages = (total + size - 1) // size if size else 0
    return {
        "content": rows,
        "page": page,
        "size": size,
        "totalElements": total,
        "totalPages": total_pages,
        "numberOfElements": len(rows),
        "first": page == 0,
        "last": page >= total_pages - 1 or total_pages == 0,
        "empty": not rows,
    }


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = connect()
    con.executescript(SCHEMA)
    _migrate(con)
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


def list_customers(page=0, size=DEFAULT_PAGE_SIZE, q=None, sort=None, order=None):
    """One page of customers. Defaults to worst-attention first; sortable by name,
    call count and last contact (see CUSTOMER_SORTS)."""
    where, params = "", []
    if q:
        where = "WHERE customer_name LIKE ? "
        params.append(f"%{q}%")
    order_by = _order_by(CUSTOMER_SORTS, sort, order,
                         "max_attention DESC, n DESC", "customer_name")
    con = connect()
    total = con.execute(
        f"SELECT COUNT(DISTINCT customer_name) FROM calls {where}", params
    ).fetchone()[0]
    rows = con.execute(
        f"SELECT customer_name, COUNT(*) n, MAX(start_ms) last_ms, "
        f"       MAX(attention) max_attention "
        f"FROM calls {where}"
        f"GROUP BY customer_name "
        f"ORDER BY {order_by} "
        f"LIMIT ? OFFSET ?",
        (*params, size, page * size),
    ).fetchall()
    con.close()
    return _page_envelope([dict(r) for r in rows], total, page, size)


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


_CTX_COLS = ("sid, customer_name, agent_name, start_ms, duration_s, intent_label, "
             "intent_cat, resolution, attention, summary, start_mood, end_mood")


def call_context(sid: str, limit: int = 12):
    """Sibling calls that give a manager somewhere to go from one call.

    Two lists, plus how the issue tends to end:
      customer_calls - this customer's other calls, newest first (is this a repeat caller?)
      same_intent    - OTHER customers with the same intent, worst first
      intent_stats   - resolution split for that intent, so a fix that worked on
                       one call can be recognised and reused on the rest
    """
    con = connect()
    row = con.execute(
        "SELECT customer_name, intent_cat, intent_label FROM calls WHERE sid=?", (sid,)
    ).fetchone()
    if not row:
        con.close()
        return None
    customer, cat, label = row["customer_name"], row["intent_cat"], row["intent_label"]

    customer_calls = con.execute(
        f"SELECT {_CTX_COLS} FROM calls WHERE customer_name=? AND sid<>? "
        f"ORDER BY start_ms DESC LIMIT ?",
        (customer, sid, limit),
    ).fetchall()

    same_intent = con.execute(
        f"SELECT {_CTX_COLS} FROM calls WHERE intent_cat=? AND customer_name<>? "
        f"ORDER BY attention DESC, start_ms DESC LIMIT ?",
        (cat, customer, limit),
    ).fetchall()

    stats = con.execute(
        "SELECT COUNT(*) n, "
        "       SUM(CASE WHEN resolution='resolved' THEN 1 ELSE 0 END) resolved, "
        "       COUNT(DISTINCT customer_name) customers, "
        "       AVG(attention) avg_attention "
        "FROM calls WHERE intent_cat=?",
        (cat,),
    ).fetchone()

    # a repeat caller on the SAME issue is the "sounded resolved but wasn't" signal
    repeats = con.execute(
        "SELECT COUNT(*) FROM calls WHERE customer_name=? AND intent_cat=? AND sid<>?",
        (customer, cat, sid),
    ).fetchone()[0]

    con.close()
    return {
        "intent_cat": cat,
        "intent_label": label,
        "customer_name": customer,
        "customer_calls": [dict(r) for r in customer_calls],
        "same_intent": [dict(r) for r in same_intent],
        "intent_stats": dict(stats) if stats else {},
        "same_issue_repeats": repeats,
    }


def filter_facets():
    """Filter options with counts -- drives the checkbox lists on Needs Attention.

    Derived from the data rather than hardcoded, so a category that stops (or starts)
    appearing after an ingest shows up in the UI without a code change. The counts also
    tell a manager where to look before they click anything.
    """
    con = connect()
    intents = con.execute(
        "SELECT intent_cat AS value, COUNT(*) n, "
        "       SUM(CASE WHEN resolution<>'resolved' THEN 1 ELSE 0 END) unresolved "
        "FROM calls WHERE intent_cat IS NOT NULL "
        "GROUP BY intent_cat ORDER BY n DESC"
    ).fetchall()
    resolutions = con.execute(
        "SELECT resolution AS value, COUNT(*) n "
        "FROM calls WHERE resolution IS NOT NULL "
        "GROUP BY resolution ORDER BY n DESC"
    ).fetchall()
    con.close()
    return {
        "intents": [dict(r) for r in intents],
        "resolutions": [dict(r) for r in resolutions],
    }


def attention_ranked(page=0, size=DEFAULT_PAGE_SIZE, q=None, intents=None, resolutions=None,
                     sort=None, order=None):
    """One page of calls ranked by attention score, worst first.

    Trailing `sid` keeps the order total -- 859 calls share a score of 0 here, so without
    it LIMIT/OFFSET would duplicate and drop rows across pages.

    `intents` / `resolutions` are optional value lists. Within a filter the values are
    OR'd (pick several intents), across filters they are AND'd (that intent AND
    unresolved), which is what a manager expects from checkbox groups.
    """
    clauses, params = [], []
    if q:
        # intent_cat matters here: it is the canonical grouping, and the free-text
        # intent_label often omits the word a user would search for ("fraud").
        clauses.append("(customer_name LIKE ? OR agent_name LIKE ? OR summary LIKE ? "
                       "OR intent_label LIKE ? OR intent_cat LIKE ?)")
        params += [f"%{q}%"] * 5
    # placeholders, not interpolation -- an IN list can be parameterised safely
    if intents:
        clauses.append(f"intent_cat IN ({','.join('?' for _ in intents)})")
        params += list(intents)
    if resolutions:
        clauses.append(f"resolution IN ({','.join('?' for _ in resolutions)})")
        params += list(resolutions)
    where = ("WHERE " + " AND ".join(clauses) + " ") if clauses else ""
    order_by = _order_by(ATTENTION_SORTS, sort, order,
                         "attention DESC, start_ms DESC", "sid")
    con = connect()
    total = con.execute(f"SELECT COUNT(*) FROM calls {where}", params).fetchone()[0]
    rows = con.execute(
        f"SELECT sid, customer_name, agent_name, intent_label, intent_cat, resolution, "
        f"       attention, summary, shift_t, shift_quote, start_mood, end_mood, start_ms "
        f"FROM calls {where}"
        f"ORDER BY {order_by} "
        f"LIMIT ? OFFSET ?",
        (*params, size, page * size),
    ).fetchall()
    con.close()
    return _page_envelope([dict(r) for r in rows], total, page, size)


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


def overview_stats():
    """Aggregate counts for the stat-card rows on the Customers / Agents views.
    'Active calls today' stands in against the dataset's own latest day, since this
    is a historical recording set rather than a live feed."""
    con = connect()
    total_calls = con.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    total_customers = con.execute("SELECT COUNT(DISTINCT customer_name) FROM calls").fetchone()[0]
    total_agents = con.execute("SELECT COUNT(DISTINCT agent_name) FROM calls").fetchone()[0]
    critical_customers = con.execute(
        "SELECT COUNT(*) FROM (SELECT customer_name, MAX(attention) m FROM calls "
        "GROUP BY customer_name HAVING m>=75)"
    ).fetchone()[0]
    high_risk_agents = con.execute(
        "SELECT COUNT(*) FROM (SELECT agent_name, AVG(attention) a FROM calls "
        "GROUP BY agent_name HAVING a>=50)"
    ).fetchone()[0]
    avg_handle_s = con.execute("SELECT AVG(handle_time_s) FROM calls").fetchone()[0]
    latest = con.execute("SELECT MAX(start_ms) m FROM calls").fetchone()
    latest_ms = latest["m"] if latest else None
    calls_latest_day = 0
    if latest_ms:
        latest_day = con.execute(
            "SELECT date(start_ms/1000, 'unixepoch') d FROM calls ORDER BY start_ms DESC LIMIT 1"
        ).fetchone()["d"]
        calls_latest_day = con.execute(
            "SELECT COUNT(*) FROM calls WHERE date(start_ms/1000,'unixepoch')=?",
            (latest_day,),
        ).fetchone()[0]
    con.close()
    return {
        "total_calls": total_calls,
        "total_customers": total_customers,
        "total_agents": total_agents,
        "critical_customers": critical_customers,
        "high_risk_agents": high_risk_agents,
        "avg_handle_s": avg_handle_s,
        "calls_latest_day": calls_latest_day,
        "latest_ms": latest_ms,
    }


def attention_timeline(days=7):
    """Avg attention per day, for the last N days present in the dataset (not
    wall-clock 'today' -- this is a historical recording set)."""
    con = connect()
    rows = con.execute(
        "SELECT date(start_ms/1000,'unixepoch') d, AVG(attention) avg_attention, COUNT(*) n "
        "FROM calls GROUP BY d ORDER BY d DESC LIMIT ?",
        (days,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows][::-1]


def agent_stats(page=0, size=DEFAULT_PAGE_SIZE, q=None, sort=None, order=None):
    """One page of per-agent stats. Defaults to busiest first; sortable by name, call
    volume and average attention score (see AGENT_SORTS).

    The clarification signal compares an agent against the WHOLE cohort, not just the
    current page -- a page-local average would rank an agent differently depending on
    which page they happened to land on.
    """
    where, params = "", []
    if q:
        where = "WHERE agent_name LIKE ? "
        params.append(f"%{q}%")
    order_by = _order_by(AGENT_SORTS, sort, order, "calls DESC", "agent_name")
    con = connect()
    total = con.execute(
        f"SELECT COUNT(DISTINCT agent_name) FROM calls {where}", params
    ).fetchone()[0]
    cohort_avg = con.execute(
        "SELECT AVG(clarification_count) FROM calls"
    ).fetchone()[0] or 0
    rows = con.execute(
        f"SELECT agent_name, COUNT(*) calls, "
        f"       AVG(handle_time_s) avg_handle_s, "
        f"       AVG(attention) avg_attention, "
        f"       AVG(clarification_count) avg_clarification, "
        f"       SUM(CASE WHEN resolution='resolved' THEN 1 ELSE 0 END) resolved "
        f"FROM calls {where}"
        f"GROUP BY agent_name "
        f"ORDER BY {order_by} "
        f"LIMIT ? OFFSET ?",
        (*params, size, page * size),
    ).fetchall()
    con.close()
    out = [dict(r) for r in rows]
    for r in out:
        r["clarification_signal"] = (
            "High" if cohort_avg > 0 and (r["avg_clarification"] or 0) > cohort_avg * 1.25
            else "Low"
        )
    return _page_envelope(out, total, page, size)
