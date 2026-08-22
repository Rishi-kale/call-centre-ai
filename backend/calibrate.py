"""Print attention-score thresholds recommended for whatever corpus is loaded.

The scoring formula deliberately uses fixed thresholds so a call's score is deterministic
and reproducible -- it must not drift as unrelated calls are ingested. The trade-off is
that the thresholds are corpus-relative and need recalibrating when the data changes
materially (e.g. a call centre whose calls run ten minutes, not one).

Run after ingesting a new corpus:

    python -m backend.calibrate

then put the suggested values in .env (CALLRADAR_LONG_CALL_S / CALLRADAR_POOR_MOS) and
re-score with the recompute step noted in the README.
"""
import sqlite3
from .config import DB_PATH, LONG_CALL_S, POOR_MOS


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100))]


def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT handle_time_s, caller_mos FROM calls").fetchall()
    con.close()
    if not rows:
        print("No calls in the database yet -- ingest first.")
        return

    ht = sorted(r["handle_time_s"] for r in rows if r["handle_time_s"])
    mos = sorted(r["caller_mos"] for r in rows if r["caller_mos"] is not None)
    n = len(rows)
    print(f"corpus: {n} calls\n")

    if ht:
        p90 = _pct(ht, 90)
        print("handle time")
        print(f"  min {ht[0]:.1f}s   median {_pct(ht,50):.1f}s   p90 {p90:.1f}s   max {ht[-1]:.1f}s")
        hits = sum(1 for x in ht if x > LONG_CALL_S)
        print(f"  current LONG_CALL_S={LONG_CALL_S:g}s flags {hits} calls ({100*hits/len(ht):.1f}%)")
        if hits == 0:
            print("  !! never fires -- the factor is dead weight on this corpus")
        elif hits / len(ht) > 0.5:
            print("  !! fires on most calls -- it is noise, not a signal")
        print(f"  suggested: CALLRADAR_LONG_CALL_S={round(p90)}   (p90 -> flags ~10%)\n")

    if mos:
        print("caller line quality (MOS)")
        print(f"  min {mos[0]}   median {_pct(mos,50)}   max {mos[-1]}")
        hits = sum(1 for x in mos if x <= POOR_MOS)
        print(f"  current POOR_MOS={POOR_MOS:g} flags {hits} calls ({100*hits/len(mos):.1f}%)")
        if hits == 0:
            print(f"  !! never fires -- nothing scores at or below {POOR_MOS:g}; "
                  f"lowest present is {mos[0]}")
        print(f"  suggested: CALLRADAR_POOR_MOS={mos[0]}   (the worst tier actually present)")


if __name__ == "__main__":
    main()
