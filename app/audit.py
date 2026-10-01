"""Append-only audit trail (JSON Lines). One line per event: who, what, when, with which model."""
import json
from datetime import datetime, timezone

from . import config


def log_event(run_id, alert_id, actor, event, details=None, path=None):
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "alert_id": alert_id,
        "actor": actor,
        "event": event,
        "details": details or {},
    }
    with open(path or config.AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_log(path=None):
    p = path or config.AUDIT_LOG
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise_runs(log):
    """One row per case run, from the audit log: final level, route, and whether it is closed."""
    runs = {}
    for e in log:
        rid = e.get("run_id")
        if not rid or rid == "-":
            continue
        r = runs.setdefault(rid, {"run_id": rid, "alert_id": e["alert_id"], "final_level": None,
                                  "route": None, "closed": False, "outcome": None})
        d = e.get("details") or {}
        if e["event"] == "case_routed":
            r["final_level"], r["route"] = d.get("final_level"), d.get("route")
        elif e["event"] in ("data_exception", "agent_failed"):
            r["route"] = "exception_review"
        elif e["event"] == "case_closed":
            r["closed"], r["outcome"] = True, d.get("outcome")
    return list(runs.values())
