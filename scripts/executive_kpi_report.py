#!/usr/bin/env python3
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_json_lines(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def load_audit(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def day(ts: str):
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def main():
    perf_rows = load_json_lines(Path("v2_perf_metrics.jsonl"))
    audit_rows = load_audit(Path("v2_watchlist_audit.json"))

    by_day = defaultdict(lambda: {"fast_refresh": 0, "scan_all": 0, "scan_new": 0, "alerts": 0, "trades": 0, "scan_sec": []})
    for r in perf_rows:
        d = day(r.get("ts", ""))
        kind = r.get("kind")
        if kind == "fast_refresh":
            by_day[d]["fast_refresh"] += 1
        elif kind == "scan_run":
            mode = r.get("mode")
            if mode == "all":
                by_day[d]["scan_all"] += 1
            elif mode == "new_only":
                by_day[d]["scan_new"] += 1
            by_day[d]["scan_sec"].append(float(r.get("total_sec", 0.0)))
        elif kind == "dashboard_render":
            pass

    for e in audit_rows:
        d = str(e.get("ts", ""))[:10]
        act = e.get("action")
        if act == "ALERT_CHECK":
            by_day[d]["alerts"] += 1
        if act == "ENTER_TRADE":
            by_day[d]["trades"] += 1

    print("Daily Executive KPI Report")
    print("=" * 30)
    for d in sorted(by_day.keys()):
        row = by_day[d]
        avg_scan = sum(row["scan_sec"]) / len(row["scan_sec"]) if row["scan_sec"] else 0.0
        print(
            f"{d} | fast_refresh={row['fast_refresh']} scan_all={row['scan_all']} "
            f"scan_new={row['scan_new']} alerts_checked={row['alerts']} trades={row['trades']} "
            f"avg_scan_sec={avg_scan:.1f}"
        )


if __name__ == "__main__":
    main()

