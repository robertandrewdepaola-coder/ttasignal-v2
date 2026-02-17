#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def p95(values):
    if not values:
        return None
    values = sorted(values)
    idx = int(0.95 * (len(values) - 1))
    return values[idx]


def load_rows(path: Path):
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
            continue
    return rows


def main():
    ap = argparse.ArgumentParser(description="Check runtime SLOs from perf metrics.")
    ap.add_argument("--file", default="v2_perf_metrics.jsonl")
    ap.add_argument("--strict", action="store_true", help="Fail if required samples are missing.")
    args = ap.parse_args()

    rows = load_rows(Path(args.file))
    dashboard = [float(r.get("sec", 0)) for r in rows if r.get("kind") == "dashboard_render"]
    full_scan = [float(r.get("total_sec", 0)) for r in rows if r.get("kind") == "scan_run" and r.get("mode") == "all"]
    fast_refresh = [float(r.get("sec", 0)) for r in rows if r.get("kind") == "fast_refresh"]

    slo = {
        "dashboard_render_p95_lt": 5.0,
        "full_refresh_p95_lt": 90.0,
        "fast_refresh_p95_lt": 20.0,
    }
    stats = {
        "dashboard_render_p95": p95(dashboard),
        "full_refresh_p95": p95(full_scan),
        "fast_refresh_p95": p95(fast_refresh),
        "samples": {
            "dashboard_render": len(dashboard),
            "full_refresh": len(full_scan),
            "fast_refresh": len(fast_refresh),
        },
    }

    print("[slo] samples:", stats["samples"])
    print("[slo] p95:", {k: v for k, v in stats.items() if k.endswith("p95")})

    failures = []
    if stats["dashboard_render_p95"] is None:
        if args.strict:
            failures.append("Missing dashboard render samples")
    elif stats["dashboard_render_p95"] >= slo["dashboard_render_p95_lt"]:
        failures.append(f"dashboard render p95 {stats['dashboard_render_p95']:.2f}s >= {slo['dashboard_render_p95_lt']:.2f}s")

    if stats["full_refresh_p95"] is None:
        if args.strict:
            failures.append("Missing full refresh samples")
    elif stats["full_refresh_p95"] >= slo["full_refresh_p95_lt"]:
        failures.append(f"full refresh p95 {stats['full_refresh_p95']:.2f}s >= {slo['full_refresh_p95_lt']:.2f}s")

    if stats["fast_refresh_p95"] is None:
        if args.strict:
            failures.append("Missing fast refresh samples")
    elif stats["fast_refresh_p95"] >= slo["fast_refresh_p95_lt"]:
        failures.append(f"fast refresh p95 {stats['fast_refresh_p95']:.2f}s >= {slo['fast_refresh_p95_lt']:.2f}s")

    if failures:
        print("[slo] FAIL")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)

    print("[slo] PASS")


if __name__ == "__main__":
    main()

