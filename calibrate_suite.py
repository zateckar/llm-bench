"""Analyze saved quality runs without modifying the application database."""

import argparse
import json
from pathlib import Path

from app.benchmarking.calibration import analyze_runs, markdown, read_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path(__file__).parent / "data" / "bench.db")
    parser.add_argument("--suite-hash", help="Select one recorded suite; protocols remain separate")
    parser.add_argument("--runs", type=int, nargs="+", help="Explicit recorded run IDs")
    parser.add_argument("--output", type=Path, required=True, help="JSON output; also writes adjacent Markdown")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("Calibration output must have a .json extension")
    if args.database.resolve() in {args.output.resolve(), args.output.with_suffix(".md").resolve()}:
        parser.error("Calibration output must not overwrite its source database")
    runs = read_runs(args.database)
    if args.runs and set(args.runs) - {run["id"] for run in runs}:
        parser.error("Requested run IDs were not found")
    report = analyze_runs(runs, suite_hash=args.suite_hash, run_ids=set(args.runs) if args.runs else None)
    if not report["cohorts"]:
        parser.error("No complete, valid quality reports match the selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"runs": report["included_run_ids"], "cohorts": len(report["cohorts"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
