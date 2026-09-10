#!/usr/bin/env python3
"""Apply a derived frozen package to every segment in a manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def apply(
    row: dict[str, str], config: dict, overwrite: bool, max_events: int
) -> dict[str, object]:
    run, segment = int(row["run"]), int(row["segment"])
    source = Path(row["hao_production_root"]).resolve()
    member_dir = path(config["paths"]["member_dir"])
    sidecar = member_dir / f"nps_production_{run}_{segment}_wf_calib.production_members.tsv"
    output = path(config["paths"]["corrected_root_dir"]) / source.name
    package = path(
        config["paths"].get(
            "package_dir", str(path(config["paths"]["work_dir"]) / "frozen_package")
        )
    )
    command = [
        sys.executable, str(HERE.parent / "root" / "apply_root.py"),
        "--input-root", str(source), "--output-root", str(output),
        "--package", str(package), "--run", str(run), "--segment", str(segment),
        "--sidecar", str(sidecar), "--missing-seed", "error",
        "--max-events", str(max_events),
    ]
    if overwrite:
        command.append("--overwrite")
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    ok = result.returncode == 0 and output.is_file() and output.stat().st_size > 0
    return {
        "run": run, "segment": segment, "status": "ok" if ok else "failed", "output": output,
        "detail": (result.stderr or result.stdout)[-1000:] if not ok else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--run", type=int, help="Process only this run (batch use).")
    parser.add_argument("--segment", type=int, help="Process only this segment; requires --run.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    with path(config["paths"]["segment_manifest"]).open(newline="", encoding="ascii") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("status", "ready") == "ready"]
    if args.segment is not None and args.run is None:
        parser.error("--segment requires --run")
    rows = [
        row for row in rows
        if (args.run is None or int(row["run"]) == args.run)
        and (args.segment is None or int(row["segment"]) == args.segment)
    ]
    if not rows:
        parser.error("no matching ready segment rows")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(apply, row, config, args.overwrite, args.max_events)
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"apply {result['run']}/{result['segment']} {result['status']}", flush=True)
    report_name = (
        f"application_report_run{args.run}_seg{args.segment}.tsv"
        if args.run is not None and args.segment is not None
        else "application_report.tsv"
    )
    report = path(config["paths"]["work_dir"]) / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "segment", "status", "output", "detail"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: (row["run"], row["segment"])))
    failures = [row for row in results if row["status"] != "ok"]
    print(f"segments={len(results)} failures={len(failures)} report={report}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
