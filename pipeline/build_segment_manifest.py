#!/usr/bin/env python3
"""Match Hao-calibrated production ROOTs to auxiliary raw-WF ROOTs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


HAO_RE = re.compile(r"nps_production_(\d+)_(\d+)_wf_calib\.root$")
RAW_RE = re.compile(r"nps_production_(\d+)_(\d+)_wf\.root$")


def read_runs(path: Path) -> set[int]:
    runs: set[int] = set()
    for line in path.read_text(encoding="ascii").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and not value.lower().startswith("run"):
            runs.add(int(value.split()[0]))
    if not runs:
        raise ValueError(f"empty run manifest: {path}")
    return runs


def discover(base: Path, pattern: re.Pattern[str], runs: set[int]) -> dict[tuple[int, int], Path]:
    found: dict[tuple[int, int], Path] = {}
    for path in base.rglob("*.root"):
        match = pattern.fullmatch(path.name)
        if not match or int(match.group(1)) not in runs:
            continue
        key = int(match.group(1)), int(match.group(2))
        if key in found:
            raise ValueError(f"duplicate run/segment {key}: {found[key]} and {path}")
        found[key] = path.resolve()
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--hao-dir", required=True, type=Path)
    parser.add_argument("--raw-wf-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    runs = read_runs(args.runs)
    hao = discover(args.hao_dir, HAO_RE, runs)
    raw = discover(args.raw_wf_dir, RAW_RE, runs)
    keys = sorted(set(hao) | set(raw))
    rows = []
    missing = 0
    for run, segment in keys:
        status = "ready" if (run, segment) in hao and (run, segment) in raw else "missing_input"
        missing += status != "ready"
        rows.append(
            {
                "run": run,
                "segment": segment,
                "hao_production_root": hao.get((run, segment), ""),
                "auxiliary_raw_wf_root": raw.get((run, segment), ""),
                "status": status,
            }
        )
    absent_runs = sorted(runs - {run for run, _ in keys})
    if absent_runs:
        raise SystemExit(f"runs absent from both input trees: {absent_runs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"segments={len(rows)} ready={len(rows)-missing} missing={missing} output={args.output}")
    if missing and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

