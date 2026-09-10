#!/usr/bin/env python3
"""Freeze non-secret replay DB values needed by exact member extraction."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


SCALARS = (
    "BEAM_param_Energy",
    "SIMU_param_HMSmomentum",
    "SIMU_param_HMSangle",
    "TARGET_param_Amu",
    "CALO_geom_Dist",
    "CALO_geom_Yaw",
)
ARRAYS = (
    "CALO_calib_TimeOffset",
    "CALO_calib_Pi0Coef",
    "CALO_calib_ElasCoef",
    "CALO_flag_MaskBlock",
)


def read_runs(path: Path) -> list[int]:
    runs = set()
    for line in path.read_text(encoding="ascii").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and not value.lower().startswith("run"):
            runs.add(int(value.split()[0]))
    return sorted(runs)


def query(run: int, table: str, args: argparse.Namespace) -> list[str]:
    command = [
        "mysql", "--protocol=TCP", "-h", args.host, "-u", args.user,
        "-D", args.database, "-N", "-B", "-e",
        f"SELECT * FROM {table} WHERE minRun <= {run} AND maxRun >= {run} ORDER BY time DESC LIMIT 1",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=os.environ)
    row = result.stdout.rstrip("\n").split("\t")
    if not row or not row[0]:
        raise RuntimeError(f"run {run}: empty DB row for {table}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default=os.environ.get("NPS_DB_HOST", "jmysql.jlab.org"))
    parser.add_argument("--database", default=os.environ.get("NPS_DB_NAME", "nps"))
    parser.add_argument("--user", default=os.environ.get("NPS_DB_USER", "dvcs"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    password = os.environ.get("MYSQL_PWD") or os.environ.get("DB_PASS")
    if not password:
        raise SystemExit("MYSQL_PWD or DB_PASS must be set")
    os.environ["MYSQL_PWD"] = password
    summary = []
    for run in read_runs(args.runs):
        run_dir = args.output_dir / f"run_{run}"
        expected = [run_dir / f"{table}.value.txt" for table in SCALARS]
        expected += [run_dir / f"{table}.txt" for table in ARRAYS]
        if all(path.is_file() and path.stat().st_size > 0 for path in expected) and not args.overwrite:
            summary.append({"run": run, "status": "existing", "snapshot_dir": run_dir})
            continue
        if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
            raise RuntimeError(
                f"run {run}: snapshot is incomplete; use --overwrite to rebuild {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        for table in (*SCALARS, *ARRAYS):
            row = query(run, table, args)
            (run_dir / f"{table}.row.tsv").write_text("\t".join(row) + "\n", encoding="ascii")
            if table in SCALARS:
                (run_dir / f"{table}.value.txt").write_text(row[0] + "\n", encoding="ascii")
                metadata = row[1:]
            else:
                if len(row) < 1080:
                    raise RuntimeError(f"run {run}: {table} has only {len(row)} columns")
                (run_dir / f"{table}.txt").write_text("\n".join(row[:1080]) + "\n", encoding="ascii")
                metadata = row[1080:]
            (run_dir / f"{table}.meta.txt").write_text("\t".join(metadata) + "\n", encoding="ascii")
        summary.append({"run": run, "status": "ok", "snapshot_dir": run_dir})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "snapshot_summary.tsv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(summary)} run snapshots under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
