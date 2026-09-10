#!/usr/bin/env python3
"""Plan or execute the full frozen-calibration workflow."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "derive"))
from kinematic_calibration_config import load_config, read_run_manifest, resolve_path, run_period_rows  # noqa: E402


STAGES = ("snapshots", "extract", "derive", "apply")


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def audit(config: dict) -> tuple[list[dict[str, str]], Path]:
    paths = config["paths"]
    required = (
        "run_manifest", "segment_manifest", "db_snapshot_dir", "member_dir",
        "production_pair_dir", "work_dir", "corrected_root_dir",
    )
    missing = [key for key in required if key not in paths]
    if missing:
        raise ValueError(f"config lacks pipeline paths: {missing}")
    if str(config["kinematic"].get("input_mode")) != "hao_production":
        raise ValueError("full pipeline requires kinematic.input_mode: hao_production")
    member_dir = resolve_path(config, paths["member_dir"])
    member_sources = {resolve_path(config, value) for value in paths["member_source_dirs"]}
    if member_dir not in member_sources:
        raise ValueError("paths.member_dir must also appear in paths.member_source_dirs")

    runs = set(read_run_manifest(resolve_path(config, paths["run_manifest"])))
    period_runs = {int(row["run"]) for row in run_period_rows(config)}
    if runs != period_runs:
        raise ValueError("run manifest and period coverage differ")
    manifest = resolve_path(config, paths["segment_manifest"])
    with manifest.open(newline="", encoding="ascii") as handle:
        segments = list(csv.DictReader(handle, delimiter="\t"))
    if not segments:
        raise ValueError("segment manifest is empty")
    keys = [(int(row["run"]), int(row["segment"])) for row in segments]
    if len(keys) != len(set(keys)):
        raise ValueError("segment manifest contains duplicate run/segment rows")
    extra = sorted({run for run, _ in keys} - runs)
    absent = sorted(runs - {run for run, _ in keys})
    if extra or absent:
        raise ValueError(f"segment/run mismatch: extra={extra} absent={absent}")
    bad_status = [key for key, row in zip(keys, segments) if row.get("status", "ready") != "ready"]
    if bad_status:
        raise ValueError(f"segment manifest has non-ready rows: {bad_status[:8]}")
    for row in segments:
        for field in ("hao_production_root", "auxiliary_raw_wf_root"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(f"missing {field}: {row[field]}")

    work = resolve_path(config, paths["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    summary = work / "pipeline_input_audit.txt"
    summary.write_text(
        "\n".join(
            (
                f"kinematic: {config['kinematic']['name']}",
                f"target: {config['kinematic']['target']}",
                f"runs: {len(runs)}",
                f"segments: {len(segments)}",
                f"periods: {len(config['periods'])}",
                f"member_dir: {member_dir}",
                f"pair_dir: {resolve_path(config, paths['production_pair_dir'])}",
                f"corrected_root_dir: {resolve_path(config, paths['corrected_root_dir'])}",
                "status: ready",
            )
        ) + "\n",
        encoding="ascii",
    )
    return segments, summary


def commands(
    config_path: Path, config: dict, workers: int, max_events: int, overwrite: bool
) -> dict[str, list[str]]:
    paths = config["paths"]
    database = config.get("database", {})
    work = resolve_path(config, paths["work_dir"])
    result = {
        "snapshots": [
            sys.executable, str(HERE / "build_db_snapshots.py"),
            "--runs", str(resolve_path(config, paths["run_manifest"])),
            "--output-dir", str(resolve_path(config, paths["db_snapshot_dir"])),
            "--host", str(database.get("host", "jmysql.jlab.org")),
            "--database", str(database.get("name", "nps")),
            "--user", str(database.get("user", "dvcs")),
        ],
        "extract": [
            sys.executable, str(HERE / "extract_inputs.py"), "--config", str(config_path),
            "--kind", "both", "--workers", str(workers), "--max-events", str(max_events),
        ],
        "derive": [
            sys.executable, str(HERE / "derive" / "run_kinematic_calibration.py"),
            "--config", str(config_path), "--stages", "prepare,run-scale,seed,lowe,export",
            "--execute",
        ],
        "apply": [
            sys.executable, str(HERE / "apply_dataset.py"), "--config", str(config_path),
            "--workers", str(workers), "--max-events", str(max_events),
        ],
    }
    if overwrite:
        result["snapshots"].append("--overwrite")
        result["extract"].append("--overwrite")
        result["derive"].append("--overwrite-package")
        result["apply"].append("--overwrite")
    plan = work / "pipeline_commands.sh"
    plan.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" +
        "\n".join(command_text(result[stage]) for stage in STAGES) + "\n",
        encoding="ascii",
    )
    plan.chmod(0o755)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    selected = [value.strip() for value in args.stages.split(",") if value.strip()]
    if unknown := [stage for stage in selected if stage not in STAGES]:
        parser.error(f"unknown stages: {unknown}")
    _, summary = audit(config)
    stage_commands = commands(
        config_path, config, max(1, args.workers), args.max_events, args.overwrite
    )
    print(summary.read_text(encoding="ascii"), end="")
    print(f"command plan: {resolve_path(config, config['paths']['work_dir']) / 'pipeline_commands.sh'}")
    if not args.execute:
        print("dry run only; pass --execute to run selected stages")
        return 0
    if "snapshots" in selected and not (os.environ.get("MYSQL_PWD") or os.environ.get("DB_PASS")):
        raise SystemExit("snapshots blocked: MYSQL_PWD or DB_PASS is not set")
    for stage in selected:
        executable = "mysql" if stage == "snapshots" else "root" if stage in {"extract", "apply"} else None
        if executable and shutil.which(executable) is None:
            raise SystemExit(f"{stage} blocked: {executable} is not available in PATH")
        if stage == "extract":
            nps_soft = Path(os.environ.get("NPS_SOFT", ""))
            if not nps_soft.is_dir() or not (nps_soft / "TCaloEvent.h").is_file():
                raise SystemExit(
                    "extract blocked: NPS_SOFT is unset or lacks TCaloEvent.h; "
                    "initialize the NPS replay environment"
                )
        print(f"[{stage}] starting", flush=True)
        result = subprocess.run(stage_commands[stage], check=False)
        if result.returncode:
            raise SystemExit(f"{stage} failed with rc={result.returncode}")
        print(f"[{stage}] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
