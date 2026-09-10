#!/usr/bin/env python3
"""Generate one SWIF2 extraction or application job per manifest segment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def expanded(value: str) -> Path:
    # Preserve configured namespace spelling (for example /work versus its
    # symlink target); SWIF remote paths are data-management contracts.
    value = os.path.expandvars(os.path.expanduser(value))
    return Path(os.path.abspath(value))


def replace_prefix(path: str, old: str, new: str) -> str:
    return new + path[len(old):] if old and path.startswith(old) else path


def shell_command(environment_script: Path, command: list[str]) -> str:
    inner = (
        "source /etc/skel/.bashrc; "
        "source /etc/profile.d/modules.sh; "
        f"source {shlex.quote(str(environment_script))}; "
        "set -euo pipefail; "
        + shlex.join(command)
    )
    return "bash -lc " + shlex.quote(inner)


def common_job(args: argparse.Namespace, name: str, command: str) -> dict[str, object]:
    return {
        "account": args.account,
        "command": [command],
        "constraint": args.constraint,
        "cpu_cores": args.cpu_cores,
        "disk_bytes": args.disk_gb * 1_000_000_000,
        "disk_bytes_type": "scratch",
        "name": name,
        "partition": args.partition,
        "ram_bytes": args.ram_gb * 1_000_000_000,
        "stderr": str(args.log_dir / f"{name}.err"),
        "stdout": str(args.log_dir / f"{name}.out"),
        "time_secs": args.time_hours * 3600,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("extract", "apply"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--environment-script", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--raw-prefix-from", default="/cache/")
    parser.add_argument("--raw-prefix-to", default="/mss/")
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--account", default="hallc")
    parser.add_argument("--partition", default="production")
    parser.add_argument("--constraint", default="el9")
    parser.add_argument("--cpu-cores", type=int, default=1)
    parser.add_argument("--ram-gb", type=int, default=12)
    parser.add_argument("--disk-gb", type=int, default=40)
    parser.add_argument("--time-hours", type=int, default=12)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with expanded(config["paths"]["segment_manifest"]).open(newline="", encoding="ascii") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("status", "ready") == "ready"
        ]
    if not rows:
        parser.error("segment manifest has no ready rows")
    pair_dir = expanded(config["paths"]["production_pair_dir"])
    member_dir = expanded(config["paths"]["member_dir"])
    work_dir = expanded(config["paths"]["work_dir"])
    corrected_dir = expanded(config["paths"]["corrected_root_dir"])
    package_dir = expanded(config["paths"].get("package_dir", str(work_dir / "frozen_package")))
    jobs = []
    for row in rows:
        run, segment = int(row["run"]), int(row["segment"])
        suffix = f"run{run}_seg{segment}"
        hao_name = Path(row["hao_production_root"]).name
        raw_name = Path(row["auxiliary_raw_wf_root"]).name
        member_name = f"nps_production_{run}_{segment}_wf_calib.production_members.tsv"
        pair_name = f"run{run}_seg{segment}_production_exact2_pairs.tsv"
        if args.mode == "extract":
            report_name = f"extraction_report_{suffix}.tsv"
            command = [
                "python3", str(HERE / "extract_inputs.py"), "--config", str(config_path),
                "--kind", "both", "--run", str(run), "--segment", str(segment),
                "--workers", "1", "--max-events", str(args.max_events),
                "--hao-root-override", hao_name, "--raw-wf-root-override", raw_name,
                "--output-dir-override", ".", "--report", report_name,
            ]
            name = f"{args.workflow_name}_{suffix}"
            job = common_job(args, name, shell_command(args.environment_script, command))
            job["inputs"] = [
                {"local": hao_name, "remote": row["hao_production_root"]},
                {
                    "local": raw_name,
                    "remote": replace_prefix(
                        row["auxiliary_raw_wf_root"], args.raw_prefix_from, args.raw_prefix_to
                    ),
                },
            ]
            job["outputs"] = [
                {"local": pair_name, "remote": str(pair_dir / pair_name)},
                {"local": member_name, "remote": str(member_dir / member_name)},
                {
                    "local": report_name,
                    "remote": str(work_dir / "batch_reports" / "extract" / report_name),
                },
            ]
        else:
            if config["paths"].get("package_dir") is None:
                parser.error("apply mode requires paths.package_dir pointing to a promoted release")
            output_name = f"corrected_{hao_name}"
            command = [
                "python3", str(HERE.parent / "root" / "apply_root.py"),
                "--input-root", hao_name, "--output-root", output_name,
                "--package", str(package_dir), "--run", str(run),
                "--segment", str(segment), "--sidecar", member_name,
                "--max-events", str(args.max_events), "--missing-seed", "error",
            ]
            name = f"{args.workflow_name}_{suffix}"
            job = common_job(args, name, shell_command(args.environment_script, command))
            job["inputs"] = [
                {"local": hao_name, "remote": row["hao_production_root"]},
                {"local": member_name, "remote": str(member_dir / member_name)},
            ]
            job["outputs"] = [
                {"local": output_name, "remote": str(corrected_dir / hao_name)},
            ]
        jobs.append(job)
    payload = {"name": args.workflow_name, "jobs": jobs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(f"mode={args.mode} jobs={len(jobs)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
