#!/usr/bin/env python3
"""Audit, derive, and freeze a Hao-start calibration for one kinematic.

The driver runs locally and never submits SWIF jobs. By default it writes an
input audit and reproducible command plan only. Pass --execute to run selected
stages.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from kinematic_calibration_config import (
    PROJECT_ROOT,
    load_config,
    read_run_manifest,
    resolve_path,
    run_period_rows,
    write_run_period_lut,
)


STAGES = ("prepare", "run-scale", "seed", "lowe", "export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma-separated subset of: {','.join(STAGES)}",
    )
    parser.add_argument("--execute", action="store_true", help="Run stages; default is audit/plan only")
    parser.add_argument("--overwrite-package", action="store_true")
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_ints(values: list[Any]) -> str:
    return ",".join(str(int(value)) for value in values)


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def paths(config: dict[str, Any]) -> dict[str, Path]:
    work = resolve_path(config, config["paths"]["work_dir"])
    return {
        "work": work,
        "period_lut": work / "run_period_lut.tsv",
        "baseline": work / "baseline",
        "compact": work / "baseline" / "compact",
        "summary": work / "baseline" / "compact" / "conversion_summary.tsv",
        "run_scale": work / "run_scale",
        "seed": work / "seed",
        "lowe": work / "lowe",
        "package": work / "frozen_package",
        "logs": work / "logs",
    }


def audit_inputs(config: dict[str, Any], resolved: dict[str, Path]) -> list[dict[str, object]]:
    manifest = resolve_path(config, config["paths"]["run_manifest"])
    runs = set(read_run_manifest(manifest))
    input_mode = str(config["kinematic"].get("input_mode", ""))
    if input_mode != "hao_production":
        raise ValueError("portable derivation requires kinematic.input_mode: hao_production")
    pair_by_run: Counter[int] = Counter()
    member_by_run: Counter[int] = Counter()
    production_dir = resolve_path(config, config["paths"]["production_pair_dir"])
    expected_by_run: Counter[int] = Counter()
    production_root_by_run: Counter[int] = Counter()
    auxiliary_root_by_run: Counter[int] = Counter()
    segment_manifest = resolve_path(config, config["paths"]["segment_manifest"])
    with segment_manifest.open(newline="") as handle:
        for segment in csv.DictReader(handle, delimiter="\t"):
            run, seg = int(segment["run"]), int(segment["segment"])
            if run not in runs:
                continue
            expected_by_run[run] += 1
            production_root_by_run[run] += int(Path(segment["hao_production_root"]).is_file())
            auxiliary_root_by_run[run] += int(Path(segment["auxiliary_raw_wf_root"]).is_file())
            pair_path = production_dir / f"run{run}_seg{seg}_production_exact2_pairs.tsv"
            pair_by_run[run] += int(pair_path.is_file())
            member_name = f"nps_production_{run}_{seg}_wf_calib.production_members.tsv"
            member_by_run[run] += int(
                any((resolve_path(config, source) / member_name).is_file() for source in config["paths"]["member_source_dirs"])
            )
    validation_runs = {int(run) for run in config.get("execution", {}).get("production_validation_runs", [])}
    rows: list[dict[str, object]] = []
    period_by_run = {int(row["run"]): str(row["period"]) for row in run_period_rows(config)}
    for run in sorted(runs):
        production_exists = any(production_dir.glob(f"run{run}_*_production_exact2_pairs.tsv")) or (
            production_dir / f"run{run}_production_exact2_pairs.tsv"
        ).is_file()
        rows.append(
            {
                "run": run,
                "period": period_by_run[run],
                "expected_segments": expected_by_run[run],
                "hao_production_roots": production_root_by_run[run],
                "auxiliary_raw_wf_roots": auxiliary_root_by_run[run],
                "pair_files": pair_by_run[run],
                "member_sidecars": member_by_run[run],
                "pair_member_complete": int(
                    pair_by_run[run] > 0
                    and pair_by_run[run] == member_by_run[run]
                    and pair_by_run[run] == expected_by_run[run]
                ),
                "production_validation_requested": int(run in validation_runs),
                "production_pair_exists": int(production_exists),
            }
        )
    write_tsv(resolved["work"] / "input_audit.tsv", rows)
    missing_sidecars = [row["run"] for row in rows if not int(row["pair_member_complete"])]
    missing_production = [
        row["run"]
        for row in rows
        if int(row["production_validation_requested"]) and not int(row["production_pair_exists"])
    ]
    summary = [
        f"kinematic: {config['kinematic']['name']}",
        f"target: {config['kinematic']['target']}",
        f"input mode: {input_mode}",
        f"configured runs: {len(rows)}",
        f"runs with complete pair/member inputs: {sum(int(row['pair_member_complete']) for row in rows)}",
        f"runs missing complete pair/member inputs: {len(missing_sidecars)}",
        f"requested production QA runs missing pairs: {len(missing_production)}",
        f"DB credential present: {int(bool(os.environ.get('MYSQL_PWD') or os.environ.get('DB_PASS')))}",
    ]
    summary.extend(
        (
            f"configured segments: {sum(expected_by_run.values())}",
            f"Hao production ROOTs available: {sum(production_root_by_run.values())}",
            f"auxiliary raw-WF ROOTs available: {sum(auxiliary_root_by_run.values())}",
        )
    )
    if missing_sidecars:
        summary.append("missing pair/member runs: " + ",".join(map(str, missing_sidecars)))
    if missing_production:
        summary.append("missing production QA runs: " + ",".join(map(str, missing_production)))
    (resolved["work"] / "input_audit_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    (resolved["work"] / "missing_member_runs.txt").write_text(
        "".join(f"{run}\n" for run in missing_sidecars), encoding="ascii"
    )
    return rows


def build_commands(config: dict[str, Any], resolved: dict[str, Path], overwrite: bool) -> dict[str, list[str]]:
    python = sys.executable
    manifest = resolve_path(config, config["paths"]["run_manifest"])
    selection = config["selection"]
    fit = config["fit"]
    run_cfg = config["run_scale"]
    seed_cfg = config["seed"]
    lowe_cfg = config["low_energy"]
    validation_runs = csv_ints(config["execution"]["production_validation_runs"])
    excluded = csv_ints(run_cfg.get("excluded_calibration_runs", []))

    convert = [
        python,
        str(PROJECT_ROOT / "make_hao_baseline_compact_from_production_pairs.py"),
        "--pairs-dir", str(resolve_path(config, config["paths"]["production_pair_dir"])),
        "--runs", str(manifest),
    ]
    for source in config["paths"]["member_source_dirs"]:
        convert.extend(("--sidecar-dir", str(resolve_path(config, source))))
    convert.extend(
        (
            "--output-dir", str(resolved["compact"]),
            "--summary", str(resolved["compact"] / "input_paths.txt"),
            "--workers", str(int(config["execution"].get("conversion_workers", 1))),
            "--max-missing-fraction",
            str(float(config["execution"].get("max_pair_member_join_loss_fraction", 0.001))),
        )
    )
    run_scale = [
        python,
        str(PROJECT_ROOT / "validate_hao_direct_period_photon_run_scale.py"),
        "--conversion-summary", str(resolved["summary"]),
        "--production-pair-dir", str(resolve_path(config, config["paths"]["production_pair_dir"])),
        "--output-dir", str(resolved["run_scale"]),
        "--runs", validation_runs,
        "--run-period-lut", str(resolved["period_lut"]),
        "--timing-center", str(selection["timing_center_ns"]),
        "--timing-window", str(selection["timing_half_window_ns"]),
        "--emin-range", str(selection["min_photon_energy_gev"]), str(selection["max_photon_energy_gev"]),
        "--min-fit-entries", str(fit["min_entries"]),
        "--min-photon-cell-entries", str(fit["min_photon_cell_entries"]),
        "--fit-max-mu-err", str(fit["max_mu_error_mev"]),
        "--fit-warn-chi2-ndf", str(fit["warn_chi2_ndf"]),
        "--fit-max-window-shift", str(fit["max_window_shift_mev"]),
        "--max-run-update-frac", str(run_cfg["max_update_fraction"]),
        "--support-max-adjacent-gap", str(run_cfg["support_max_adjacent_gap"]),
        "--support-max-run-distance", str(run_cfg["support_max_run_distance"]),
        "--support-max-members", str(run_cfg["support_max_members"]),
        "--run-own-max-mu-err", str(run_cfg["run_own_max_mu_error_mev"]),
        "--support-shrink-tau-mev", str(run_cfg["support_shrink_tau_mev"]),
        "--one-sided-support-weight", str(run_cfg["one_sided_support_weight"]),
        "--excluded-calibration-runs", excluded,
        "--edge-support-policy",
    ]
    common = [
        "--conversion-summary", str(resolved["summary"]),
        "--baseline-dir", str(resolved["run_scale"]),
        "--run-period-lut", str(resolved["period_lut"]),
        "--timing-center", str(selection["timing_center_ns"]),
        "--timing-window", str(selection["timing_half_window_ns"]),
        "--emin-range", str(selection["min_photon_energy_gev"]), str(selection["max_photon_energy_gev"]),
        "--min-fit-entries", str(fit["min_entries"]),
        "--min-photon-cell-entries", str(fit["min_photon_cell_entries"]),
        "--fit-max-mu-err", str(fit["max_mu_error_mev"]),
        "--fit-warn-chi2-ndf", str(fit["warn_chi2_ndf"]),
        "--fit-max-window-shift", str(fit["max_window_shift_mev"]),
        "--min-seed-events", str(seed_cfg["min_events"]),
        "--seed-shrink-events", str(seed_cfg["shrink_events"]),
        "--max-seed-update-frac", str(seed_cfg["max_update_fraction"]),
        "--left-cols", *[str(col) for col in seed_cfg["identity_columns"]],
        "--excluded-calibration-runs", excluded,
    ]
    seed = [python, str(PROJECT_ROOT / "validate_hao_direct_run_scale_seed.py"), *common, "--output-dir", str(resolved["seed"])]
    lowe = [
        python,
        str(PROJECT_ROOT / "validate_hao_direct_run_seed_lowe.py"),
        *common,
        "--seed-scale-tsv", str(resolved["seed"] / "frozen_seed_package" / f"seed_scale_d{float(seed_cfg['damping']):.2f}.tsv"),
        "--seed-damping", str(seed_cfg["damping"]),
        "--lowe-range", *[str(value) for value in lowe_cfg["range_gev"]],
        "--lowe-scope", str(lowe_cfg.get("scope", "global")),
        "--max-lowe-energy-update-frac", str(lowe_cfg["max_update_fraction"]),
        "--output-dir", str(resolved["lowe"]),
    ]
    export = [
        python,
        str(PROJECT_ROOT / "export_hao_final_production_package.py"),
        "--curve-tsv", str(resolved["run_scale"] / "frozen_package" / "period_photon_curve.tsv"),
        "--run-scale-tsv", str(resolved["run_scale"] / "frozen_package" / f"run_scale_d{float(run_cfg['damping']):.2f}.tsv"),
        "--seed-scale-tsv", str(resolved["seed"] / "frozen_seed_package" / f"seed_scale_d{float(seed_cfg['damping']):.2f}.tsv"),
        "--lowe-scale-tsv", str(resolved["lowe"] / "frozen_lowe_package" / "lowe_photon_energy_scales.tsv"),
        "--kinematic-config", str(Path(config["_config_path"])),
        "--output-dir", str(resolved["package"]),
    ]
    if overwrite:
        export.append("--overwrite")
    return {
        "prepare_compact": convert,
        "run-scale": run_scale,
        "seed": seed,
        "lowe": lowe,
        "export": export,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    requested = [value.strip() for value in args.stages.split(",") if value.strip()]
    unknown = [stage for stage in requested if stage not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {','.join(unknown)}")
    resolved = paths(config)
    resolved["logs"].mkdir(parents=True, exist_ok=True)
    write_run_period_lut(config, resolved["period_lut"])
    audit_rows = audit_inputs(config, resolved)
    commands = build_commands(config, resolved, args.overwrite_package)
    ordered = ["prepare_compact", "run-scale", "seed", "lowe", "export"]
    selected = [name for name in ordered if (name.startswith("prepare") and "prepare" in requested) or name in requested]
    plan_path = resolved["work"] / "commands.sh"
    plan_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(PROJECT_ROOT))}"]
    plan_lines.extend(command_text(commands[name]) for name in selected)
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="ascii")
    plan_path.chmod(0o755)
    print((resolved["work"] / "input_audit_summary.txt").read_text(), end="")
    print(f"command plan: {plan_path}")
    if not args.execute:
        print("dry run only; pass --execute to run selected stages")
        return 0
    if any(not int(row["pair_member_complete"]) for row in audit_rows) and "prepare" in requested:
        raise SystemExit("prepare blocked: one or more configured runs lack complete compact/member inputs")
    for name in selected:
        log_path = resolved["logs"] / f"{name}.log"
        print(f"[{name}] starting; log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(commands[name], cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(f"{name} failed with rc={result.returncode}; see {log_path}")
        print(f"[{name}] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
