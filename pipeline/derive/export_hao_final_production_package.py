#!/usr/bin/env python3
"""Freeze selected Hao-start calibration tables into one production package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import yaml

from kinematic_calibration_config import (
    config_sha256,
    load_config,
    read_run_period_lut,
    write_run_period_lut,
)


FILES = {
    "period_photon_curve.tsv": "period photon-energy correction curve",
    "run_scalar_lut.tsv": "full-sample run scalar policy",
    "seed_block_scale.tsv": "neutralized seed-block scale, damping 0.75",
    "lowe_photon_scale.tsv": "configured low-energy photon closure scale",
    "run_period_lut.tsv": "explicit run-to-period assignment",
    "kinematic_config.yaml": "complete derivation and application configuration",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curve-tsv", required=True, type=Path)
    ap.add_argument("--run-scale-tsv", required=True, type=Path)
    ap.add_argument("--seed-scale-tsv", required=True, type=Path)
    ap.add_argument("--lowe-scale-tsv", required=True, type=Path)
    ap.add_argument("--kinematic-config", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.kinematic_config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to modify non-empty frozen package: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "period_photon_curve.tsv": args.curve_tsv,
        "run_scalar_lut.tsv": args.run_scale_tsv,
        "seed_block_scale.tsv": args.seed_scale_tsv,
        "lowe_photon_scale.tsv": args.lowe_scale_tsv,
    }
    manifest: list[dict[str, object]] = []
    for target_name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.output_dir / target_name
        shutil.copy2(source, target)
        manifest.append(
            {
                "file": target_name,
                "description": FILES[target_name],
                "source": source.resolve(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    config_target = args.output_dir / "kinematic_config.yaml"
    normalized = {key: value for key, value in config.items() if not key.startswith("_")}
    config_target.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")
    period_target = write_run_period_lut(config, args.output_dir / "run_period_lut.tsv")
    for target, source in ((config_target, args.kinematic_config), (period_target, args.kinematic_config)):
        manifest.append(
            {
                "file": target.name,
                "description": FILES[target.name],
                "source": source.resolve(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    run_rows = read_tsv(args.output_dir / "run_scalar_lut.tsv")
    period_map = read_run_period_lut(args.output_dir / "run_period_lut.tsv")
    run_set = {int(row["run"]) for row in run_rows if int(row.get("fit_ok", "1")) == 1}
    if run_set != set(period_map):
        missing_period = sorted(run_set - set(period_map))
        missing_scale = sorted(set(period_map) - run_set)
        raise RuntimeError(
            f"run coverage mismatch: no period={missing_period[:12]}, no valid run scale={missing_scale[:12]}"
        )
    curve_periods = {row["period"] for row in read_tsv(args.output_dir / "period_photon_curve.tsv")}
    missing_curves = sorted(set(period_map.values()) - curve_periods)
    if missing_curves:
        raise RuntimeError(f"periods lack photon curves: {missing_curves}")
    modes = Counter(row["support_mode"] for row in run_rows)
    seed_rows = read_tsv(args.output_dir / "seed_block_scale.tsv")
    supported_seed = sum(int(row["support_pass"]) == 1 for row in seed_rows)
    lowe_rows = read_tsv(args.output_dir / "lowe_photon_scale.tsv")
    selected_damping = float(config["low_energy"]["damping"])
    lowe_scope = str(config["low_energy"].get("scope", "global"))
    selected_lowe = [
        row
        for row in lowe_rows
        if row["fold"] == "full_sample" and abs(float(row["damping"]) - selected_damping) < 1.0e-12
    ]
    expected_lowe = 1 if lowe_scope == "global" else len(set(period_map.values()))
    if len(selected_lowe) != expected_lowe:
        raise RuntimeError(
            f"expected {expected_lowe} full_sample d={selected_damping:.2f} "
            f"{lowe_scope} low-E rows, found {len(selected_lowe)}"
        )
    if lowe_scope == "period":
        selected_periods = {row.get("period", "") for row in selected_lowe}
        missing_lowe = sorted(set(period_map.values()) - selected_periods)
        extra_lowe = sorted(selected_periods - set(period_map.values()))
        if missing_lowe or extra_lowe:
            raise RuntimeError(
                f"low-E period coverage mismatch: missing={missing_lowe}, extra={extra_lowe}"
            )

    lowe_profile = str(config["low_energy"].get("profile", "hard_window"))
    lowe_full_until = float(config["low_energy"].get("full_until_gev", selected_lowe[0]["emax"]))
    lowe_unity_at = float(config["low_energy"].get("unity_at_gev", selected_lowe[0]["emax"]))
    lowe_min = float(selected_lowe[0]["emin"])
    lowe_max = float(selected_lowe[0]["emax"])
    if lowe_profile not in {"hard_window", "smoothstep"}:
        raise RuntimeError(f"unsupported low-E profile: {lowe_profile}")
    if lowe_profile == "smoothstep" and not (
        lowe_min <= lowe_full_until < lowe_unity_at <= lowe_max
    ):
        raise RuntimeError("smooth low-E profile must satisfy emin <= full_until < unity_at <= emax")
    metadata = [
        {"key": "schema_version", "value": 3},
        {"key": "kinematic", "value": config["kinematic"]["name"]},
        {"key": "target", "value": config["kinematic"]["target"]},
        {"key": "baseline", "value": config["kinematic"]["baseline"]},
        {"key": "config_source", "value": args.kinematic_config.resolve()},
        {"key": "config_source_sha256", "value": config_sha256(args.kinematic_config)},
        {"key": "run_count", "value": len(period_map)},
        {"key": "period_count", "value": len(set(period_map.values()))},
        {"key": "seed_damping", "value": config["seed"]["damping"]},
        {"key": "lowe_damping", "value": selected_damping},
        {"key": "lowe_scope", "value": lowe_scope},
        {"key": "lowe_profile", "value": lowe_profile},
        {"key": "lowe_full_until", "value": lowe_full_until},
        {"key": "lowe_unity_at", "value": lowe_unity_at},
        {"key": "approval_status", "value": "diagnostic_unreviewed"},
    ]
    write_tsv(args.output_dir / "package_metadata.tsv", metadata)
    manifest.append(
        {
            "file": "package_metadata.tsv",
            "description": "machine-readable package identity and provenance",
            "source": args.kinematic_config.resolve(),
            "bytes": (args.output_dir / "package_metadata.tsv").stat().st_size,
            "sha256": sha256(args.output_dir / "package_metadata.tsv"),
        }
    )
    write_tsv(args.output_dir / "package_manifest.tsv", manifest)

    lines = [
        f"Frozen Hao-start {config['kinematic']['name']} production calibration package",
        "",
        "Baseline:",
        "  Input cluster energies already use Hao CALO_calib_Pi0Coef.",
        "",
        "Application per cluster:",
        "  E1 = E_Hao * period_photon_curve(period(run), E_Hao)",
        "  E2 = E1 * run_scalar(run)",
        "  E3 = E2 * seed_block_scale(seed_block)",
        f"  E_final = E3 * lowe_photon_scale({lowe_scope}) when E_Hao is in "
        f"{float(selected_lowe[0]['emin']):g}-{float(selected_lowe[0]['emax']):g} GeV",
        f"  Low-E profile: {lowe_profile}; full through {lowe_full_until:g} GeV; "
        f"unity at {lowe_unity_at:g} GeV.",
        "",
        "Policy:",
        f"  Target: {config['kinematic']['target']}; covered runs: {len(period_map)}.",
        "  Run-to-period assignments come only from run_period_lut.tsv.",
        "  Run scalar uses own-run Gaussian fit when accepted; weak runs use existing chronological support policy.",
        "  Seed columns 0-2 remain identity/QA when unsupported.",
        "  Member/hybrid residual layer is not included.",
        "  All mass peak means and sigmas used in derivation/QA come from canonical Gaussian+linear-background fits.",
        "",
        f"Run LUT rows: {len(run_rows)}",
        "Approval status: diagnostic_unreviewed",
        "Run support modes: " + ", ".join(f"{key}={value}" for key, value in sorted(modes.items())),
        f"Supported seed rows: {supported_seed}/{len(seed_rows)}",
        f"Selected low-E photon factors (d={selected_damping:.2f}, scope={lowe_scope}): "
        + ", ".join(
            f"{row.get('period', 'global')}={float(row['applied_photon_energy_scale']):.9f}"
            for row in selected_lowe
        ),
        "",
        "This package is frozen. Do not tune tables in place; create a versioned successor.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
