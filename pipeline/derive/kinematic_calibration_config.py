#!/usr/bin/env python3
"""Load and validate configuration for one kinematic calibration package."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")
    kinematic = config.get("kinematic")
    if not isinstance(kinematic, dict):
        raise ValueError("kinematic mapping is required")
    for field in ("name", "target", "baseline"):
        if not str(kinematic.get(field, "")).strip():
            raise ValueError(f"kinematic.{field} is required")
    if str(kinematic["baseline"]) != "hao_pi0_db":
        raise ValueError("only the validated hao_pi0_db baseline is currently supported")
    input_mode = str(kinematic.get("input_mode", ""))
    if input_mode != "hao_production":
        raise ValueError("kinematic.input_mode must be hao_production")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("paths mapping is required")
    for field in (
        "run_manifest", "segment_manifest", "member_source_dirs",
        "production_pair_dir", "work_dir",
    ):
        if field not in paths:
            raise ValueError(f"paths.{field} is required")
    if not isinstance(paths["member_source_dirs"], list) or not paths["member_source_dirs"]:
        raise ValueError("paths.member_source_dirs must be a non-empty list")
    periods = config.get("periods")
    if not isinstance(periods, list) or not periods:
        raise ValueError("periods must be a non-empty list")
    names: set[str] = set()
    for period in periods:
        if not isinstance(period, dict) or not str(period.get("name", "")).strip():
            raise ValueError("every period requires a name")
        name = str(period["name"])
        if name in names:
            raise ValueError(f"duplicate period name: {name}")
        names.add(name)
        explicit = "runs" in period
        ranged = "run_min" in period and "run_max" in period
        if explicit == ranged:
            raise ValueError(f"period {name}: specify either runs or run_min/run_max")
        if ranged and int(period["run_min"]) > int(period["run_max"]):
            raise ValueError(f"period {name}: run_min exceeds run_max")
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection mapping is required")
    for field in ("timing_center_ns", "timing_half_window_ns", "min_photon_energy_gev", "max_photon_energy_gev"):
        if field not in selection:
            raise ValueError(f"selection.{field} is required")
    if float(selection["min_photon_energy_gev"]) >= float(selection["max_photon_energy_gev"]):
        raise ValueError("selection photon-energy range is invalid")
    low_energy = config.get("low_energy")
    if not isinstance(low_energy, dict):
        raise ValueError("low_energy mapping is required")
    if str(low_energy.get("scope", "global")) not in {"global", "period"}:
        raise ValueError("low_energy.scope must be global or period")
    profile = str(low_energy.get("profile", "hard_window"))
    if profile not in {"hard_window", "smoothstep"}:
        raise ValueError("low_energy.profile must be hard_window or smoothstep")
    if profile == "smoothstep":
        emin, emax = [float(value) for value in low_energy["range_gev"]]
        full_until = float(low_energy["full_until_gev"])
        unity_at = float(low_energy["unity_at_gev"])
        if not (emin <= full_until < unity_at <= emax):
            raise ValueError("smooth low-energy profile requires emin <= full_until < unity_at <= emax")


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def read_run_manifest(path: Path) -> list[int]:
    runs: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.lower().startswith("run"):
                continue
            try:
                runs.add(int(stripped.split()[0]))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid run row") from error
    if not runs:
        raise ValueError(f"{path}: no runs found")
    return sorted(runs)


def run_period_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    manifest = resolve_path(config, config["paths"]["run_manifest"])
    runs = read_run_manifest(manifest)
    rows: list[dict[str, object]] = []
    for run in runs:
        matches = []
        for period in config["periods"]:
            if "runs" in period:
                contains = run in {int(value) for value in period["runs"]}
            else:
                contains = int(period["run_min"]) <= run <= int(period["run_max"])
            if contains:
                matches.append(str(period["name"]))
        if len(matches) != 1:
            raise ValueError(f"run {run}: expected exactly one period, found {matches}")
        rows.append(
            {
                "run": run,
                "period": matches[0],
                "kinematic": str(config["kinematic"]["name"]),
                "target": str(config["kinematic"]["target"]),
            }
        )
    return rows


def write_run_period_lut(config: dict[str, Any], output: Path) -> Path:
    rows = run_period_rows(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output


def read_run_period_lut(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open(encoding="ascii") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            run, period = int(row["run"]), row["period"].strip()
            if run in mapping and mapping[run] != period:
                raise ValueError(f"{path}: conflicting period assignments for run {run}")
            if not period:
                raise ValueError(f"{path}: empty period for run {run}")
            mapping[run] = period
    if not mapping:
        raise ValueError(f"{path}: empty run-period LUT")
    return mapping


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
