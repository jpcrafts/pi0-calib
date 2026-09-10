#!/usr/bin/env python3
"""Hao-start full_iter2 plus trusted merged run-band energy layer."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import validate_hao_full706_fitband_energy as fitband
import validate_hao_full_iter2_pairspace as pairspace
import validate_hao_period_energy_smoothed_seed as v
import validate_hao_start_layered_member_stack as base
from build_energy_correction_stack import M_PI0_MEV
from pi0_mass_fit import DEFAULT_CONFIG, FitConfig, fit_pi0_mass


PHOTON_ENERGY_KNOTS = np.asarray([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25], dtype=float)
PHOTON_CELL_EDGES = np.asarray([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.8, 2.5], dtype=float)
PERIOD_PHOTON_KNOTS = np.asarray([0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0], dtype=float)
PERIOD_PHOTON_CELL_EDGES = PERIOD_PHOTON_KNOTS.copy()
DENSE_LOW_PERIOD_PHOTON_KNOTS = np.asarray(
    [0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0], dtype=float
)
DENSE_LOW_PERIOD_PHOTON_CELL_EDGES = DENSE_LOW_PERIOD_PHOTON_KNOTS.copy()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[105.0, 165.0])
    ap.add_argument("--min-fit-entries", type=int, default=200)
    ap.add_argument("--min-seed-events", type=int, default=20)
    ap.add_argument("--min-merged-core-entries", type=int, default=500)
    ap.add_argument("--fit-max-mu-err", type=float, default=0.50)
    ap.add_argument("--fit-max-chi2-ndf", type=float, default=math.inf)
    ap.add_argument("--fit-warn-chi2-ndf", type=float, default=3.0)
    ap.add_argument("--fit-max-window-shift", type=float, default=0.50)
    ap.add_argument("--trusted-damping", type=float, default=0.35)
    ap.add_argument("--max-trusted-mass-update-frac", type=float, default=0.005)
    ap.add_argument("--max-global-lowe-mass-update-frac", type=float, default=0.0075)
    ap.add_argument("--global-anchor-transition-halfwidth", type=float, default=0.05)
    ap.add_argument("--min-photon-cell-entries", type=int, default=500)
    ap.add_argument("--max-photon-energy-update-frac", type=float, default=0.015)
    ap.add_argument("--max-period-photon-update-frac", type=float, default=0.03)
    ap.add_argument("--trusted-energy-max", type=float, default=1.2)
    ap.add_argument("--min-band-fit-entries", type=int, default=5000)
    ap.add_argument("--energy-damping", type=float, default=0.75)
    ap.add_argument("--period-energy-damping", type=float, default=0.75)
    ap.add_argument("--energy-bin-damping", action="append", default=[])
    ap.add_argument("--period-energy-bin-damping", action="append", default=[])
    ap.add_argument("--seed-damping", type=float, default=0.75)
    ap.add_argument("--max-energy-mass-update-frac", type=float, default=0.03)
    ap.add_argument("--max-period-energy-mass-update-frac", type=float, default=0.015)
    ap.add_argument("--max-seed-energy-update-frac", type=float, default=0.005)
    ap.add_argument("--max-fitband-mass-update-frac", type=float, default=0.015)
    ap.add_argument("--left-cols", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--period-mode", choices=["hao_lh2_cycles", "stat3", "stat_n"], default="hao_lh2_cycles")
    ap.add_argument("--n-run-bands", type=int, default=6)
    ap.add_argument("--smooth-degree", type=int, default=2)
    ap.add_argument("--lowe-bin-patch", action="append", default=[])
    ap.add_argument("--lowe-run-patch", action="append", default=[])
    ap.add_argument("--lowe-patch-bands", type=int, default=0)
    ap.add_argument("--lowe-patch-emin", type=float, default=0.0)
    ap.add_argument("--lowe-patch-emax", type=float, default=1.0)
    ap.add_argument("--lowe-patch-damping", type=float, default=0.75)
    ap.add_argument("--max-lowe-patch-mass-update-frac", type=float, default=0.0075)
    ap.add_argument("--seed-regularization", choices=["legacy", "centered_shrink"], default="centered_shrink")
    ap.add_argument("--seed-shrink-events", type=float, default=200.0)
    ap.add_argument("--support-groups", default="")
    ap.add_argument("--support-max-adjacent-gap", type=int, default=5)
    ap.add_argument("--support-max-run-distance", type=int, default=8)
    ap.add_argument("--support-max-members", type=int, default=9)
    ap.add_argument("--support-max-neighbor-mu-diff", type=float, default=2.0)
    ap.add_argument("--candidate-model", action="append", default=[])
    return ap.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="ascii")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate_specs() -> list[dict[str, object]]:
    return [
        {"model": "hao_baseline", "mode": "hao_identity", "damping": 0.0},
        {"model": "full_iter2", "mode": "none", "damping": 0.0},
        {"model": "fitBand_stat12_d075", "mode": "reference_fitband", "damping": 0.75},
        {"model": "canonical_runlocal_merged_d025", "mode": "canonical_runlocal_merged", "damping": 0.25},
        {"model": "fixed4_global06_08_d050", "mode": "fixed4_global_lowe", "damping": 0.50},
        {"model": "fixed4_global06_08_d075", "mode": "fixed4_global_lowe", "damping": 0.75},
        {"model": "fixed4_global06_08_d100", "mode": "fixed4_global_lowe", "damping": 1.00},
        {"model": "fixed4_smooth06_12_d075", "mode": "fixed4_smooth_lowe", "damping": 0.75},
        {"model": "fixed4_smooth06_12_d100", "mode": "fixed4_smooth_lowe", "damping": 1.00},
        {"model": "fixed4_photoncurve_lam1_d100", "mode": "fixed4_photon_curve", "damping": 1.00, "smooth_lambda": 1.0},
        {"model": "fixed4_photoncurve_lam10_d075", "mode": "fixed4_photon_curve", "damping": 0.75, "smooth_lambda": 10.0},
        {"model": "fixed4_photoncurve_lam10_d100", "mode": "fixed4_photon_curve", "damping": 1.00, "smooth_lambda": 10.0},
        {"model": "fixed4_photoncurve_lam100_d100", "mode": "fixed4_photon_curve", "damping": 1.00, "smooth_lambda": 100.0},
        {"model": "hao_global_photon_l10_d075", "mode": "hao_global_photon_curve", "damping": 0.75, "smooth_lambda": 10.0},
        {"model": "hao_period_photon_l10_d075", "mode": "hao_period_photon_curve", "damping": 0.75, "smooth_lambda": 10.0},
        {"model": "hao_period_photon_l10_d100", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 10.0},
        {"model": "hao_period_photon_l10_d100_low045", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 10.0, "low_positive_clip_frac": 0.045},
        {"model": "hao_period_photon_l10_d100_low060", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 10.0, "low_positive_clip_frac": 0.060},
        {"model": "hao_period_photon_l30_d100_low060", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 30.0, "low_positive_clip_frac": 0.060},
        {"model": "hao_period_photon_l10_d100_low045_k07", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 10.0, "low_positive_clip_frac": 0.045, "dense_low_knots": True},
        {"model": "hao_period_photon_l10_d100_low045_k07_w150", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 10.0, "low_positive_clip_frac": 0.045, "dense_low_knots": True, "low_constraint_weight": 1.5},
        {"model": "hao_period_photon_l10_d100_low045_k07_w200", "mode": "hao_period_photon_curve", "damping": 1.00, "smooth_lambda": 10.0, "low_positive_clip_frac": 0.045, "dense_low_knots": True, "low_constraint_weight": 2.0},
        {"model": "trusted_fixed4_lowE_d015", "mode": "fixed4", "damping": 0.15},
        {"model": "trusted_fixed4_lowE_d025", "mode": "fixed4", "damping": 0.25},
        {"model": "trusted_fixed4_lowE_d035", "mode": "fixed4", "damping": 0.35},
        {"model": "trusted_fixed4_lowE_d050", "mode": "fixed4", "damping": 0.50},
        {"model": "trusted_stat12_lowE_d015", "mode": "stat12", "damping": 0.15},
        {"model": "trusted_stat12_lowE_d025", "mode": "stat12", "damping": 0.25},
        {"model": "trusted_stat12_lowE_d035", "mode": "stat12", "damping": 0.35},
        {"model": "trusted_stat12_lowE_d050", "mode": "stat12", "damping": 0.50},
    ]


ACTIVE_MODELS: set[str] | None = None


def active_candidate_specs() -> list[dict[str, object]]:
    specs = candidate_specs()
    return specs if ACTIVE_MODELS is None else [spec for spec in specs if str(spec["model"]) in ACTIVE_MODELS]


def model_names() -> list[str]:
    return [str(x["model"]) for x in active_candidate_specs()]


def build_band_map(train: list[dict[str, object]], mode: str) -> dict[int, str]:
    if mode == "fixed4":
        return {int(ev["run"]): fitband.fixed4_band(int(ev["run"])) for ev in train}
    if mode == "stat12":
        return v.build_stat_run_bands(train, 12)
    raise ValueError(f"unknown mode {mode}")


def nearest_band(run: int, band_map: dict[int, str]) -> str:
    if run in band_map:
        return band_map[run]
    nearest = min(band_map, key=lambda r: abs(r - run))
    return band_map[nearest]


def ebin_label(ev: dict[str, object]) -> str | None:
    eb = base.energy_bin(float(ev["min_e"]))
    return base.bin_label(eb) if eb is not None else None


def ebin_hi(label: str) -> float:
    return float(label.split("-")[1])


def canonical_fit_config(args: argparse.Namespace, min_entries: int) -> FitConfig:
    return FitConfig(
        **{
            **DEFAULT_CONFIG.__dict__,
            "min_entries": min_entries,
            "max_mu_err": args.fit_max_mu_err,
            "max_chi2_ndf": args.fit_max_chi2_ndf,
            "warn_chi2_ndf": args.fit_warn_chi2_ndf,
            "max_window_shift": args.fit_max_window_shift,
        }
    )


def derive_trusted_updates(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    mode: str,
    damping: float,
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], dict[int, str], list[dict[str, object]]]:
    band_map = build_band_map(train, mode)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ev, mass in zip(train, train_masses):
        label = ebin_label(ev)
        if label is None or ebin_hi(label) > args.trusted_energy_max:
            continue
        grouped[(nearest_band(int(ev["run"]), band_map), label)].append(float(mass))

    updates: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    for key, vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        fit = fit_pi0_mass(arr, canonical_fit_config(args, args.min_merged_core_entries))
        ok = int(fit["quality_ok"]) == 1
        full_factor = M_PI0_MEV / float(fit["mu_mev"]) if ok else 1.0
        applied = base.clamp(full_factor**damping, args.max_trusted_mass_update_frac) if ok else 1.0
        rows.append(
            {
                "fold": fold,
                "model": model,
                "band_mode": mode,
                "band": key[0],
                "energy_bin": key[1],
                "events": len(vals),
                "fit_definition": "canonical_gaussian_linear_118_150",
                "canonical_fit_entries": int(fit["window_entries"]),
                "canonical_fit_ok": int(fit["quality_ok"]),
                "canonical_fit_reasons": fit["quality_reasons"],
                "canonical_fit_warnings": fit["quality_warnings"],
                "canonical_mu_residual_mev": fit["mu_residual_mev"],
                "canonical_mu_err_mev": fit["mu_err_mev"],
                "canonical_sigma_mev": fit["sigma_mev"],
                "canonical_chi2_ndf": fit["chi2_ndf"],
                "canonical_window_shift_mev": fit["max_window_shift_mev"],
                "support_pass": int(ok),
                "factor_full": full_factor,
                "damping": damping,
                "clip_frac": args.max_trusted_mass_update_frac,
                "factor_applied": applied,
            }
        )
        if ok:
            updates[key] = applied
    return updates, band_map, rows


def apply_trusted(mass: float, ev: dict[str, object], updates: dict[tuple[str, str], float], band_map: dict[int, str]) -> float:
    label = ebin_label(ev)
    if label is None:
        return mass
    key = (nearest_band(int(ev["run"]), band_map), label)
    return float(mass) * updates.get(key, 1.0)


def run_epoch_map(runs: list[int], max_gap: int) -> dict[int, int]:
    epochs: dict[int, int] = {}
    epoch = 0
    previous: int | None = None
    for run in sorted(runs):
        if previous is not None and run - previous > max_gap:
            epoch += 1
        epochs[run] = epoch
        previous = run
    return epochs


def derive_canonical_runlocal_merged_updates(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    damping: float,
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, str], float], list[dict[str, object]]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for ev, mass in zip(train, train_masses):
        label = ebin_label(ev)
        if label is None or ebin_hi(label) > args.trusted_energy_max:
            continue
        grouped[(int(ev["run"]), label)].append(float(mass))

    runs = sorted({int(ev["run"]) for ev in train})
    epochs = run_epoch_map(runs, args.support_max_adjacent_gap)
    labels = sorted({label for _, label in grouped}, key=lambda label: float(label.split("-")[0]))
    config = canonical_fit_config(args, args.min_merged_core_entries)
    individual = {key: fit_pi0_mass(np.asarray(vals), config) for key, vals in grouped.items()}

    updates: dict[tuple[int, str], float] = {}
    rows: list[dict[str, object]] = []
    for target_run in runs:
        for label in labels:
            target_vals = grouped.get((target_run, label), [])
            if not target_vals:
                continue
            members = [target_run]
            values = list(target_vals)
            fit = individual[(target_run, label)]
            blocked_sides: set[str] = set()
            candidates = sorted(
                (
                    run
                    for run in runs
                    if run != target_run
                    and epochs[run] == epochs[target_run]
                    and abs(run - target_run) <= args.support_max_run_distance
                ),
                key=lambda run: (abs(run - target_run), 0 if run < target_run else 1, run),
            )
            decisions: list[str] = []
            for candidate_run in candidates:
                if int(fit["quality_ok"]) == 1 or len(members) >= args.support_max_members:
                    break
                side = "left" if candidate_run < target_run else "right"
                if side in blocked_sides:
                    decisions.append(f"{candidate_run}:blocked")
                    continue
                candidate_fit = individual.get((candidate_run, label))
                if candidate_fit is None:
                    continue
                current_mu = float(fit["mu_mev"])
                candidate_mu = float(candidate_fit["mu_mev"])
                compatible = True
                if int(candidate_fit["quality_ok"]) == 1 and math.isfinite(current_mu) and math.isfinite(candidate_mu):
                    compatible = abs(candidate_mu - current_mu) <= args.support_max_neighbor_mu_diff
                if not compatible:
                    blocked_sides.add(side)
                    decisions.append(f"{candidate_run}:incompatible")
                    continue
                members.append(candidate_run)
                members.sort()
                values.extend(grouped.get((candidate_run, label), []))
                fit = fit_pi0_mass(np.asarray(values), config)
                decisions.append(f"{candidate_run}:added")

            ok = int(fit["quality_ok"]) == 1
            full_factor = M_PI0_MEV / float(fit["mu_mev"]) if ok else 1.0
            applied = base.clamp(full_factor**damping, args.max_trusted_mass_update_frac) if ok else 1.0
            if ok:
                updates[(target_run, label)] = applied
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "band_mode": "canonical_runlocal_merged",
                    "target_run": target_run,
                    "energy_bin": label,
                    "member_runs": ",".join(str(run) for run in members),
                    "n_member_runs": len(members),
                    "max_target_distance": max(abs(run - target_run) for run in members),
                    "support_decisions": ";".join(decisions),
                    "events": len(values),
                    "fit_definition": "canonical_gaussian_linear_118_150",
                    "canonical_fit_entries": int(fit["window_entries"]),
                    "canonical_fit_ok": int(fit["quality_ok"]),
                    "canonical_fit_reasons": fit["quality_reasons"],
                    "canonical_fit_warnings": fit["quality_warnings"],
                    "canonical_mu_residual_mev": fit["mu_residual_mev"],
                    "canonical_mu_err_mev": fit["mu_err_mev"],
                    "canonical_sigma_mev": fit["sigma_mev"],
                    "canonical_chi2_ndf": fit["chi2_ndf"],
                    "canonical_window_shift_mev": fit["max_window_shift_mev"],
                    "support_pass": int(ok),
                    "factor_full": full_factor,
                    "damping": damping,
                    "clip_frac": args.max_trusted_mass_update_frac,
                    "factor_applied": applied,
                }
            )
    return updates, rows


def apply_canonical_runlocal_merged(
    mass: float,
    ev: dict[str, object],
    updates: dict[tuple[int, str], float],
) -> float:
    label = ebin_label(ev)
    if label is None:
        return mass
    return float(mass) * updates.get((int(ev["run"]), label), 1.0)


def derive_global_lowe_anchor(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    damping: float,
    args: argparse.Namespace,
) -> tuple[float, dict[str, object]]:
    values = np.asarray(
        [mass for ev, mass in zip(train, train_masses) if ebin_label(ev) == "0.6-0.8"],
        dtype=float,
    )
    fit = fit_pi0_mass(values, canonical_fit_config(args, args.min_merged_core_entries))
    ok = int(fit["quality_ok"]) == 1
    full_factor = M_PI0_MEV / float(fit["mu_mev"]) if ok else 1.0
    applied = base.clamp(full_factor**damping, args.max_global_lowe_mass_update_frac) if ok else 1.0
    row = {
        "fold": fold,
        "model": model,
        "band_mode": "global_0.6-0.8",
        "energy_bin": "0.6-0.8",
        "events": len(values),
        "fit_definition": "canonical_gaussian_linear_118_150",
        "canonical_fit_entries": int(fit["window_entries"]),
        "canonical_fit_ok": int(fit["quality_ok"]),
        "canonical_fit_reasons": fit["quality_reasons"],
        "canonical_fit_warnings": fit["quality_warnings"],
        "canonical_mu_residual_mev": fit["mu_residual_mev"],
        "canonical_mu_err_mev": fit["mu_err_mev"],
        "canonical_sigma_mev": fit["sigma_mev"],
        "canonical_chi2_ndf": fit["chi2_ndf"],
        "canonical_window_shift_mev": fit["max_window_shift_mev"],
        "support_pass": int(ok),
        "factor_full": full_factor,
        "damping": damping,
        "clip_frac": args.max_global_lowe_mass_update_frac,
        "factor_applied": applied,
    }
    return applied, row


def apply_global_lowe_anchor(mass: float, ev: dict[str, object], factor: float) -> float:
    return float(mass) * factor if ebin_label(ev) == "0.6-0.8" else float(mass)


def derive_smooth_lowe_anchors(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    damping: float,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    factors: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for label in ["0.6-0.8", "0.8-1.0", "1.0-1.2"]:
        values = np.asarray(
            [mass for ev, mass in zip(train, train_masses) if ebin_label(ev) == label],
            dtype=float,
        )
        fit = fit_pi0_mass(values, canonical_fit_config(args, args.min_merged_core_entries))
        ok = int(fit["quality_ok"]) == 1
        full_factor = M_PI0_MEV / float(fit["mu_mev"]) if ok else 1.0
        applied = base.clamp(full_factor**damping, args.max_global_lowe_mass_update_frac) if ok else 1.0
        factors[label] = applied
        rows.append(
            {
                "fold": fold,
                "model": model,
                "band_mode": "smooth_global_0.6-1.2",
                "energy_bin": label,
                "events": len(values),
                "fit_definition": "canonical_gaussian_linear_118_150",
                "canonical_fit_entries": int(fit["window_entries"]),
                "canonical_fit_ok": int(fit["quality_ok"]),
                "canonical_fit_reasons": fit["quality_reasons"],
                "canonical_fit_warnings": fit["quality_warnings"],
                "canonical_mu_residual_mev": fit["mu_residual_mev"],
                "canonical_mu_err_mev": fit["mu_err_mev"],
                "canonical_sigma_mev": fit["sigma_mev"],
                "canonical_chi2_ndf": fit["chi2_ndf"],
                "canonical_window_shift_mev": fit["max_window_shift_mev"],
                "support_pass": int(ok),
                "factor_full": full_factor,
                "damping": damping,
                "clip_frac": args.max_global_lowe_mass_update_frac,
                "factor_applied": applied,
            }
        )
    return factors, rows


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def blend_factor(energy: float, center: float, halfwidth: float, left: float, right: float) -> float:
    low = center - halfwidth
    high = center + halfwidth
    if energy <= low:
        return left
    if energy >= high:
        return right
    weight = smoothstep((energy - low) / (high - low))
    return left + weight * (right - left)


def apply_smooth_lowe_anchors(
    mass: float,
    ev: dict[str, object],
    factors: dict[str, float],
    halfwidth: float,
) -> float:
    energy = float(ev["min_e"])
    f0 = factors["0.6-0.8"]
    f1 = factors["0.8-1.0"]
    f2 = factors["1.0-1.2"]
    if energy < 0.8 - halfwidth:
        factor = f0
    elif energy < 0.8 + halfwidth:
        factor = blend_factor(energy, 0.8, halfwidth, f0, f1)
    elif energy < 1.0 - halfwidth:
        factor = f1
    elif energy < 1.0 + halfwidth:
        factor = blend_factor(energy, 1.0, halfwidth, f1, f2)
    elif energy < 1.2 - halfwidth:
        factor = f2
    elif energy < 1.2 + halfwidth:
        factor = blend_factor(energy, 1.2, halfwidth, f2, 1.0)
    else:
        factor = 1.0
    return float(mass) * factor


def photon_energy_features(energy: float) -> dict[int, float]:
    # Final knot is fixed to log(scale)=0, enforcing a smooth return to unity.
    if energy >= PHOTON_ENERGY_KNOTS[-1]:
        return {}
    nfree = len(PHOTON_ENERGY_KNOTS) - 1
    if energy <= PHOTON_ENERGY_KNOTS[0]:
        return {0: 1.0}
    high = int(np.searchsorted(PHOTON_ENERGY_KNOTS, energy))
    low = high - 1
    fraction = (energy - PHOTON_ENERGY_KNOTS[low]) / (
        PHOTON_ENERGY_KNOTS[high] - PHOTON_ENERGY_KNOTS[low]
    )
    output: dict[int, float] = {}
    if low < nfree:
        output[low] = 1.0 - fraction
    if high < nfree:
        output[high] = fraction
    return output


def photon_cell_index(energy: float) -> int:
    return min(
        max(int(np.searchsorted(PHOTON_CELL_EDGES, energy, side="right") - 1), 0),
        len(PHOTON_CELL_EDGES) - 2,
    )


def derive_photon_energy_curve(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    damping: float,
    smooth_lambda: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    nfree = len(PHOTON_ENERGY_KNOTS) - 1
    normal = np.zeros((nfree, nfree), dtype=float)
    rhs = np.zeros(nfree, dtype=float)
    grouped: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for ev, mass in zip(train, train_masses):
        e1, e2 = float(ev["e1"]), float(ev["e2"])
        i1, i2 = photon_cell_index(e1), photon_cell_index(e2)
        low_energy, high_energy = sorted((e1, e2))
        grouped[(min(i1, i2), max(i1, i2))].append((float(mass), low_energy, high_energy))

    rows: list[dict[str, object]] = []
    accepted_constraints = 0
    for (cell1, cell2), values in sorted(grouped.items()):
        masses = np.asarray([value[0] for value in values], dtype=float)
        fit = fit_pi0_mass(masses, canonical_fit_config(args, args.min_photon_cell_entries))
        e1 = float(np.mean([value[1] for value in values]))
        e2 = float(np.mean([value[2] for value in values]))
        feature = photon_energy_features(e1)
        for index, weight in photon_energy_features(e2).items():
            feature[index] = feature.get(index, 0.0) + weight
        ok = int(fit["quality_ok"]) == 1 and bool(feature)
        fit_mu_error = float(fit["mu_err_mev"])
        constraint_weight = min((0.25 / fit_mu_error) ** 2, 100.0) if ok and fit_mu_error > 0.0 else 0.0
        rows.append(
            {
                "fold": fold,
                "model": model,
                "row_type": "photon_pair_cell_constraint",
                "cell1": f"{PHOTON_CELL_EDGES[cell1]:.1f}-{PHOTON_CELL_EDGES[cell1 + 1]:.1f}",
                "cell2": f"{PHOTON_CELL_EDGES[cell2]:.1f}-{PHOTON_CELL_EDGES[cell2 + 1]:.1f}",
                "events": len(values),
                "mean_e1_gev": e1,
                "mean_e2_gev": e2,
                "fit_definition": "canonical_gaussian_linear_118_150",
                "canonical_fit_entries": int(fit["window_entries"]),
                "canonical_fit_ok": int(fit["quality_ok"]),
                "canonical_fit_reasons": fit["quality_reasons"],
                "canonical_fit_warnings": fit["quality_warnings"],
                "canonical_mu_residual_mev": fit["mu_residual_mev"],
                "canonical_mu_err_mev": fit["mu_err_mev"],
                "canonical_sigma_mev": fit["sigma_mev"],
                "constraint_used": int(ok),
                "constraint_weight": constraint_weight,
            }
        )
        if not ok:
            continue
        target = 2.0 * math.log(M_PI0_MEV / float(fit["mu_mev"]))
        for i, wi in feature.items():
            rhs[i] += constraint_weight * wi * target
            for j, wj in feature.items():
                normal[i, j] += constraint_weight * wi * wj
        accepted_constraints += 1

    # First-difference penalty, including final free knot to fixed-unity endpoint.
    for index in range(nfree - 1):
        normal[index, index] += smooth_lambda
        normal[index + 1, index + 1] += smooth_lambda
        normal[index, index + 1] -= smooth_lambda
        normal[index + 1, index] -= smooth_lambda
    normal[nfree - 1, nfree - 1] += smooth_lambda
    normal.flat[:: nfree + 1] += 1.0e-8
    coeff = np.linalg.solve(normal, rhs) * damping
    coeff = np.clip(
        coeff,
        math.log1p(-args.max_photon_energy_update_frac),
        math.log1p(args.max_photon_energy_update_frac),
    )
    for index, energy in enumerate(PHOTON_ENERGY_KNOTS):
        log_scale = float(coeff[index]) if index < nfree else 0.0
        rows.append(
            {
                "fold": fold,
                "model": model,
                "row_type": "photon_energy_knot",
                "energy_gev": energy,
                "energy_scale": math.exp(log_scale),
                "log_energy_scale": log_scale,
                "damping": damping,
                "smooth_lambda": smooth_lambda,
                "clip_frac": args.max_photon_energy_update_frac,
                "accepted_constraints": accepted_constraints,
            }
        )
    return coeff, rows


def photon_energy_scale(coeff: np.ndarray, energy: float) -> float:
    log_scale = sum(weight * float(coeff[index]) for index, weight in photon_energy_features(energy).items())
    return math.exp(log_scale)


def apply_photon_energy_curve(mass: float, ev: dict[str, object], coeff: np.ndarray) -> float:
    scale1 = photon_energy_scale(coeff, float(ev["e1"]))
    scale2 = photon_energy_scale(coeff, float(ev["e2"]))
    return float(mass) * math.sqrt(scale1 * scale2)


def full_photon_energy_features(energy: float, knots: np.ndarray) -> dict[int, float]:
    """Linear interpolation weights with both endpoint scales free."""
    if energy <= knots[0]:
        return {0: 1.0}
    if energy >= knots[-1]:
        return {len(knots) - 1: 1.0}
    high = int(np.searchsorted(knots, energy))
    low = high - 1
    fraction = (energy - knots[low]) / (knots[high] - knots[low])
    return {low: 1.0 - fraction, high: fraction}


def full_photon_cell_index(energy: float, cell_edges: np.ndarray) -> int:
    return min(
        max(int(np.searchsorted(cell_edges, energy, side="right") - 1), 0),
        len(cell_edges) - 2,
    )


def solve_direct_photon_curve(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    period: str,
    damping: float,
    smooth_lambda: float,
    low_positive_clip_frac: float,
    low_constraint_weight: float,
    knots: np.ndarray,
    cell_edges: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    """Fit independent per-photon scales directly from Hao-start masses."""
    ncoeff = len(knots)
    normal = np.zeros((ncoeff, ncoeff), dtype=float)
    rhs = np.zeros(ncoeff, dtype=float)
    grouped: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for ev, mass in zip(train, train_masses):
        low_energy, high_energy = sorted((float(ev["e1"]), float(ev["e2"])))
        i1 = full_photon_cell_index(low_energy, cell_edges)
        i2 = full_photon_cell_index(high_energy, cell_edges)
        grouped[(i1, i2)].append((float(mass), low_energy, high_energy))

    rows: list[dict[str, object]] = []
    accepted_constraints = 0
    for (cell1, cell2), values in sorted(grouped.items()):
        masses = np.asarray([value[0] for value in values], dtype=float)
        fit = fit_pi0_mass(masses, canonical_fit_config(args, args.min_photon_cell_entries))
        mean_e1 = float(np.mean([value[1] for value in values]))
        mean_e2 = float(np.mean([value[2] for value in values]))
        feature = full_photon_energy_features(mean_e1, knots)
        for index, weight in full_photon_energy_features(mean_e2, knots).items():
            feature[index] = feature.get(index, 0.0) + weight
        fit_mu_error = float(fit["mu_err_mev"])
        ok = int(fit["quality_ok"]) == 1 and fit_mu_error > 0.0
        constraint_weight = min((0.25 / fit_mu_error) ** 2, 100.0) if ok else 0.0
        if mean_e1 < 0.8:
            constraint_weight *= low_constraint_weight
        rows.append(
            {
                "fold": fold,
                "model": model,
                "period": period,
                "row_type": "direct_photon_pair_cell_constraint",
                "cell1": (
                    f"{cell_edges[cell1]:.1f}-"
                    f"{cell_edges[cell1 + 1]:.1f}"
                ),
                "cell2": (
                    f"{cell_edges[cell2]:.1f}-"
                    f"{cell_edges[cell2 + 1]:.1f}"
                ),
                "events": len(values),
                "mean_e1_gev": mean_e1,
                "mean_e2_gev": mean_e2,
                "fit_definition": "canonical_gaussian_linear_118_150",
                "canonical_fit_entries": int(fit["window_entries"]),
                "canonical_fit_ok": int(fit["quality_ok"]),
                "canonical_fit_reasons": fit["quality_reasons"],
                "canonical_fit_warnings": fit["quality_warnings"],
                "canonical_mu_residual_mev": fit["mu_residual_mev"],
                "canonical_mu_err_mev": fit["mu_err_mev"],
                "canonical_sigma_mev": fit["sigma_mev"],
                "constraint_used": int(ok),
                "constraint_weight": constraint_weight,
                "low_constraint_weight": low_constraint_weight,
            }
        )
        if not ok:
            continue
        target = 2.0 * math.log(M_PI0_MEV / float(fit["mu_mev"]))
        for i, wi in feature.items():
            rhs[i] += constraint_weight * wi * target
            for j, wj in feature.items():
                normal[i, j] += constraint_weight * wi * wj
        accepted_constraints += 1

    for index in range(ncoeff - 1):
        normal[index, index] += smooth_lambda
        normal[index + 1, index + 1] += smooth_lambda
        normal[index, index + 1] -= smooth_lambda
        normal[index + 1, index] -= smooth_lambda
    normal.flat[:: ncoeff + 1] += 1.0e-8
    coeff = np.linalg.solve(normal, rhs) * damping if accepted_constraints else np.zeros(ncoeff)
    lower = math.log1p(-args.max_period_photon_update_frac)
    upper = np.asarray(
        [
            math.log1p(low_positive_clip_frac)
            if energy <= 0.8
            else math.log1p(args.max_period_photon_update_frac)
            for energy in knots
        ],
        dtype=float,
    )
    coeff = np.minimum(np.maximum(coeff, lower), upper)
    for index, energy in enumerate(knots):
        rows.append(
            {
                "fold": fold,
                "model": model,
                "period": period,
                "row_type": "direct_photon_energy_knot",
                "energy_gev": energy,
                "energy_scale": math.exp(float(coeff[index])),
                "log_energy_scale": float(coeff[index]),
                "damping": damping,
                "smooth_lambda": smooth_lambda,
                "clip_frac": args.max_period_photon_update_frac,
                "low_positive_clip_frac": low_positive_clip_frac,
                "low_constraint_weight": low_constraint_weight,
                "accepted_constraints": accepted_constraints,
            }
        )
    return coeff, rows, accepted_constraints


def derive_hao_direct_photon_curves(
    fold: str,
    train: list[dict[str, object]],
    model: str,
    damping: float,
    smooth_lambda: float,
    low_positive_clip_frac: float,
    low_constraint_weight: float,
    dense_low_knots: bool,
    use_periods: bool,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], np.ndarray]:
    knots = DENSE_LOW_PERIOD_PHOTON_KNOTS if dense_low_knots else PERIOD_PHOTON_KNOTS
    cell_edges = (
        DENSE_LOW_PERIOD_PHOTON_CELL_EDGES if dense_low_knots else PERIOD_PHOTON_CELL_EDGES
    )
    train_masses = [float(ev["mass_mev"]) for ev in train]
    global_coeff, rows, _ = solve_direct_photon_curve(
        fold,
        train,
        train_masses,
        model,
        "global",
        damping,
        smooth_lambda,
        low_positive_clip_frac,
        low_constraint_weight,
        knots,
        cell_edges,
        args,
    )
    curves = {"global": global_coeff}
    if not use_periods:
        return curves, rows, knots

    periods = sorted({v.event_period(int(ev["run"]), train, args) for ev in train})
    for period in periods:
        indexes = [
            index
            for index, ev in enumerate(train)
            if v.event_period(int(ev["run"]), train, args) == period
        ]
        period_events = [train[index] for index in indexes]
        period_masses = [train_masses[index] for index in indexes]
        coeff, period_rows, accepted = solve_direct_photon_curve(
            fold,
            period_events,
            period_masses,
            model,
            period,
            damping,
            smooth_lambda,
            low_positive_clip_frac,
            low_constraint_weight,
            knots,
            cell_edges,
            args,
        )
        rows.extend(period_rows)
        curves[period] = coeff if accepted >= 3 else global_coeff
        rows.append(
            {
                "fold": fold,
                "model": model,
                "period": period,
                "row_type": "direct_photon_period_status",
                "accepted_constraints": accepted,
                "used_global_fallback": int(accepted < 3),
            }
        )
    return curves, rows, knots


def full_photon_energy_scale(coeff: np.ndarray, energy: float, knots: np.ndarray) -> float:
    log_scale = sum(
        weight * float(coeff[index])
        for index, weight in full_photon_energy_features(energy, knots).items()
    )
    return math.exp(log_scale)


def apply_hao_direct_photon_curve(
    ev: dict[str, object],
    curves: dict[str, np.ndarray],
    knots: np.ndarray,
    use_periods: bool,
    train: list[dict[str, object]],
    args: argparse.Namespace,
) -> float:
    period = v.event_period(int(ev["run"]), train, args) if use_periods else "global"
    coeff = curves.get(period, curves["global"])
    scale1 = full_photon_energy_scale(coeff, float(ev["e1"]), knots)
    scale2 = full_photon_energy_scale(coeff, float(ev["e2"]), knots)
    return float(ev["mass_mev"]) * math.sqrt(scale1 * scale2)


def fit_eval_rows(fold: str, events: list[dict[str, object]], masses: dict[str, list[float]], args: argparse.Namespace) -> list[dict[str, object]]:
    groups: list[tuple[str, list[int]]] = [("pooled", list(range(len(events))))]
    runs = sorted({int(ev["run"]) for ev in events})
    for run in runs:
        groups.append((f"run_{run}", [i for i, ev in enumerate(events) if int(ev["run"]) == run]))
    for eb in base.ENERGY_BINS:
        label = base.bin_label(eb)
        groups.append((f"emin_{label}", [i for i, ev in enumerate(events) if base.energy_bin(float(ev["min_e"])) == eb]))
        for run in runs:
            groups.append(
                (
                    f"run_{run}_emin_{label}",
                    [i for i, ev in enumerate(events) if int(ev["run"]) == run and base.energy_bin(float(ev["min_e"])) == eb],
                )
            )
    rows: list[dict[str, object]] = []
    for group, idxs in groups:
        if not idxs:
            continue
        for model in model_names():
            vals = np.asarray([masses[model][i] for i in idxs], dtype=float)
            fit = fit_pi0_mass(vals, canonical_fit_config(args, args.min_fit_entries))
            rows.append(
                {
                    "fold": fold,
                    "sample": "test",
                    "group": group,
                    "model": model,
                    "entries": len(vals),
                    "fit_definition": "canonical_gaussian_linear_118_150",
                    "fit_entries": int(fit["window_entries"]),
                    "fit_ok": int(fit["quality_ok"]),
                    "fit_reasons": fit["quality_reasons"],
                    "fit_warnings": fit["quality_warnings"],
                    "fit_mu_mev": fit["mu_mev"],
                    "fit_mu_residual_mev": fit["mu_residual_mev"],
                    "fit_mu_err_mev": fit["mu_err_mev"],
                    "fit_sigma_mev": fit["sigma_mev"],
                    "chi2_ndf": fit["chi2_ndf"],
                    "fit_window_shift_mev": fit["max_window_shift_mev"],
                }
            )
    return rows


def pooled_rows(events: list[dict[str, object]], masses: dict[str, list[float]], args: argparse.Namespace) -> list[dict[str, object]]:
    rows = fit_eval_rows("pooled_folds", events, masses, args)
    for row in rows:
        row["fold"] = "pooled_folds"
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def label_for_event(ev: dict[str, object]) -> str:
    return base.bin_label(base.energy_bin(float(ev["min_e"])))


def true_merged_support_rows(
    events: list[dict[str, object]],
    masses: dict[str, list[float]],
    support_groups: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for support in support_groups:
        runs = {int(x) for x in support["member_runs"].split(",") if x}
        ebin = support["energy_bin"]
        idx = [i for i, ev in enumerate(events) if int(ev["run"]) in runs and label_for_event(ev) == ebin]
        if not idx:
            continue
        for model in model_names():
            vals = np.asarray([masses[model][i] for i in idx], dtype=float)
            fit = fit_pi0_mass(vals, canonical_fit_config(args, args.min_merged_core_entries))
            quality_ok = int(fit["quality_ok"]) == 1
            residual = float(fit["mu_residual_mev"]) if quality_ok else math.nan
            rows.append(
                {
                    "energy_bin": ebin,
                    "target_run": int(support["target_run"]),
                    "model": model,
                    "member_runs": support["member_runs"],
                    "run_min": int(support["run_min"]),
                    "run_max": int(support["run_max"]),
                    "events": int(vals.size),
                    "fit_definition": "canonical_gaussian_linear_118_150",
                    "canonical_fit_entries": int(fit["window_entries"]),
                    "canonical_fit_ok": int(fit["quality_ok"]),
                    "canonical_fit_reasons": fit["quality_reasons"],
                    "canonical_fit_warnings": fit["quality_warnings"],
                    "canonical_mu_residual_mev": fit["mu_residual_mev"],
                    "canonical_mu_err_mev": fit["mu_err_mev"],
                    "canonical_sigma_mev": fit["sigma_mev"],
                    "canonical_chi2_ndf": fit["chi2_ndf"],
                    "canonical_window_shift_mev": fit["max_window_shift_mev"],
                    "best_residual_mev": residual,
                }
            )
    return rows


def true_merged_support_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for energy_bin in sorted({str(r["energy_bin"]) for r in rows}, key=lambda s: float(s.split("-")[0])):
        for model in model_names():
            vals = [
                float(r["best_residual_mev"])
                for r in rows
                if r["energy_bin"] == energy_bin and r["model"] == model and math.isfinite(float(r["best_residual_mev"]))
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out.append(
                {
                    "energy_bin": energy_bin,
                    "model": model,
                    "n_groups": len(vals),
                    "mean_residual_mev": float(np.mean(arr)),
                    "median_abs_residual_mev": float(np.median(np.abs(arr))),
                    "p90_abs_residual_mev": float(np.percentile(np.abs(arr), 90)),
                    "max_abs_residual_mev": float(np.max(np.abs(arr))),
                    "positive_gt1": int(np.sum(arr > 1.0)),
                    "negative_lt_minus1": int(np.sum(arr < -1.0)),
                }
            )
    return out


def stability_summary(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for eb in base.ENERGY_BINS:
        suffix = f"_emin_{base.bin_label(eb)}"
        for model in model_names():
            vals = [
                abs(float(row["fit_mu_residual_mev"]))
                for row in rows
                if row["fold"] == "pooled_folds"
                and row["model"] == model
                and str(row["group"]).startswith("run_")
                and str(row["group"]).endswith(suffix)
                and int(row["fit_ok"]) == 1
                and int(row["fit_entries"]) >= args.min_fit_entries
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out.append(
                {
                    "energy_group": f"emin_{base.bin_label(eb)}",
                    "model": model,
                    "nrun": len(vals),
                    "median_abs_mu_residual_mev": float(np.median(arr)),
                    "p90_abs_mu_residual_mev": float(np.percentile(arr, 90)),
                    "p95_abs_mu_residual_mev": float(np.percentile(arr, 95)),
                    "max_abs_mu_residual_mev": float(np.max(arr)),
                }
            )
    return out


def support_best_residual(row: dict[str, object]) -> float:
    val = float(row["best_residual_mev"])
    return val if math.isfinite(val) else math.nan


def support_replacement_map(support_rows: list[dict[str, object]]) -> dict[tuple[int, str, str], dict[str, object]]:
    out: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in support_rows:
        key = (int(row["target_run"]), f"emin_{row['energy_bin']}", str(row["model"]))
        out[key] = row
    return out


def merged_replaced_run_points(
    rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
    egroup: str,
    model: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    replacements = support_replacement_map(support_rows)
    points: list[dict[str, object]] = []
    used_replacements: set[tuple[int, str, str]] = set()
    for row in rows:
        group = str(row["group"])
        if (
            row["fold"] != "pooled_folds"
            or row["model"] != model
            or not group.startswith("run_")
            or not group.endswith("_" + egroup)
        ):
            continue
        run = int(group.split("_")[1])
        key = (run, egroup, model)
        if key in replacements:
            if key in used_replacements:
                continue
            rep = replacements[key]
            residual = support_best_residual(rep)
            if math.isfinite(residual):
                points.append(
                    {
                        "run": run,
                        "residual": residual,
                        "source": "merged",
                        "run_min": int(rep["run_min"]),
                        "run_max": int(rep["run_max"]),
                    }
                )
                used_replacements.add(key)
            continue
        if int(row["fit_ok"]) == 1 and int(row["fit_entries"]) >= args.min_fit_entries:
            residual = float(row["fit_mu_residual_mev"])
            if math.isfinite(residual):
                points.append(
                    {
                        "run": run,
                        "residual": residual,
                        "source": "individual",
                        "run_min": run,
                        "run_max": run,
                    }
                )
    points.sort(key=lambda p: int(p["run"]))
    return points


def merged_replaced_stability_summary(
    rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for eb in base.ENERGY_BINS:
        egroup = f"emin_{base.bin_label(eb)}"
        for model in model_names():
            points = merged_replaced_run_points(rows, support_rows, egroup, model, args)
            vals = [abs(float(p["residual"])) for p in points]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out.append(
                {
                    "energy_group": egroup,
                    "model": model,
                    "nrun_or_replacement": len(vals),
                    "n_merged_replacements": sum(1 for p in points if p["source"] == "merged"),
                    "median_abs_mu_residual_mev": float(np.median(arr)),
                    "p90_abs_mu_residual_mev": float(np.percentile(arr, 90)),
                    "p95_abs_mu_residual_mev": float(np.percentile(arr, 95)),
                    "max_abs_mu_residual_mev": float(np.max(arr)),
                }
            )
    return out


def plot_pdf(
    rows: list[dict[str, object]],
    stability: list[dict[str, object]],
    outdir: Path,
    args: argparse.Namespace,
    support_rows: list[dict[str, object]] | None = None,
) -> None:
    support_rows = support_rows or []
    pdf = outdir / "trusted_merged_energy_layer_scan.pdf"
    colors = {
        "hao_baseline": "#000000",
        "full_iter2": "#444444",
        "fitBand_stat12_d075": "#B279A2",
        "canonical_runlocal_merged_d025": "#59A14F",
        "fixed4_global06_08_d050": "#EDC948",
        "fixed4_global06_08_d075": "#E15759",
        "fixed4_global06_08_d100": "#9C755F",
        "fixed4_smooth06_12_d075": "#76B7B2",
        "fixed4_smooth06_12_d100": "#E15759",
        "fixed4_photoncurve_lam1_d100": "#4E79A7",
        "fixed4_photoncurve_lam10_d075": "#F28E2B",
        "fixed4_photoncurve_lam10_d100": "#E15759",
        "fixed4_photoncurve_lam100_d100": "#76B7B2",
        "hao_global_photon_l10_d075": "#9C755F",
        "hao_period_photon_l10_d075": "#E15759",
        "hao_period_photon_l10_d100": "#4E79A7",
        "hao_period_photon_l10_d100_low045": "#59A14F",
        "hao_period_photon_l10_d100_low060": "#F28E2B",
        "hao_period_photon_l30_d100_low060": "#B279A2",
        "hao_period_photon_l10_d100_low045_k07": "#59A14F",
        "hao_period_photon_l10_d100_low045_k07_w150": "#F28E2B",
        "hao_period_photon_l10_d100_low045_k07_w200": "#B279A2",
        "trusted_fixed4_lowE_d015": "#A0CBE8",
        "trusted_fixed4_lowE_d025": "#4C78A8",
        "trusted_fixed4_lowE_d035": "#4C78A8",
        "trusted_fixed4_lowE_d050": "#72B7B2",
        "trusted_stat12_lowE_d015": "#FFBE7D",
        "trusted_stat12_lowE_d025": "#F58518",
        "trusted_stat12_lowE_d035": "#F58518",
        "trusted_stat12_lowE_d050": "#E45756",
    }
    labels = {
        "hao_baseline": "Hao baseline",
        "full_iter2": "full iter2",
        "fitBand_stat12_d075": "old stat12",
        "canonical_runlocal_merged_d025": "canonical run-local merged d0.25",
        "fixed4_global06_08_d050": "fixed4 + global 0.6-0.8 d0.50",
        "fixed4_global06_08_d075": "fixed4 + global 0.6-0.8 d0.75",
        "fixed4_global06_08_d100": "fixed4 + global 0.6-0.8 d1.00",
        "fixed4_smooth06_12_d075": "fixed4 + smooth 0.6-1.2 d0.75",
        "fixed4_smooth06_12_d100": "fixed4 + smooth 0.6-1.2 d1.00",
        "fixed4_photoncurve_lam1_d100": "photon curve l1 d1.00",
        "fixed4_photoncurve_lam10_d075": "photon curve l10 d0.75",
        "fixed4_photoncurve_lam10_d100": "photon curve l10 d1.00",
        "fixed4_photoncurve_lam100_d100": "photon curve l100 d1.00",
        "hao_global_photon_l10_d075": "Hao + global photon curve d0.75",
        "hao_period_photon_l10_d075": "Hao + period photon curve d0.75",
        "hao_period_photon_l10_d100": "Hao + period photon curve d1.00",
        "hao_period_photon_l10_d100_low045": "period photon +4.5% low cap",
        "hao_period_photon_l10_d100_low060": "period photon +6% low cap",
        "hao_period_photon_l30_d100_low060": "period photon +6% low cap, l30",
        "hao_period_photon_l10_d100_low045_k07": "period photon + 0.7 knot",
        "hao_period_photon_l10_d100_low045_k07_w150": "0.7 knot + low weight 1.5",
        "hao_period_photon_l10_d100_low045_k07_w200": "0.7 knot + low weight 2.0",
        "trusted_fixed4_lowE_d015": "trusted fixed4 d0.15",
        "trusted_fixed4_lowE_d025": "trusted fixed4 d0.25",
        "trusted_fixed4_lowE_d035": "trusted fixed4 d0.35",
        "trusted_fixed4_lowE_d050": "trusted fixed4 d0.50",
        "trusted_stat12_lowE_d015": "trusted stat12 d0.15",
        "trusted_stat12_lowE_d025": "trusted stat12 d0.25",
        "trusted_stat12_lowE_d035": "trusted stat12 d0.35",
        "trusted_stat12_lowE_d050": "trusted stat12 d0.50",
    }
    with PdfPages(pdf) as pp:
        pooled = {r["model"]: r for r in rows if r["fold"] == "pooled_folds" and r["group"] == "pooled"}
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        models = model_names()
        x = np.arange(len(models))
        axes[0].bar(x, [float(pooled[m]["fit_mu_residual_mev"]) for m in models], color=[colors[m] for m in models])
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_ylabel("pooled fit mean residual (MeV)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([labels[m] for m in models], rotation=25, ha="right")
        axes[1].bar(x, [float(pooled[m]["fit_sigma_mev"]) for m in models], color=[colors[m] for m in models])
        axes[1].set_ylabel("pooled fit sigma (MeV)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([labels[m] for m in models], rotation=25, ha="right")
        pp.savefig(fig)
        plt.close(fig)

        for egroup in ["emin_0.6-0.8", "emin_0.8-1.0", "emin_1.0-1.2", "emin_1.2-1.5"]:
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            for model in models:
                points = merged_replaced_run_points(rows, support_rows, egroup, model, args)
                if not points:
                    continue
                segments: list[list[dict[str, object]]] = []
                for point in points:
                    if not segments or int(point["run"]) - int(segments[-1][-1]["run"]) > args.support_max_adjacent_gap:
                        segments.append([])
                    segments[-1].append(point)
                for i, segment in enumerate(segments):
                    ax.plot(
                        [int(p["run"]) for p in segment],
                        [float(p["residual"]) for p in segment],
                        lw=1,
                        color=colors[model],
                        label=labels[model] if i == 0 else None,
                    )
                individual = [p for p in points if p["source"] == "individual"]
                merged = [p for p in points if p["source"] == "merged"]
                if individual:
                    ax.scatter(
                        [int(p["run"]) for p in individual],
                        [float(p["residual"]) for p in individual],
                        marker="o",
                        s=12,
                        color=colors[model],
                        zorder=3,
                    )
                if merged:
                    ax.scatter(
                        [int(p["run"]) for p in merged],
                        [float(p["residual"]) for p in merged],
                        marker="D",
                        s=28,
                        color=colors[model],
                        edgecolor="black",
                        linewidth=0.35,
                        zorder=4,
                    )
            ax.axhline(0, color="black", lw=1)
            ax.axhline(1, color="gray", lw=0.8, ls="--")
            ax.axhline(-1, color="gray", lw=0.8, ls="--")
            ax.set_title(f"{egroup} fitted peak residual vs run (diamonds = merged support)")
            ax.set_xlabel("run")
            ax.set_ylabel("fit mean residual (MeV)")
            ax.legend(fontsize=7, ncol=2)
            pp.savefig(fig)
            plt.close(fig)

        for egroup in ["emin_0.6-0.8", "emin_0.8-1.0", "emin_1.0-1.2"]:
            sub = [r for r in stability if r["energy_group"] == egroup]
            by = {r["model"]: r for r in sub}
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            w = 0.25
            x = np.arange(len(models))
            ax.bar(x - w, [float(by[m]["median_abs_mu_residual_mev"]) for m in models], w, label="median")
            ax.bar(x, [float(by[m]["p90_abs_mu_residual_mev"]) for m in models], w, label="p90")
            ax.bar(x + w, [float(by[m]["max_abs_mu_residual_mev"]) for m in models], w, label="max")
            ax.set_title(f"{egroup} run residual stability")
            ax.set_ylabel("|fit mean residual| (MeV)")
            ax.set_xticks(x)
            ax.set_xticklabels([labels[m] for m in models], rotation=25, ha="right")
            ax.legend()
            pp.savefig(fig)
            plt.close(fig)


def main() -> int:
    global ACTIVE_MODELS
    args = parse_args()
    if args.candidate_model:
        requested = set(args.candidate_model)
        known = {str(spec["model"]) for spec in candidate_specs()}
        unknown = sorted(requested - known)
        if unknown:
            raise SystemExit(f"unknown --candidate-model values: {unknown}")
        ACTIVE_MODELS = requested
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = base.load_conversion_summary(Path(args.conversion_summary))
    events, qa = pairspace.read_events(summary_rows, args)
    write_tsv(outdir / "sidecar_quality.tsv", qa)

    all_test_events: list[dict[str, object]] = []
    all_masses: dict[str, list[float]] = {m: [] for m in model_names()}
    fit_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []

    for fold_value in [0, 1]:
        fold = f"segment_parity_test_{fold_value}"
        train = [ev for ev in events if int(ev["fold"]) != fold_value]
        test = [ev for ev in events if int(ev["fold"]) == fold_value]
        needs_full_iter = any(
            str(spec["mode"]) not in {"hao_identity", "hao_global_photon_curve", "hao_period_photon_curve"}
            for spec in active_candidate_specs()
        )
        if needs_full_iter:
            train_base, test_base, full_rows = pairspace.derive_full_iter2(train, test, args)
            update_rows.extend({"fold": fold, "model": "full_iter2", **row} for row in full_rows)
        else:
            train_base = [float(ev["mass_mev"]) for ev in train]
            test_base = [float(ev["mass_mev"]) for ev in test]

        masses: dict[str, list[float]] = {}
        if "full_iter2" in model_names():
            masses["full_iter2"] = test_base[:]
        for spec in active_candidate_specs():
            model = str(spec["model"])
            mode = str(spec["mode"])
            if model == "full_iter2":
                continue
            if mode == "hao_identity":
                masses[model] = [float(ev["mass_mev"]) for ev in test]
                continue
            if mode in {"hao_global_photon_curve", "hao_period_photon_curve"}:
                use_periods = mode == "hao_period_photon_curve"
                curves, rows, knots = derive_hao_direct_photon_curves(
                    fold,
                    train,
                    model,
                    float(spec["damping"]),
                    float(spec["smooth_lambda"]),
                    float(spec.get("low_positive_clip_frac", args.max_period_photon_update_frac)),
                    float(spec.get("low_constraint_weight", 1.0)),
                    bool(spec.get("dense_low_knots", False)),
                    use_periods,
                    args,
                )
                update_rows.extend(rows)
                masses[model] = [
                    apply_hao_direct_photon_curve(ev, curves, knots, use_periods, train, args)
                    for ev in test
                ]
                continue
            if mode == "reference_fitband":
                updates, band_map, rows = fitband.derive_fitband_updates(fold, train, train_base, model, "stat12", 0.75, args)
                update_rows.extend(rows)
                masses[model] = [fitband.apply_fitband(mass, ev, updates, band_map) for ev, mass in zip(test, test_base)]
                continue
            if mode == "canonical_runlocal_merged":
                updates, rows = derive_canonical_runlocal_merged_updates(fold, train, train_base, model, float(spec["damping"]), args)
                update_rows.extend(rows)
                masses[model] = [
                    apply_canonical_runlocal_merged(mass, ev, updates) for ev, mass in zip(test, test_base)
                ]
                continue
            if mode == "fixed4_global_lowe":
                fixed_updates, fixed_band_map, fixed_rows = derive_trusted_updates(
                    fold, train, train_base, model, "fixed4", 0.25, args
                )
                update_rows.extend(fixed_rows)
                train_fixed = [
                    apply_trusted(mass, ev, fixed_updates, fixed_band_map) for ev, mass in zip(train, train_base)
                ]
                test_fixed = [
                    apply_trusted(mass, ev, fixed_updates, fixed_band_map) for ev, mass in zip(test, test_base)
                ]
                global_factor, global_row = derive_global_lowe_anchor(
                    fold, train, train_fixed, model, float(spec["damping"]), args
                )
                update_rows.append(global_row)
                masses[model] = [
                    apply_global_lowe_anchor(mass, ev, global_factor) for ev, mass in zip(test, test_fixed)
                ]
                continue
            if mode == "fixed4_smooth_lowe":
                fixed_updates, fixed_band_map, fixed_rows = derive_trusted_updates(
                    fold, train, train_base, model, "fixed4", 0.25, args
                )
                update_rows.extend(fixed_rows)
                train_fixed = [
                    apply_trusted(mass, ev, fixed_updates, fixed_band_map) for ev, mass in zip(train, train_base)
                ]
                test_fixed = [
                    apply_trusted(mass, ev, fixed_updates, fixed_band_map) for ev, mass in zip(test, test_base)
                ]
                smooth_factors, smooth_rows = derive_smooth_lowe_anchors(
                    fold, train, train_fixed, model, float(spec["damping"]), args
                )
                update_rows.extend(smooth_rows)
                masses[model] = [
                    apply_smooth_lowe_anchors(
                        mass, ev, smooth_factors, args.global_anchor_transition_halfwidth
                    )
                    for ev, mass in zip(test, test_fixed)
                ]
                continue
            if mode == "fixed4_photon_curve":
                fixed_updates, fixed_band_map, fixed_rows = derive_trusted_updates(
                    fold, train, train_base, model, "fixed4", 0.25, args
                )
                update_rows.extend(fixed_rows)
                train_fixed = [
                    apply_trusted(mass, ev, fixed_updates, fixed_band_map) for ev, mass in zip(train, train_base)
                ]
                test_fixed = [
                    apply_trusted(mass, ev, fixed_updates, fixed_band_map) for ev, mass in zip(test, test_base)
                ]
                photon_coeff, photon_rows = derive_photon_energy_curve(
                    fold,
                    train,
                    train_fixed,
                    model,
                    float(spec["damping"]),
                    float(spec["smooth_lambda"]),
                    args,
                )
                update_rows.extend(photon_rows)
                masses[model] = [
                    apply_photon_energy_curve(mass, ev, photon_coeff) for ev, mass in zip(test, test_fixed)
                ]
                continue
            updates, band_map, rows = derive_trusted_updates(fold, train, train_base, model, mode, float(spec["damping"]), args)
            update_rows.extend(rows)
            masses[model] = [apply_trusted(mass, ev, updates, band_map) for ev, mass in zip(test, test_base)]

        fit_rows.extend(fit_eval_rows(fold, test, masses, args))
        all_test_events.extend(test)
        for model in model_names():
            all_masses[model].extend(masses[model])

    fit_rows.extend(pooled_rows(all_test_events, all_masses, args))
    stability = stability_summary(fit_rows, args)
    plot_stability = stability
    support_rows: list[dict[str, object]] = []
    if args.support_groups:
        support_groups = read_tsv(Path(args.support_groups))
        support_rows = true_merged_support_rows(all_test_events, all_masses, support_groups, args)
        plot_stability = merged_replaced_stability_summary(fit_rows, support_rows, args)
        write_tsv(outdir / "true_merged_support_fit_table.tsv", support_rows)
        write_tsv(outdir / "true_merged_support_summary.tsv", true_merged_support_summary(support_rows))
        write_tsv(outdir / "merged_replaced_run_residual_stability_summary.tsv", plot_stability)
    write_tsv(outdir / "trusted_merged_update_table.tsv", update_rows)
    write_tsv(outdir / "heldout_fit_metrics.tsv", fit_rows)
    write_tsv(outdir / "run_residual_stability_summary.tsv", stability)
    plot_pdf(fit_rows, plot_stability, outdir, args, support_rows)
    print(outdir)
    print(outdir / "trusted_merged_energy_layer_scan.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
