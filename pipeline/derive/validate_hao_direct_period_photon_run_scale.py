#!/usr/bin/env python3
"""Validate a Gaussian-fit run scale after the direct Hao-start photon curve."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

import validate_hao_full_iter2_pairspace as pairspace
import validate_hao_start_layered_member_stack as base
import validate_hao_trusted_merged_energy_layer as direct
from pi0_mass_fit import DEFAULT_CONFIG, FitConfig, M_PI0_MEV, evaluate_model, fit_pi0_mass
from validate_hao_direct_period_photon_production import (
    configure_run_period_lut,
    period_for_run,
    read_production_pairs,
)


DIRECT_MODEL = "hao_period_photon_l10_d100_low045_k07_w150"
RUN_DAMPINGS = (0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--production-pair-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--runs", default="4300,4417,4503")
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--min-fit-entries", type=int, default=500)
    ap.add_argument("--min-photon-cell-entries", type=int, default=500)
    ap.add_argument("--fit-max-mu-err", type=float, default=0.50)
    ap.add_argument("--fit-max-chi2-ndf", type=float, default=math.inf)
    ap.add_argument("--fit-warn-chi2-ndf", type=float, default=3.0)
    ap.add_argument("--fit-max-window-shift", type=float, default=0.50)
    ap.add_argument("--max-run-update-frac", type=float, default=0.02)
    ap.add_argument("--support-max-adjacent-gap", type=int, default=5)
    ap.add_argument("--support-max-run-distance", type=int, default=8)
    ap.add_argument("--support-max-members", type=int, default=9)
    ap.add_argument("--edge-support-policy", action="store_true")
    ap.add_argument("--run-own-max-mu-err", type=float, default=0.20)
    ap.add_argument("--support-shrink-tau-mev", type=float, default=0.25)
    ap.add_argument("--one-sided-support-weight", type=float, default=0.50)
    ap.add_argument("--excluded-calibration-runs", default="")
    ap.add_argument("--reuse-existing-curves", action="store_true")
    ap.add_argument("--curve-cache-dir", default="")
    ap.add_argument("--run-period-lut", type=Path)
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


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    mapping = configure_run_period_lut(getattr(args, "run_period_lut", None))
    return SimpleNamespace(
        timing_center=args.timing_center,
        timing_window=args.timing_window,
        emin_range=args.emin_range,
        period_mode="explicit" if mapping else "hao_lh2_cycles",
        _run_period_map=mapping,
        n_run_bands=3,
        min_photon_cell_entries=args.min_photon_cell_entries,
        max_period_photon_update_frac=0.03,
        fit_max_mu_err=args.fit_max_mu_err,
        fit_max_chi2_ndf=args.fit_max_chi2_ndf,
        fit_warn_chi2_ndf=args.fit_warn_chi2_ndf,
        fit_max_window_shift=args.fit_max_window_shift,
    )


def fit_config(args: argparse.Namespace, min_entries: int | None = None) -> FitConfig:
    return FitConfig(
        **{
            **DEFAULT_CONFIG.__dict__,
            "min_entries": args.min_fit_entries if min_entries is None else min_entries,
            "max_mu_err": args.fit_max_mu_err,
            "max_chi2_ndf": args.fit_max_chi2_ndf,
            "warn_chi2_ndf": args.fit_warn_chi2_ndf,
            "max_window_shift": args.fit_max_window_shift,
        }
    )


def clamp(value: float, fraction: float) -> float:
    return min(1.0 + fraction, max(1.0 - fraction, value))


def parse_run_set(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def derive_direct_curves(fold: str, train: list[dict[str, object]], args: argparse.Namespace, margs: SimpleNamespace):
    return direct.derive_hao_direct_photon_curves(
        fold,
        train,
        DIRECT_MODEL,
        1.0,
        10.0,
        0.045,
        1.5,
        True,
        True,
        margs,
    )


def apply_direct(events: list[dict[str, object]], curves, knots, train, margs) -> list[float]:
    return [direct.apply_hao_direct_photon_curve(event, curves, knots, True, train, margs) for event in events]


def derive_run_scales(
    fold: str,
    events: list[dict[str, object]],
    masses: list[float],
    target_runs: set[int],
    damping: float,
    args: argparse.Namespace,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    excluded_runs = parse_run_set(args.excluded_calibration_runs)
    for event, mass in zip(events, masses):
        run = int(event["run"])
        if run not in excluded_runs:
            grouped[run].append(float(mass))
    individual_fits = {
        run: fit_pi0_mass(np.asarray(values, dtype=float), fit_config(args))
        for run, values in grouped.items()
    }
    period_values: dict[str, list[float]] = defaultdict(list)
    for run, values in grouped.items():
        period_values[period_for_run(run)].extend(values)
    period_fits = {
        period: fit_pi0_mass(np.asarray(values, dtype=float), fit_config(args))
        for period, values in period_values.items()
    }

    scales: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for run in sorted(target_runs):
        own_values = grouped.get(run, [])
        fit = individual_fits.get(run)
        support_runs = [run] if own_values else []
        support_values = list(own_values)
        support_mode = "run_local"
        accepted = fit is not None and int(fit["quality_ok"]) == 1
        if accepted and args.edge_support_policy:
            accepted = (
                math.isfinite(float(fit["mu_err_mev"]))
                and float(fit["mu_err_mev"]) <= args.run_own_max_mu_err
            )
        support_two_sided = False
        support_shrink_alpha = 1.0
        group_full_scale = math.nan
        period_fit = period_fits.get(period_for_run(run))
        period_accepted = period_fit is not None and int(period_fit["quality_ok"]) == 1
        period_full_scale = (
            M_PI0_MEV / float(period_fit["mu_mev"]) if period_accepted else math.nan
        )

        if not accepted:
            same_period_runs = sorted(
                candidate for candidate in grouped if period_for_run(candidate) == period_for_run(run)
            )
            connected = sorted(set(same_period_runs + [run]))
            target_index = connected.index(run)
            lo = target_index
            hi = target_index
            while lo > 0 and connected[lo] - connected[lo - 1] <= args.support_max_adjacent_gap:
                lo -= 1
            while hi + 1 < len(connected) and connected[hi + 1] - connected[hi] <= args.support_max_adjacent_gap:
                hi += 1
            candidates = [
                candidate
                for candidate in connected[lo : hi + 1]
                if candidate != run and abs(candidate - run) <= args.support_max_run_distance
            ]
            candidates.sort(key=lambda candidate: (abs(candidate - run), candidate))

            if args.edge_support_policy:
                # Prefer a bracketing pair before accepting one-sided support.
                lower = [candidate for candidate in candidates if candidate < run]
                upper = [candidate for candidate in candidates if candidate > run]
                initial: list[int] = []
                if lower:
                    initial.append(max(lower))
                if upper:
                    initial.append(min(upper))
                if not initial and candidates:
                    initial.append(candidates[0])

                added: set[int] = set()
                for candidate in initial:
                    if len(support_runs) >= args.support_max_members:
                        break
                    support_runs.append(candidate)
                    support_values.extend(grouped[candidate])
                    added.add(candidate)

                if support_values:
                    fit = fit_pi0_mass(np.asarray(support_values, dtype=float), fit_config(args))
                    if int(fit["quality_ok"]) == 1:
                        accepted = True
                for candidate in candidates:
                    if accepted:
                        break
                    if candidate in added:
                        continue
                    if len(support_runs) >= args.support_max_members:
                        break
                    support_runs.append(candidate)
                    support_values.extend(grouped[candidate])
                    fit = fit_pi0_mass(np.asarray(support_values, dtype=float), fit_config(args))
                    if int(fit["quality_ok"]) == 1:
                        accepted = True
                        break
                if accepted:
                    has_lower = any(member < run for member in support_runs)
                    has_upper = any(member > run for member in support_runs)
                    support_two_sided = has_lower and has_upper
                    support_mode = (
                        "chronological_group_two_sided"
                        if support_two_sided
                        else "chronological_group_one_sided_shrunk"
                    )
            else:
                for candidate in candidates:
                    if len(support_runs) >= args.support_max_members:
                        break
                    support_runs.append(candidate)
                    support_values.extend(grouped[candidate])
                    fit = fit_pi0_mass(np.asarray(support_values, dtype=float), fit_config(args))
                    if int(fit["quality_ok"]) == 1:
                        accepted = True
                        support_mode = "chronological_group"
                        break

        if not accepted:
            fit = period_fit
            if period_accepted:
                accepted = True
                support_mode = "period_fallback"
                support_runs = sorted(
                    candidate for candidate in grouped if period_for_run(candidate) == period_for_run(run)
                )
                support_values = period_values[period_for_run(run)]
            else:
                fit = fit_pi0_mass(np.asarray([], dtype=float), fit_config(args))
                support_mode = "identity_fallback"

        full_scale = M_PI0_MEV / float(fit["mu_mev"]) if accepted else 1.0
        if support_mode.startswith("chronological_group"):
            group_full_scale = full_scale
        if (
            args.edge_support_policy
            and support_mode == "chronological_group_one_sided_shrunk"
            and period_accepted
        ):
            mu_err = float(fit["mu_err_mev"])
            tau2 = args.support_shrink_tau_mev**2
            precision_weight = tau2 / (tau2 + mu_err**2)
            support_shrink_alpha = args.one_sided_support_weight * precision_weight
            full_scale = math.exp(
                support_shrink_alpha * math.log(group_full_scale)
                + (1.0 - support_shrink_alpha) * math.log(period_full_scale)
            )
        applied = clamp(full_scale**damping, args.max_run_update_frac) if accepted else 1.0
        scales[run] = applied
        rows.append(
            {
                "fold": fold,
                "run": run,
                "calibration_excluded": int(run in excluded_runs),
                "damping": damping,
                "events": len(own_values),
                "support_mode": support_mode,
                "support_runs": ",".join(str(member) for member in support_runs),
                "support_run_count": len(support_runs),
                "support_events": len(support_values),
                "support_two_sided": int(support_two_sided),
                "support_shrink_alpha": support_shrink_alpha,
                "fit_entries": fit["window_entries"],
                "fit_ok": fit["quality_ok"],
                "fit_reasons": fit["quality_reasons"],
                "fit_mu_mev": fit["mu_mev"],
                "fit_mu_residual_mev": fit["mu_residual_mev"],
                "fit_mu_err_mev": fit["mu_err_mev"],
                "fit_sigma_mev": fit["sigma_mev"],
                "group_full_energy_scale": group_full_scale,
                "period_full_energy_scale": period_full_scale,
                "full_energy_scale": full_scale,
                "applied_energy_scale": applied,
                "clipped": int(accepted and not math.isclose(applied, full_scale**damping, rel_tol=0, abs_tol=1e-12)),
            }
        )
    return scales, rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def curves_from_derivation_rows(rows: list[dict[str, str]], fold: str):
    selected = [
        row
        for row in rows
        if row.get("fold") == fold and row.get("row_type") == "direct_photon_energy_knot"
    ]
    if not selected:
        raise ValueError(f"no saved direct photon knots for {fold}")
    knots = np.asarray(sorted({float(row["energy_gev"]) for row in selected}), dtype=float)
    curves = {}
    for period in sorted({row["period"] for row in selected}):
        by_energy = {float(row["energy_gev"]): float(row["log_energy_scale"]) for row in selected if row["period"] == period}
        curves[period] = np.asarray([by_energy[float(energy)] for energy in knots], dtype=float)
    return curves, knots


def curves_from_package_rows(rows: list[dict[str, str]]):
    knots = np.asarray(sorted({float(row["energy_gev"]) for row in rows}), dtype=float)
    curves = {}
    for period in sorted({row["period"] for row in rows}):
        by_energy = {float(row["energy_gev"]): float(row["log_energy_scale"]) for row in rows if row["period"] == period}
        curves[period] = np.asarray([by_energy[float(energy)] for energy in knots], dtype=float)
    return curves, knots


def apply_run_scales(events: list[dict[str, object]], masses: list[float], scales: dict[int, float]) -> list[float]:
    return [float(mass) * scales.get(int(event["run"]), 1.0) for event, mass in zip(events, masses)]


def redamp_run_scale_rows(
    source_rows: list[dict[str, object]], damping: float, args: argparse.Namespace
) -> tuple[dict[int, float], list[dict[str, object]]]:
    scales: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for source in source_rows:
        row = dict(source)
        full_scale = float(row["full_energy_scale"])
        accepted = int(row["fit_ok"]) == 1
        applied = clamp(full_scale**damping, args.max_run_update_frac) if accepted else 1.0
        row["damping"] = damping
        row["applied_energy_scale"] = applied
        row["clipped"] = int(
            accepted and not math.isclose(applied, full_scale**damping, rel_tol=0, abs_tol=1e-12)
        )
        scales[int(row["run"])] = applied
        rows.append(row)
    return scales, rows


def group_indexes(events: list[dict[str, object]], include_run_energy: bool = False) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = [("pooled", list(range(len(events))))]
    runs = sorted({int(event["run"]) for event in events})
    for run in runs:
        groups.append((f"run_{run}", [i for i, event in enumerate(events) if int(event["run"]) == run]))
    for energy_bin in base.ENERGY_BINS:
        label = base.bin_label(energy_bin)
        indexes = [i for i, event in enumerate(events) if base.energy_bin(float(event["min_e"])) == energy_bin]
        groups.append((f"emin_{label}", indexes))
        if include_run_energy:
            for run in runs:
                groups.append((f"run_{run}_emin_{label}", [i for i in indexes if int(events[i]["run"]) == run]))
    return groups


def evaluate(fold: str, events: list[dict[str, object]], masses: dict[str, list[float]], args: argparse.Namespace, include_run_energy: bool = False) -> list[dict[str, object]]:
    rows = []
    for group, indexes in group_indexes(events, include_run_energy):
        if not indexes:
            continue
        for model, all_values in masses.items():
            values = np.asarray([all_values[i] for i in indexes], dtype=float)
            fit = fit_pi0_mass(values, fit_config(args))
            rows.append(
                {
                    "fold": fold,
                    "group": group,
                    "model": model,
                    "entries": len(values),
                    "fit_entries": fit["window_entries"],
                    "fit_ok": fit["quality_ok"],
                    "fit_reasons": fit["quality_reasons"],
                    "fit_warnings": fit["quality_warnings"],
                    "fit_mu_mev": fit["mu_mev"],
                    "fit_mu_residual_mev": fit["mu_residual_mev"],
                    "fit_mu_err_mev": fit["mu_err_mev"],
                    "fit_sigma_mev": fit["sigma_mev"],
                    "fit_sigma_err_mev": fit["sigma_err_mev"],
                    "chi2_ndf": fit["chi2_ndf"],
                    "fit_window_shift_mev": fit["max_window_shift_mev"],
                }
            )
    return rows


def stability_summary(
    rows: list[dict[str, object]], models: list[str], summary_fold: str = "pooled_folds"
) -> list[dict[str, object]]:
    out = []
    pooled_rows = [row for row in rows if row["fold"] == summary_fold]
    for model in models:
        run_rows = [
            row
            for row in pooled_rows
            if row["model"] == model
            and str(row["group"]).startswith("run_")
            and "_emin_" not in str(row["group"])
            and int(row["fit_ok"]) == 1
        ]
        residuals = np.asarray([abs(float(row["fit_mu_residual_mev"])) for row in run_rows], dtype=float)
        pooled = next(row for row in pooled_rows if row["model"] == model and row["group"] == "pooled")
        low = next(row for row in pooled_rows if row["model"] == model and row["group"] == "emin_0.6-0.8")
        out.append(
            {
                "model": model,
                "fit_valid_runs": len(run_rows),
                "median_abs_run_residual_mev": float(np.median(residuals)),
                "p90_abs_run_residual_mev": float(np.percentile(residuals, 90)),
                "p95_abs_run_residual_mev": float(np.percentile(residuals, 95)),
                "max_abs_run_residual_mev": float(np.max(residuals)),
                "pooled_fit_residual_mev": pooled["fit_mu_residual_mev"],
                "pooled_fit_sigma_mev": pooled["fit_sigma_mev"],
                "lowe_fit_residual_mev": low["fit_mu_residual_mev"],
                "lowe_fit_sigma_mev": low["fit_sigma_mev"],
            }
        )
    return out


def choose_model(summary: list[dict[str, object]]) -> str:
    candidates = [row for row in summary if str(row["model"]).startswith("direct_run_d")]
    return str(
        min(
            candidates,
            key=lambda row: (
                float(row["p95_abs_run_residual_mev"]),
                float(row["p90_abs_run_residual_mev"]),
                float(row["median_abs_run_residual_mev"]),
                float(row["pooled_fit_sigma_mev"]),
            ),
        )["model"]
    )


def support_summary(
    update_rows: list[dict[str, object]], chosen: str, excluded_runs: set[int] | None = None
) -> list[dict[str, object]]:
    excluded_runs = excluded_runs or set()
    damping = float(chosen.rsplit("d", 1)[1])
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in update_rows:
        if math.isclose(float(row["damping"]), damping, abs_tol=1e-12):
            grouped[int(row["run"])].append(row)
    rows: list[dict[str, object]] = []
    for run, items in sorted(grouped.items()):
        modes = [str(item["support_mode"]) for item in items]
        if run in excluded_runs:
            status = "excluded_qa"
        elif any(int(item["events"]) == 0 for item in items):
            status = "parity_incomplete"
        elif any(mode.startswith("chronological_group") for mode in modes):
            status = "grouped"
        else:
            status = "run_local"
        rows.append(
            {
                "run": run,
                "status": status,
                "fold_count": len(items),
                "min_own_events": min(int(item["events"]) for item in items),
                "support_modes": ";".join(modes),
                "support_runs": ";".join(str(item["support_runs"]) for item in items),
            }
        )
    return rows


def plot_holdout(
    path: Path, events, masses, fit_rows, update_rows, models, chosen, excluded_runs, args
) -> None:
    labels = {"hao_baseline": "Hao baseline", "direct_w150": "direct photon w=1.5"}
    labels.update({f"direct_run_d{d:.2f}": f"direct + run scale d={d:.2f}" for d in RUN_DAMPINGS})
    colors = {model: plt.cm.viridis(i / max(len(models) - 1, 1)) for i, model in enumerate(models)}
    metric = {(str(row["group"]), str(row["model"])): row for row in fit_rows if row["fold"] == "pooled_folds"}
    with PdfPages(path) as pdf:
        for group in ["pooled", "emin_0.6-0.8", "emin_0.8-1.0", "emin_1.0-1.2"]:
            indexes = dict(group_indexes(events))[group]
            fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
            bins = np.arange(105.0, 165.01, 1.0)
            centers = 0.5 * (bins[:-1] + bins[1:])
            for model in ["direct_w150", chosen]:
                values = np.asarray([masses[model][i] for i in indexes], dtype=float)
                fit = fit_pi0_mass(values, fit_config(args))
                ax.hist(values, bins=bins, histtype="step", lw=1.8, color=colors[model], label=labels[model])
                if int(fit["optimizer_ok"]):
                    ax.plot(centers, evaluate_model(centers, fit), ls="--", lw=1.4, color=colors[model])
                row = metric[(group, model)]
                ax.plot([], [], color=colors[model], label=f"mu={float(row['fit_mu_mev']):.3f}, sigma={float(row['fit_sigma_mev']):.3f} MeV")
            ax.axvline(M_PI0_MEV, color="black", ls=":", lw=1)
            ax.set_title(f"Held-out invariant mass: {group}")
            ax.set_xlabel("m(gamma gamma) (MeV)")
            ax.set_ylabel("events / 1 MeV")
            ax.legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        runs = sorted({int(event["run"]) for event in events})
        run_status = {
            int(row["run"]): str(row["status"])
            for row in support_summary(update_rows, chosen, excluded_runs)
        }
        markers = {"run_local": "o", "grouped": "s", "parity_incomplete": "X", "excluded_qa": "D"}
        for model in models:
            rows = [metric.get((f"run_{run}", model)) for run in runs]
            valid = [(run, row) for run, row in zip(runs, rows) if row and int(row["fit_ok"]) == 1]
            if model != chosen:
                ax.errorbar(
                    [run for run, _ in valid],
                    [float(row["fit_mu_residual_mev"]) for _, row in valid],
                    yerr=[float(row["fit_mu_err_mev"]) for _, row in valid],
                    marker="o",
                    ms=2.5,
                    linestyle="none",
                    label=labels[model],
                    color=colors[model],
                    alpha=0.65,
                )
                continue
            for status, marker in markers.items():
                selected = [(run, row) for run, row in valid if run_status.get(run, "run_local") == status]
                if not selected:
                    continue
                ax.errorbar(
                    [run for run, _ in selected],
                    [float(row["fit_mu_residual_mev"]) for _, row in selected],
                    yerr=[float(row["fit_mu_err_mev"]) for _, row in selected],
                    marker=marker,
                    ms=5,
                    linestyle="none",
                    label=labels[model] if status == "run_local" else None,
                    color=colors[model],
                )
        ax.axhline(0, color="black", lw=1)
        ax.axhline(1, color="gray", lw=0.8, ls="--")
        ax.axhline(-1, color="gray", lw=0.8, ls="--")
        ax.set_title("Held-out Gaussian-fit peak residual by run")
        ax.set_xlabel("run")
        ax.set_ylabel("fit peak residual (MeV)")
        handles, legend_labels = ax.get_legend_handles_labels()
        handles.extend(
            Line2D([], [], color="black", marker=marker, linestyle="none", markersize=6)
            for marker in ("o", "s", "X", "D")
        )
        legend_labels.extend(
            ["trusted run fit", "grouped support", "parity incomplete", "excluded DAQ QA"]
        )
        ax.legend(handles, legend_labels, fontsize=8, ncol=2)
        pdf.savefig(fig)
        plt.close(fig)


def plot_production(path: Path, events, masses, metrics, chosen, args) -> None:
    labels = {"hao_baseline": "Hao production baseline", "direct_w150": "direct photon w=1.5", chosen: "direct + full-stat run scale"}
    colors = {"hao_baseline": "#777777", "direct_w150": "#00798c", chosen: "#d1495b"}
    metric = {(str(row["group"]), str(row["model"])): row for row in metrics}
    with PdfPages(path) as pdf:
        for group in ["pooled", "emin_0.6-0.8", "emin_0.8-1.0", "emin_1.0-1.2"] + [f"run_{run}" for run in sorted({int(event['run']) for event in events})]:
            indexes = dict(group_indexes(events))[group]
            fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
            bins = np.arange(105.0, 165.01, 1.0)
            centers = 0.5 * (bins[:-1] + bins[1:])
            for model in ["hao_baseline", "direct_w150", chosen]:
                values = np.asarray([masses[model][i] for i in indexes], dtype=float)
                fit = fit_pi0_mass(values, fit_config(args, 200))
                ax.hist(values, bins=bins, histtype="step", lw=1.6, color=colors[model], label=labels[model])
                if int(fit["optimizer_ok"]):
                    ax.plot(centers, evaluate_model(centers, fit), ls="--", lw=1.2, color=colors[model])
                row = metric[(group, model)]
                ax.plot([], [], color=colors[model], label=f"mu={float(row['fit_mu_mev']):.3f}, sigma={float(row['fit_sigma_mev']):.3f} MeV")
            ax.axvline(M_PI0_MEV, color="black", ls=":", lw=1)
            ax.set_title(f"Production-style run-scale transfer: {group}")
            ax.set_xlabel("m(gamma gamma) (MeV)")
            ax.set_ylabel("events / 1 MeV")
            ax.legend(fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = {int(item) for item in args.runs.split(",") if item.strip()}
    excluded_runs = parse_run_set(args.excluded_calibration_runs)
    margs = model_args(args)
    summary_rows = base.load_conversion_summary(Path(args.conversion_summary))
    events, qa = pairspace.read_events(summary_rows, margs)
    calibration_events = [
        event for event in events if int(event["run"]) not in excluded_runs
    ]
    write_tsv(outdir / "sidecar_quality.tsv", qa)

    saved_curve_rows: list[dict[str, str]] = []
    saved_full_curves = None
    saved_full_curve_rows: list[dict[str, str]] = []
    if args.reuse_existing_curves:
        cache_dir = Path(args.curve_cache_dir) if args.curve_cache_dir else outdir
        heldout_curve_path = cache_dir / "heldout_curve_derivation.tsv"
        package_curve_path = cache_dir / "frozen_package" / "period_photon_curve.tsv"
        package_derivation_path = cache_dir / "frozen_package" / "curve_derivation.tsv"
        if not heldout_curve_path.exists() or not package_curve_path.exists() or not package_derivation_path.exists():
            raise SystemExit("--reuse-existing-curves requested but saved curve tables are missing")
        saved_curve_rows = read_tsv(heldout_curve_path)
        saved_full_curves = curves_from_package_rows(read_tsv(package_curve_path))
        saved_full_curve_rows = read_tsv(package_derivation_path)

    all_events: list[dict[str, object]] = []
    all_masses: dict[str, list[float]] = {"hao_baseline": [], "direct_w150": []}
    for damping in RUN_DAMPINGS:
        all_masses[f"direct_run_d{damping:.2f}"] = []
    fit_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for test_fold in (0, 1):
        fold = f"segment_parity_test_{test_fold}"
        train = [event for event in events if int(event["fold"]) != test_fold]
        test = [event for event in events if int(event["fold"]) == test_fold]
        curve_train = [event for event in train if int(event["run"]) not in excluded_runs]
        if args.reuse_existing_curves:
            curves, knots = curves_from_derivation_rows(saved_curve_rows, fold)
        else:
            curves, rows, knots = derive_direct_curves(fold, curve_train, args, margs)
            curve_rows.extend(rows)
        train_direct = apply_direct(train, curves, knots, curve_train, margs)
        test_direct = apply_direct(test, curves, knots, curve_train, margs)
        fold_masses = {"hao_baseline": [float(event["mass_mev"]) for event in test], "direct_w150": test_direct}
        _, unit_rows = derive_run_scales(
            fold,
            train,
            train_direct,
            {int(event["run"]) for event in test},
            1.0,
            args,
        )
        for damping in RUN_DAMPINGS:
            model = f"direct_run_d{damping:.2f}"
            scales, rows = redamp_run_scale_rows(unit_rows, damping, args)
            update_rows.extend(rows)
            fold_masses[model] = apply_run_scales(test, test_direct, scales)
        fit_rows.extend(evaluate(fold, test, fold_masses, args))
        all_events.extend(test)
        for model, values in fold_masses.items():
            all_masses[model].extend(values)
    fit_rows.extend(evaluate("pooled_folds", all_events, all_masses, args))
    eligible_indexes = [
        index for index, event in enumerate(all_events) if int(event["run"]) not in excluded_runs
    ]
    eligible_events = [all_events[index] for index in eligible_indexes]
    eligible_masses = {
        model: [values[index] for index in eligible_indexes] for model, values in all_masses.items()
    }
    fit_rows.extend(evaluate("selection_eligible", eligible_events, eligible_masses, args))
    models = list(all_masses)
    stability = stability_summary(fit_rows, models, "selection_eligible")
    chosen = choose_model(stability)
    write_tsv(outdir / "heldout_fit_metrics.tsv", fit_rows)
    write_tsv(outdir / "heldout_run_scale_updates.tsv", update_rows)
    write_tsv(outdir / "heldout_curve_derivation.tsv", saved_curve_rows if args.reuse_existing_curves else curve_rows)
    write_tsv(outdir / "heldout_run_stability_summary.tsv", stability)
    chosen_support = support_summary(update_rows, chosen, excluded_runs)
    write_tsv(outdir / "heldout_run_support_summary.tsv", chosen_support)
    plot_holdout(
        outdir / "heldout_run_scale_validation.pdf",
        all_events,
        all_masses,
        fit_rows,
        update_rows,
        models,
        chosen,
        excluded_runs,
        args,
    )

    # Freeze the chosen damping from all calibration events.
    chosen_damping = float(chosen.rsplit("d", 1)[1])
    if saved_full_curves is not None:
        full_curves, full_knots = saved_full_curves
        full_curve_rows = saved_full_curve_rows
    else:
        full_curves, full_curve_rows, full_knots = derive_direct_curves(
            "full_sample", calibration_events, args, margs
        )
    full_direct = apply_direct(events, full_curves, full_knots, calibration_events, margs)
    full_scales, full_scale_rows = derive_run_scales(
        "full_sample",
        events,
        full_direct,
        {int(event["run"]) for event in events},
        chosen_damping,
        args,
    )
    package = outdir / "frozen_package"
    package.mkdir(exist_ok=True)
    write_tsv(package / "curve_derivation.tsv", full_curve_rows)
    write_tsv(package / "run_scale.tsv", full_scale_rows)
    for damping in RUN_DAMPINGS:
        _, damping_rows = redamp_run_scale_rows(full_scale_rows, damping, args)
        write_tsv(package / f"run_scale_d{damping:.2f}.tsv", damping_rows)
    lut_rows = []
    for period, coeff in sorted(full_curves.items()):
        for energy, log_scale in zip(full_knots, coeff):
            lut_rows.append({"period": period, "energy_gev": energy, "energy_scale": math.exp(float(log_scale)), "log_energy_scale": float(log_scale)})
    write_tsv(package / "period_photon_curve.tsv", lut_rows)

    production_events, production_qa = read_production_pairs(Path(args.production_pair_dir), runs, args)
    write_tsv(outdir / "production_pair_quality.tsv", production_qa)
    production_masses = {"hao_baseline": [], "direct_w150": [], chosen: []}
    production_rows = []
    for event in production_events:
        run = int(event["run"])
        coeff = full_curves.get(period_for_run(run), full_curves["global"])
        s1 = direct.full_photon_energy_scale(coeff, float(event["e1"]), full_knots)
        s2 = direct.full_photon_energy_scale(coeff, float(event["e2"]), full_knots)
        mass0 = float(event["mass_mev"])
        mass1 = mass0 * math.sqrt(s1 * s2)
        run_scale = full_scales.get(run, 1.0)
        mass2 = mass1 * run_scale
        production_masses["hao_baseline"].append(mass0)
        production_masses["direct_w150"].append(mass1)
        production_masses[chosen].append(mass2)
        production_rows.append({**event, "period": period_for_run(run), "photon_scale1": s1, "photon_scale2": s2, "run_scale": run_scale, "mass_direct_mev": mass1, "mass_direct_run_mev": mass2})
    write_tsv(outdir / "production_event_comparison.tsv", production_rows)
    production_metrics = evaluate("production_transfer", production_events, production_masses, args)
    write_tsv(outdir / "production_fit_metrics.tsv", production_metrics)
    plot_production(outdir / "production_run_scale_transfer.pdf", production_events, production_masses, production_metrics, chosen, args)

    summary_by_model = {str(row["model"]): row for row in stability}
    prod_map = {(str(row["group"]), str(row["model"])): row for row in production_metrics}
    lines = [
        "Direct Hao-start photon curve plus Gaussian-fit run-scale validation",
        f"chosen held-out candidate: {chosen}",
        f"calibration-excluded runs: {','.join(str(run) for run in sorted(excluded_runs)) or 'none'}",
        f"shared-curve derivation events: {len(calibration_events)} of {len(events)}",
        "",
        "Held-out run stability:",
    ]
    for model in models:
        row = summary_by_model[model]
        lines.append(f"{model}: median/p90/p95/max abs run residual = {float(row['median_abs_run_residual_mev']):.4f}/{float(row['p90_abs_run_residual_mev']):.4f}/{float(row['p95_abs_run_residual_mev']):.4f}/{float(row['max_abs_run_residual_mev']):.4f} MeV; pooled sigma={float(row['pooled_fit_sigma_mev']):.4f} MeV")
    lines.extend(["", "Production-style capped replay:"])
    for model in ["hao_baseline", "direct_w150", chosen]:
        row = prod_map[("pooled", model)]
        lines.append(f"{model}: mu={float(row['fit_mu_mev']):.4f}; residual={float(row['fit_mu_residual_mev']):+.4f}; sigma={float(row['fit_sigma_mev']):.4f} MeV")
    lines.append("")
    for run in sorted(runs):
        row = prod_map[(f"run_{run}", chosen)]
        lines.append(f"run {run} chosen residual/sigma: {float(row['fit_mu_residual_mev']):+.4f}/{float(row['fit_sigma_mev']):.4f} MeV")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
