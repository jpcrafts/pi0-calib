#!/usr/bin/env python3
"""Validate a seed-block residual layer on the corrected Hao DB-start package."""

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

import validate_hao_direct_period_photon_run_scale as runscale
import validate_hao_full_iter2_pairspace as pairspace
import validate_hao_start_layered_member_stack as base
from pi0_mass_fit import M_PI0_MEV, evaluate_model, fit_pi0_mass


SEED_DAMPINGS = (0.50, 0.75, 1.00)
NROWS = 36
NCOLS = 30


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--min-fit-entries", type=int, default=500)
    ap.add_argument("--min-photon-cell-entries", type=int, default=500)
    ap.add_argument("--fit-max-mu-err", type=float, default=0.50)
    ap.add_argument("--fit-max-chi2-ndf", type=float, default=math.inf)
    ap.add_argument("--fit-warn-chi2-ndf", type=float, default=3.0)
    ap.add_argument("--fit-max-window-shift", type=float, default=0.50)
    ap.add_argument("--min-seed-events", type=int, default=200)
    ap.add_argument("--seed-shrink-events", type=float, default=400.0)
    ap.add_argument("--max-seed-update-frac", type=float, default=0.01)
    ap.add_argument("--left-cols", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--excluded-calibration-runs", default="4552")
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def clamp(value: float, fraction: float) -> float:
    return min(1.0 + fraction, max(1.0 - fraction, value))


def block_row_col(block: int) -> tuple[int, int]:
    return block // NCOLS, block % NCOLS


def normalize_seed_updates(
    events: list[dict[str, object]],
    scales: dict[int, float],
    excluded_runs: set[int],
    max_update_frac: float,
) -> tuple[dict[int, float], float, float]:
    """Remove the event-weighted common mode while preserving identity fallbacks."""
    normalized = dict(scales)
    total_log_shift = 0.0
    final_mean_log_mass_shift = 0.0
    for _ in range(12):
        log_mass_sum = 0.0
        active_seed_weight = 0.0
        for event in events:
            if int(event["run"]) in excluded_runs:
                continue
            for key in ("seed1_block", "seed2_block"):
                block = int(event[key])
                if block in normalized:
                    log_mass_sum += 0.5 * math.log(normalized[block])
                    active_seed_weight += 0.5
        if active_seed_weight <= 0.0:
            break
        log_shift = log_mass_sum / active_seed_weight
        final_mean_log_mass_shift = log_mass_sum / max(1.0, len(events))
        if abs(log_shift) < 1.0e-12:
            break
        total_log_shift += log_shift
        divisor = math.exp(log_shift)
        normalized = {
            block: clamp(scale / divisor, max_update_frac) for block, scale in normalized.items()
        }
    return normalized, total_log_shift, final_mean_log_mass_shift


def fold_run_scales(rows: list[dict[str, str]], fold: str) -> dict[int, float]:
    selected = [
        row
        for row in rows
        if row["fold"] == fold and math.isclose(float(row["damping"]), 1.0, abs_tol=1.0e-12)
    ]
    if not selected:
        raise ValueError(f"no d=1.00 run scales for {fold}")
    return {int(row["run"]): float(row["applied_energy_scale"]) for row in selected}


def derive_seed_updates(
    fold: str,
    events: list[dict[str, object]],
    masses: list[float],
    args: argparse.Namespace,
) -> tuple[dict[float, dict[int, float]], list[dict[str, object]]]:
    left_cols = set(args.left_cols)
    excluded = runscale.parse_run_set(args.excluded_calibration_runs)
    by_seed: dict[int, list[float]] = defaultdict(list)
    global_values: list[float] = []
    for event, mass in zip(events, masses):
        if int(event["run"]) in excluded or int(event["low_col"]) in left_cols:
            continue
        by_seed[int(event["low_seed"])].append(float(mass))
        global_values.append(float(mass))

    cfg = runscale.fit_config(args, args.min_seed_events)
    global_fit = fit_pi0_mass(np.asarray(global_values, dtype=float), cfg)
    if int(global_fit["quality_ok"]) != 1:
        raise RuntimeError(f"global seed reference fit failed for {fold}: {global_fit['quality_reasons']}")
    global_mu = float(global_fit["mu_mev"])
    updates: dict[float, dict[int, float]] = {damping: {} for damping in SEED_DAMPINGS}
    block_records: list[dict[str, object]] = []
    for block, values in sorted(by_seed.items()):
        fit = fit_pi0_mass(np.asarray(values, dtype=float), cfg)
        accepted = len(values) >= args.min_seed_events and int(fit["quality_ok"]) == 1
        raw_residual = float(fit["mu_mev"]) - global_mu if accepted else math.nan
        shrink = len(values) / (len(values) + args.seed_shrink_events)
        regularized_residual = raw_residual * shrink if accepted else math.nan
        full_scale = (
            (M_PI0_MEV / (M_PI0_MEV + regularized_residual)) ** 2 if accepted else 1.0
        )
        row, col = block_row_col(block)
        block_records.append(
            {
                "fold": fold,
                "seed_block": block,
                "row": row,
                "col": col,
                "entries": len(values),
                "fit_entries": fit["window_entries"],
                "fit_ok": fit["quality_ok"],
                "fit_reasons": fit["quality_reasons"],
                "fit_mu_mev": fit["mu_mev"],
                "fit_mu_err_mev": fit["mu_err_mev"],
                "fit_sigma_mev": fit["sigma_mev"],
                "global_reference_mu_mev": global_mu,
                "raw_spatial_residual_mev": raw_residual,
                "shrink_factor": shrink,
                "regularized_spatial_residual_mev": regularized_residual,
                "full_energy_scale": full_scale,
                "support_pass": int(accepted),
            }
        )

    rows: list[dict[str, object]] = []
    for damping in SEED_DAMPINGS:
        pre_normalized = {
            int(record["seed_block"]): clamp(
                float(record["full_energy_scale"]) ** damping, args.max_seed_update_frac
            )
            for record in block_records
            if int(record["support_pass"]) == 1
        }
        normalized, log_shift, remaining_mass_shift = normalize_seed_updates(
            events, pre_normalized, excluded, args.max_seed_update_frac
        )
        updates[damping] = normalized
        for record in block_records:
            block = int(record["seed_block"])
            accepted = int(record["support_pass"]) == 1
            pre = pre_normalized.get(block, 1.0)
            applied = normalized.get(block, 1.0)
            rows.append(
                {
                    **record,
                    "damping": damping,
                    "pre_normalization_energy_scale": pre,
                    "normalization_log_energy_shift": log_shift,
                    "remaining_mean_log_mass_shift": remaining_mass_shift,
                    "applied_energy_scale": applied,
                    "clipped": int(
                        accepted
                        and (
                            math.isclose(pre, 1.0 - args.max_seed_update_frac, abs_tol=1e-12)
                            or math.isclose(pre, 1.0 + args.max_seed_update_frac, abs_tol=1e-12)
                            or math.isclose(applied, 1.0 - args.max_seed_update_frac, abs_tol=1e-12)
                            or math.isclose(applied, 1.0 + args.max_seed_update_frac, abs_tol=1e-12)
                        )
                    ),
                }
            )
    return updates, rows


def apply_seed(
    events: list[dict[str, object]], masses: list[float], updates: dict[int, float]
) -> list[float]:
    out = []
    for event, mass in zip(events, masses):
        scale1 = updates.get(int(event["seed1_block"]), 1.0)
        scale2 = updates.get(int(event["seed2_block"]), 1.0)
        out.append(float(mass) * math.sqrt(max(1.0e-12, scale1 * scale2)))
    return out


def fit_metric_rows(
    fold: str,
    events: list[dict[str, object]],
    masses: dict[str, list[float]],
    args: argparse.Namespace,
    include_run_energy: bool = False,
) -> list[dict[str, object]]:
    rows = runscale.evaluate(fold, events, masses, args, include_run_energy=include_run_energy)
    regions = {
        "region_all": list(range(len(events))),
        "region_bulk_cols_ge3": [i for i, event in enumerate(events) if int(event["low_col"]) >= 3],
        "region_left_cols_0_2": [i for i, event in enumerate(events) if int(event["low_col"]) <= 2],
    }
    for col in range(3):
        regions[f"region_col_{col}"] = [i for i, event in enumerate(events) if int(event["low_col"]) == col]
    for group, indexes in regions.items():
        if not indexes:
            continue
        for model, values in masses.items():
            fit = fit_pi0_mass(
                np.asarray([values[index] for index in indexes], dtype=float),
                runscale.fit_config(args),
            )
            rows.append(
                {
                    "fold": fold,
                    "group": group,
                    "model": model,
                    "entries": len(indexes),
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


def block_fit_rows(
    events: list[dict[str, object]], masses: dict[str, list[float]], args: argparse.Namespace
) -> list[dict[str, object]]:
    indexes: dict[int, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        indexes[int(event["low_seed"])].append(index)
    rows = []
    cfg = runscale.fit_config(args, args.min_seed_events)
    for block, selected in sorted(indexes.items()):
        row, col = block_row_col(block)
        for model, values in masses.items():
            fit = fit_pi0_mass(np.asarray([values[index] for index in selected], dtype=float), cfg)
            rows.append(
                {
                    "model": model,
                    "seed_block": block,
                    "row": row,
                    "col": col,
                    "entries": len(selected),
                    "fit_entries": fit["window_entries"],
                    "fit_ok": fit["quality_ok"],
                    "fit_reasons": fit["quality_reasons"],
                    "fit_mu_residual_mev": fit["mu_residual_mev"],
                    "fit_mu_err_mev": fit["mu_err_mev"],
                    "fit_sigma_mev": fit["sigma_mev"],
                }
            )
    return rows


def block_summary(rows: list[dict[str, object]], models: list[str]) -> list[dict[str, object]]:
    out = []
    for model in models:
        valid = [
            abs(float(row["fit_mu_residual_mev"]))
            for row in rows
            if row["model"] == model and int(row["fit_ok"]) == 1 and int(row["col"]) >= 3
        ]
        arr = np.asarray(valid, dtype=float)
        out.append(
            {
                "model": model,
                "valid_bulk_blocks": len(arr),
                "median_abs_block_fit_residual_mev": float(np.median(arr)) if len(arr) else math.nan,
                "p90_abs_block_fit_residual_mev": float(np.percentile(arr, 90)) if len(arr) else math.nan,
                "p95_abs_block_fit_residual_mev": float(np.percentile(arr, 95)) if len(arr) else math.nan,
                "max_abs_block_fit_residual_mev": float(np.max(arr)) if len(arr) else math.nan,
            }
        )
    return out


def plot_results(
    path: Path,
    events: list[dict[str, object]],
    masses: dict[str, list[float]],
    metrics: list[dict[str, object]],
    block_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    models = list(masses)
    colors = dict(zip(models, ["#555555", "#00798c", "#edae49", "#d1495b"]))
    labels = {"direct_run_d1.00": "direct curve + d=1 run scale"}
    labels.update({f"seed_d{d:.2f}": f"+ seed residual d={d:.2f}" for d in SEED_DAMPINGS})
    metric = {
        (str(row["group"]), str(row["model"])): row
        for row in metrics
        if row["fold"] == "pooled_folds"
    }
    groups = dict(runscale.group_indexes(events))
    with PdfPages(path) as pdf:
        for group in ("pooled", "emin_0.6-0.8", "emin_0.8-1.0", "emin_1.0-1.2"):
            indexes = groups[group]
            fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
            bins = np.arange(105.0, 165.01, 1.0)
            centers = 0.5 * (bins[:-1] + bins[1:])
            for model in models:
                values = np.asarray([masses[model][index] for index in indexes], dtype=float)
                fit = fit_pi0_mass(values, runscale.fit_config(args))
                ax.hist(values, bins=bins, histtype="step", lw=1.5, color=colors[model], label=labels[model])
                if int(fit["optimizer_ok"]) == 1:
                    ax.plot(centers, evaluate_model(centers, fit), ls="--", lw=1.1, color=colors[model])
                row = metric[(group, model)]
                ax.plot([], [], color=colors[model], label=f"mu={float(row['fit_mu_mev']):.3f}, sigma={float(row['fit_sigma_mev']):.3f} MeV")
            ax.axvline(M_PI0_MEV, color="black", ls=":", lw=1)
            ax.set(title=f"Held-out invariant mass: {group}", xlabel="m(gamma gamma) (MeV)", ylabel="events / 1 MeV")
            ax.legend(fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        runs = sorted({int(event["run"]) for event in events})
        for model in models:
            points = [metric.get((f"run_{run}", model)) for run in runs]
            valid = [(run, row) for run, row in zip(runs, points) if row and int(row["fit_ok"]) == 1]
            ax.errorbar(
                [run for run, _ in valid],
                [float(row["fit_mu_residual_mev"]) for _, row in valid],
                yerr=[float(row["fit_mu_err_mev"]) for _, row in valid],
                marker="o", ms=2.5, linestyle="none", alpha=0.75, color=colors[model], label=labels[model],
            )
        ax.axhline(0, color="black", lw=1)
        ax.axhline(1, color="gray", ls="--", lw=0.8)
        ax.axhline(-1, color="gray", ls="--", lw=0.8)
        ax.set(title="Held-out Gaussian-fit peak residual by run", xlabel="run", ylabel="fit peak residual (MeV)")
        ax.legend(fontsize=8, ncol=2)
        pdf.savefig(fig)
        plt.close(fig)

        valid_values = [
            abs(float(row["fit_mu_residual_mev"]))
            for row in block_rows
            if int(row["fit_ok"]) == 1 and int(row["col"]) >= 3
        ]
        vmax = max(0.5, float(np.percentile(valid_values, 95))) if valid_values else 1.0
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
        for ax, model in zip(axes.flat, models):
            grid = np.full((NROWS, NCOLS), np.nan)
            for row in block_rows:
                if row["model"] == model and int(row["fit_ok"]) == 1:
                    grid[int(row["row"]), int(row["col"])] = float(row["fit_mu_residual_mev"])
            image = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.axvspan(-0.5, 2.5, color="black", alpha=0.10)
            ax.set(title=labels[model], xlabel="column", ylabel="row")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="held-out Gaussian-fit peak residual (MeV)")
        fig.suptitle("Seed-block closure; columns 0-2 are QA-only and receive identity residual scale")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    baseline = Path(args.baseline_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    margs = runscale.model_args(args)
    summary_rows = base.load_conversion_summary(Path(args.conversion_summary))
    events, quality = pairspace.read_events(summary_rows, margs)
    write_tsv(outdir / "sidecar_quality.tsv", quality)

    curve_rows = read_tsv(baseline / "heldout_curve_derivation.tsv")
    run_rows = read_tsv(baseline / "heldout_run_scale_updates.tsv")
    excluded = runscale.parse_run_set(args.excluded_calibration_runs)
    all_events: list[dict[str, object]] = []
    all_masses: dict[str, list[float]] = {"direct_run_d1.00": []}
    for damping in SEED_DAMPINGS:
        all_masses[f"seed_d{damping:.2f}"] = []
    fit_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []

    for test_fold in (0, 1):
        fold = f"segment_parity_test_{test_fold}"
        train = [event for event in events if int(event["fold"]) != test_fold]
        test = [event for event in events if int(event["fold"]) == test_fold]
        curve_train = [event for event in train if int(event["run"]) not in excluded]
        curves, knots = runscale.curves_from_derivation_rows(curve_rows, fold)
        scales = fold_run_scales(run_rows, fold)
        train_base = runscale.apply_run_scales(train, runscale.apply_direct(train, curves, knots, curve_train, margs), scales)
        test_base = runscale.apply_run_scales(test, runscale.apply_direct(test, curves, knots, curve_train, margs), scales)
        seed_updates, rows = derive_seed_updates(fold, train, train_base, args)
        update_rows.extend(rows)
        fold_masses = {"direct_run_d1.00": test_base}
        for damping in SEED_DAMPINGS:
            fold_masses[f"seed_d{damping:.2f}"] = apply_seed(test, test_base, seed_updates[damping])
        fit_rows.extend(fit_metric_rows(fold, test, fold_masses, args))
        all_events.extend(test)
        for model, values in fold_masses.items():
            all_masses[model].extend(values)

    fit_rows.extend(fit_metric_rows("pooled_folds", all_events, all_masses, args))
    eligible_indexes = [
        index for index, event in enumerate(all_events) if int(event["run"]) not in excluded
    ]
    eligible_events = [all_events[index] for index in eligible_indexes]
    eligible_masses = {
        model: [values[index] for index in eligible_indexes] for model, values in all_masses.items()
    }
    fit_rows.extend(
        fit_metric_rows(
            "selection_eligible",
            eligible_events,
            eligible_masses,
            args,
            include_run_energy=False,
        )
    )
    block_rows = block_fit_rows(all_events, all_masses, args)
    models = list(all_masses)
    stability = runscale.stability_summary(fit_rows, models, "selection_eligible")
    block_stability = block_summary(block_rows, models)
    write_tsv(outdir / "heldout_fit_metrics.tsv", fit_rows)
    write_tsv(outdir / "heldout_seed_updates.tsv", update_rows)
    write_tsv(outdir / "heldout_block_fit_residuals.tsv", block_rows)
    write_tsv(outdir / "heldout_run_stability_summary.tsv", stability)
    write_tsv(outdir / "heldout_block_stability_summary.tsv", block_stability)
    plot_results(outdir / "heldout_seed_layer_validation.pdf", all_events, all_masses, fit_rows, block_rows, args)

    # Freeze full-sample seed residual tables on the already frozen corrected baseline.
    package = baseline / "frozen_package"
    full_curves, full_knots = runscale.curves_from_package_rows(read_tsv(package / "period_photon_curve.tsv"))
    full_scales = {
        int(row["run"]): float(row["applied_energy_scale"])
        for row in read_tsv(package / "run_scale_d1.00.tsv")
    }
    calibration_events = [event for event in events if int(event["run"]) not in excluded]
    full_direct = runscale.apply_direct(events, full_curves, full_knots, calibration_events, margs)
    full_base = runscale.apply_run_scales(events, full_direct, full_scales)
    _, frozen_rows = derive_seed_updates("full_sample", events, full_base, args)
    frozen_dir = outdir / "frozen_seed_package"
    frozen_dir.mkdir(exist_ok=True)
    write_tsv(frozen_dir / "seed_scale_all_dampings.tsv", frozen_rows)
    for damping in SEED_DAMPINGS:
        write_tsv(
            frozen_dir / f"seed_scale_d{damping:.2f}.tsv",
            [row for row in frozen_rows if math.isclose(float(row["damping"]), damping, abs_tol=1e-12)],
        )

    stability_map = {str(row["model"]): row for row in stability}
    block_map = {str(row["model"]): row for row in block_stability}
    lines = [
        "Corrected Hao DB-start direct+d=1 run scale plus seed-block residual validation",
        f"events: {len(events)}",
        f"excluded calibration runs: {','.join(str(run) for run in sorted(excluded)) or 'none'}",
        f"left columns held identity/QA-only: {','.join(str(col) for col in sorted(args.left_cols))}",
        f"seed support/shrink/clip: {args.min_seed_events}/{args.seed_shrink_events:.0f}/{args.max_seed_update_frac:.3f}",
        "",
    ]
    for model in models:
        run_row = stability_map[model]
        block_row = block_map[model]
        lines.append(
            f"{model}: pooled sigma={float(run_row['pooled_fit_sigma_mev']):.4f} MeV; "
            f"lowE residual={float(run_row['lowe_fit_residual_mev']):+.4f} MeV; "
            f"run median/p95 abs residual={float(run_row['median_abs_run_residual_mev']):.4f}/"
            f"{float(run_row['p95_abs_run_residual_mev']):.4f} MeV; "
            f"bulk block median/p95 abs fit residual={float(block_row['median_abs_block_fit_residual_mev']):.4f}/"
            f"{float(block_row['p95_abs_block_fit_residual_mev']):.4f} MeV"
        )
    lines.extend(
        [
            "",
            "Interpretation guard:",
            "  seed factors are trained on the opposite segment-parity fold and applied to both photon seeds;",
            "  per-block centers and all reported widths come from Gaussian fits;",
            "  the seed layer is spatially centered so it cannot replace the global/run energy layers;",
            "  no member/hybrid residual is included in this test.",
        ]
    )
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
