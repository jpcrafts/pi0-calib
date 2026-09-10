#!/usr/bin/env python3
"""Held-out per-photon low-energy adjustment on corrected Hao seed baseline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import validate_hao_direct_period_photon_run_scale as runscale
import validate_hao_direct_run_scale_seed as seed
import validate_hao_full_iter2_pairspace as pairspace
import validate_hao_start_layered_member_stack as base
from pi0_mass_fit import M_PI0_MEV, evaluate_model, fit_pi0_mass


DAMPINGS = (0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--seed-scale-tsv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--lowe-range", nargs=2, type=float, default=[0.6, 0.8])
    ap.add_argument("--min-fit-entries", type=int, default=500)
    ap.add_argument("--min-photon-cell-entries", type=int, default=500)
    ap.add_argument("--fit-max-mu-err", type=float, default=0.50)
    ap.add_argument("--fit-max-chi2-ndf", type=float, default=math.inf)
    ap.add_argument("--fit-warn-chi2-ndf", type=float, default=3.0)
    ap.add_argument("--fit-max-window-shift", type=float, default=0.50)
    ap.add_argument("--min-seed-events", type=int, default=200)
    ap.add_argument("--seed-shrink-events", type=float, default=400.0)
    ap.add_argument("--max-seed-update-frac", type=float, default=0.01)
    ap.add_argument("--max-lowe-energy-update-frac", type=float, default=0.006)
    ap.add_argument(
        "--lowe-scope",
        choices=("global", "period"),
        default="global",
        help="Derive one low-energy scale globally or one scale per configured run period.",
    )
    ap.add_argument("--seed-damping", type=float, choices=DAMPINGS, default=0.75)
    ap.add_argument("--left-cols", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--excluded-calibration-runs", default="4552")
    ap.add_argument("--run-period-lut", type=Path)
    return ap.parse_args()


def load_seed_scales(path: Path) -> dict[int, float]:
    return {
        int(row["seed_block"]): float(row["applied_energy_scale"])
        for row in seed.read_tsv(path)
        if int(row["support_pass"]) == 1
    }


def photon_count(event: dict[str, object], lowe_range: list[float]) -> int:
    lo, hi = lowe_range
    return int(lo <= float(event["e1"]) < hi) + int(lo <= float(event["e2"]) < hi)


def derive_lowe_updates(
    fold: str,
    events: list[dict[str, object]],
    masses: list[float],
    args: argparse.Namespace,
    excluded: set[int],
) -> tuple[dict[float, float], list[dict[str, object]]]:
    selected = [
        (event, mass)
        for event, mass in zip(events, masses)
        if int(event["run"]) not in excluded and photon_count(event, args.lowe_range) > 0
    ]
    values = np.asarray([float(mass) for _event, mass in selected], dtype=float)
    fit = fit_pi0_mass(values, runscale.fit_config(args))
    if int(fit["quality_ok"]) != 1:
        raise RuntimeError(f"{fold}: low-E training fit failed: {fit['quality_reasons']}")
    mean_photons = float(np.mean([photon_count(event, args.lowe_range) for event, _mass in selected]))
    full_energy_scale = (M_PI0_MEV / float(fit["mu_mev"])) ** (2.0 / mean_photons)
    updates: dict[float, float] = {}
    rows = []
    for damping in DAMPINGS:
        applied = seed.clamp(full_energy_scale**damping, args.max_lowe_energy_update_frac)
        updates[damping] = applied
        rows.append(
            {
                "fold": fold,
                "emin": args.lowe_range[0],
                "emax": args.lowe_range[1],
                "events": len(selected),
                "mean_adjusted_photons_per_event": mean_photons,
                "fit_mu_mev": fit["mu_mev"],
                "fit_mu_err_mev": fit["mu_err_mev"],
                "fit_sigma_mev": fit["sigma_mev"],
                "full_photon_energy_scale": full_energy_scale,
                "damping": damping,
                "applied_photon_energy_scale": applied,
                "clipped": int(
                    math.isclose(applied, 1.0 - args.max_lowe_energy_update_frac, abs_tol=1e-12)
                    or math.isclose(applied, 1.0 + args.max_lowe_energy_update_frac, abs_tol=1e-12)
                ),
            }
        )
    return updates, rows


def derive_period_lowe_updates(
    fold: str,
    events: list[dict[str, object]],
    masses: list[float],
    args: argparse.Namespace,
    excluded: set[int],
) -> tuple[dict[float, dict[str, float]], list[dict[str, object]]]:
    updates = {damping: {} for damping in DAMPINGS}
    rows: list[dict[str, object]] = []
    periods = sorted({runscale.period_for_run(int(event["run"])) for event in events})
    for period in periods:
        indexes = [
            index
            for index, event in enumerate(events)
            if runscale.period_for_run(int(event["run"])) == period
        ]
        period_events = [events[index] for index in indexes]
        period_masses = [masses[index] for index in indexes]
        period_updates, period_rows = derive_lowe_updates(
            f"{fold}:{period}", period_events, period_masses, args, excluded
        )
        for damping, scale in period_updates.items():
            updates[damping][period] = scale
        for row in period_rows:
            row["fold"] = fold
            row["period"] = period
            rows.append(row)
    return updates, rows


def apply_lowe(
    events: list[dict[str, object]], masses: list[float], energy_scale: float, lowe_range: list[float]
) -> list[float]:
    return [
        float(mass) * energy_scale ** (0.5 * photon_count(event, lowe_range))
        for event, mass in zip(events, masses)
    ]


def apply_period_lowe(
    events: list[dict[str, object]],
    masses: list[float],
    energy_scales: dict[str, float],
    lowe_range: list[float],
) -> list[float]:
    return [
        float(mass)
        * energy_scales[runscale.period_for_run(int(event["run"]))]
        ** (0.5 * photon_count(event, lowe_range))
        for event, mass in zip(events, masses)
    ]


def plot_results(
    path: Path,
    events: list[dict[str, object]],
    masses: dict[str, list[float]],
    metrics: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    colors = dict(zip(masses, plt.cm.tab10(np.linspace(0.05, 0.85, len(masses)))))
    metric = {
        (str(row["group"]), str(row["model"])): row
        for row in metrics
        if row["fold"] == "pooled_folds"
    }
    groups = [("pooled", list(range(len(events))))]
    groups += [
        (
            f"emin_{base.bin_label(energy_bin)}",
            [i for i, event in enumerate(events) if base.energy_bin(float(event["min_e"])) == energy_bin],
        )
        for energy_bin in base.ENERGY_BINS
    ]
    with PdfPages(path) as pdf:
        for group, indexes in groups:
            fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
            bins = np.arange(105.0, 165.01, 1.0)
            centers = 0.5 * (bins[:-1] + bins[1:])
            for model, values in masses.items():
                selected = np.asarray([values[index] for index in indexes], dtype=float)
                fit = fit_pi0_mass(selected, runscale.fit_config(args))
                row = metric[(group, model)]
                ax.hist(selected, bins=bins, histtype="step", lw=1.3, color=colors[model], label=model)
                if int(fit["optimizer_ok"]) == 1:
                    ax.plot(centers, evaluate_model(centers, fit), ls="--", lw=1.0, color=colors[model])
                ax.plot([], [], color=colors[model], label=f"mu={float(row['fit_mu_mev']):.3f}, sigma={float(row['fit_sigma_mev']):.3f}")
            ax.axvline(M_PI0_MEV, color="black", ls=":", lw=1)
            ax.set(title=f"Held-out low-E photon adjustment: {group}", xlabel="m(gamma gamma) (MeV)", ylabel="events / 1 MeV")
            ax.legend(fontsize=7, ncol=2)
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        for model in ["seed_d0.75"] + [f"lowe_d{damping:.2f}" for damping in DAMPINGS]:
            points = []
            for row in metrics:
                group = str(row["group"])
                if (
                    row["fold"] == "pooled_folds"
                    and row["model"] == model
                    and group.startswith("run_")
                    and group.endswith("_emin_0.6-0.8")
                    and int(row["fit_ok"]) == 1
                ):
                    points.append((int(group.split("_")[1]), float(row["fit_mu_residual_mev"]), float(row["fit_mu_err_mev"])))
            points.sort()
            ax.errorbar(
                [point[0] for point in points], [point[1] for point in points],
                yerr=[point[2] for point in points], marker="o", ms=3, linestyle="none",
                alpha=0.75, color=colors[model], label=model,
            )
        ax.axhline(0.0, color="black", lw=1)
        ax.set(title="Held-out 0.6-0.8 GeV fitted peak by run", xlabel="run", ylabel="fit peak residual (MeV)")
        ax.legend(fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    baseline = Path(args.baseline_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    excluded = runscale.parse_run_set(args.excluded_calibration_runs)
    summary_rows = base.load_conversion_summary(Path(args.conversion_summary))
    events, quality = pairspace.read_events(summary_rows, runscale.model_args(args))
    seed.write_tsv(outdir / "sidecar_quality.tsv", quality)
    print(f"loaded selected events: {len(events)}", flush=True)

    curve_rows = seed.read_tsv(baseline / "heldout_curve_derivation.tsv")
    run_rows = seed.read_tsv(baseline / "heldout_run_scale_updates.tsv")
    fit_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    all_masses: dict[str, list[float]] = {}

    for test_fold in (0, 1):
        fold = f"segment_parity_test_{test_fold}"
        train = [event for event in events if int(event["fold"]) != test_fold]
        test = [event for event in events if int(event["fold"]) == test_fold]
        curve_train = [event for event in train if int(event["run"]) not in excluded]
        curves, knots = runscale.curves_from_derivation_rows(curve_rows, fold)
        run_scales = seed.fold_run_scales(run_rows, fold)
        train_base = runscale.apply_run_scales(
            train, runscale.apply_direct(train, curves, knots, curve_train, runscale.model_args(args)), run_scales
        )
        test_base = runscale.apply_run_scales(
            test, runscale.apply_direct(test, curves, knots, curve_train, runscale.model_args(args)), run_scales
        )
        seed_updates, rows = seed.derive_seed_updates(fold, train, train_base, args)
        update_rows.extend(rows)
        train_seed = seed.apply_seed(train, train_base, seed_updates[args.seed_damping])
        test_seed = seed.apply_seed(test, test_base, seed_updates[args.seed_damping])
        if args.lowe_scope == "period":
            lowe_updates, rows = derive_period_lowe_updates(
                fold, train, train_seed, args, excluded
            )
        else:
            lowe_updates, rows = derive_lowe_updates(fold, train, train_seed, args, excluded)
        update_rows.extend(rows)
        fold_masses = {f"seed_d{args.seed_damping:.2f}": test_seed}
        for damping in DAMPINGS:
            if args.lowe_scope == "period":
                fold_masses[f"lowe_d{damping:.2f}"] = apply_period_lowe(
                    test, test_seed, lowe_updates[damping], args.lowe_range
                )
            else:
                fold_masses[f"lowe_d{damping:.2f}"] = apply_lowe(
                    test, test_seed, lowe_updates[damping], args.lowe_range
                )
        fit_rows.extend(seed.fit_metric_rows(fold, test, fold_masses, args))
        all_events.extend(test)
        for model, values in fold_masses.items():
            all_masses.setdefault(model, []).extend(values)

    fit_rows.extend(seed.fit_metric_rows("pooled_folds", all_events, all_masses, args, include_run_energy=True))
    eligible_indexes = [i for i, event in enumerate(all_events) if int(event["run"]) not in excluded]
    eligible_events = [all_events[i] for i in eligible_indexes]
    eligible_masses = {model: [values[i] for i in eligible_indexes] for model, values in all_masses.items()}
    fit_rows.extend(seed.fit_metric_rows("selection_eligible", eligible_events, eligible_masses, args))
    seed.write_tsv(outdir / "heldout_fit_metrics.tsv", fit_rows)
    seed.write_tsv(outdir / "heldout_update_tables.tsv", update_rows)
    stability = runscale.stability_summary(fit_rows, list(all_masses), "selection_eligible")
    seed.write_tsv(outdir / "heldout_run_stability_summary.tsv", stability)
    plot_results(outdir / "heldout_lowe_photon_adjustment.pdf", all_events, all_masses, fit_rows, args)

    package = baseline / "frozen_package"
    full_curves, full_knots = runscale.curves_from_package_rows(seed.read_tsv(package / "period_photon_curve.tsv"))
    full_scales = {int(row["run"]): float(row["applied_energy_scale"]) for row in seed.read_tsv(package / "run_scale_d1.00.tsv")}
    calibration_events = [event for event in events if int(event["run"]) not in excluded]
    full_base = runscale.apply_run_scales(
        events, runscale.apply_direct(events, full_curves, full_knots, calibration_events, runscale.model_args(args)), full_scales
    )
    full_seed = seed.apply_seed(events, full_base, load_seed_scales(Path(args.seed_scale_tsv)))
    if args.lowe_scope == "period":
        _updates, frozen_rows = derive_period_lowe_updates(
            "full_sample", events, full_seed, args, excluded
        )
    else:
        _updates, frozen_rows = derive_lowe_updates(
            "full_sample", events, full_seed, args, excluded
        )
    frozen_dir = outdir / "frozen_lowe_package"
    frozen_dir.mkdir(exist_ok=True)
    seed.write_tsv(frozen_dir / "lowe_photon_energy_scales.tsv", frozen_rows)

    stability_map = {str(row["model"]): row for row in stability}
    lines = [
        "Corrected Hao seed-only per-photon low-energy adjustment",
        f"events: {len(events)}",
        f"photon range: {args.lowe_range[0]:.1f}-{args.lowe_range[1]:.1f} GeV",
        f"low-energy scope: {args.lowe_scope}",
        "",
    ]
    for model in all_masses:
        row = stability_map[model]
        lines.append(
            f"{model}: pooled residual/sigma={float(row['pooled_fit_residual_mev']):+.4f}/"
            f"{float(row['pooled_fit_sigma_mev']):.4f} MeV; lowE residual/sigma="
            f"{float(row['lowe_fit_residual_mev']):+.4f}/{float(row['lowe_fit_sigma_mev']):.4f} MeV; "
            f"run median/p95={float(row['median_abs_run_residual_mev']):.4f}/"
            f"{float(row['p95_abs_run_residual_mev']):.4f} MeV"
        )
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
