#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import validate_hao_period_energy_smoothed_seed as v
import validate_hao_start_layered_member_stack as base
from build_energy_correction_stack import M_PI0_MEV
from plot_period_fallback_mgg_fits import fit_peak


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Full-layer Hao-start iteration check. Each pass rederives energy, "
            "period-energy, low-E patches, and seed from the previous pass corrected masses."
        )
    )
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--full-iterations", type=int, default=3)
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[105.0, 165.0])
    ap.add_argument("--min-fit-entries", type=int, default=200)
    ap.add_argument("--min-seed-events", type=int, default=20)
    ap.add_argument("--energy-damping", type=float, default=0.75)
    ap.add_argument("--period-energy-damping", type=float, default=0.75)
    ap.add_argument("--energy-bin-damping", action="append", default=[])
    ap.add_argument("--period-energy-bin-damping", action="append", default=[])
    ap.add_argument("--seed-damping", type=float, default=0.75)
    ap.add_argument("--max-energy-mass-update-frac", type=float, default=0.03)
    ap.add_argument("--max-period-energy-mass-update-frac", type=float, default=0.015)
    ap.add_argument("--max-seed-energy-update-frac", type=float, default=0.005)
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
    ap.add_argument("--fit-detail", choices=["summary", "run_energy"], default="run_energy")
    ap.add_argument("--skip-plots", action="store_true")
    return ap.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="ascii")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reset_period_caches(args: argparse.Namespace) -> None:
    for key in list(vars(args)):
        if key.startswith("_run_band_map") or key.startswith("_lowe_run_band_map") or key.startswith("_lowe_bin_patch_"):
            delattr(args, key)


def events_with_masses(events: list[dict[str, object]], masses: list[float]) -> list[dict[str, object]]:
    if len(events) != len(masses):
        raise ValueError(f"event/mass length mismatch: {len(events)} != {len(masses)}")
    out = []
    for ev, mass in zip(events, masses):
        row = dict(ev)
        row["mass_mev"] = float(mass)
        out.append(row)
    return out


def derive_energy_updates_from_masses(
    train: list[dict[str, object]],
    masses: list[float],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    temp = events_with_masses(train, masses)
    return v.derive_energy_updates(temp, args)


def apply_full_pass(
    pass_index: int,
    fold: str,
    train: list[dict[str, object]],
    train_input: list[float],
    target: list[dict[str, object]],
    target_input: list[float],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[dict[str, object]]]:
    reset_period_caches(args)
    update_rows: list[dict[str, object]] = []

    energy_updates, rows = derive_energy_updates_from_masses(train, train_input, args)
    update_rows.extend({"fold": fold, "full_iteration": pass_index, **row} for row in rows)

    train_energy = {
        i: base.apply_energy(train_input[i], ev, energy_updates) for i, ev in enumerate(train)
    }
    target_energy = [
        base.apply_energy(target_input[i], ev, energy_updates) for i, ev in enumerate(target)
    ]

    period_updates, rows = v.derive_period_energy_updates(train, train_energy, args)
    update_rows.extend({"fold": fold, "full_iteration": pass_index, **row} for row in rows)

    train_period = {
        i: v.apply_period_energy(train_energy[i], ev, train, period_updates, args) for i, ev in enumerate(train)
    }
    target_period = [
        v.apply_period_energy(target_energy[i], ev, train, period_updates, args) for i, ev in enumerate(target)
    ]

    lowe_updates, rows = v.derive_lowe_patch_updates(train, train_period, args)
    update_rows.extend({"fold": fold, "full_iteration": pass_index, **row} for row in rows)

    train_lowe = {
        i: v.apply_lowe_patch(train_period[i], ev, train, lowe_updates, args) for i, ev in enumerate(train)
    }
    target_lowe = [
        v.apply_lowe_patch(target_period[i], ev, train, lowe_updates, args) for i, ev in enumerate(target)
    ]

    lowe_run_updates, rows = v.derive_lowe_run_patch_updates(train, train_lowe, args)
    update_rows.extend({"fold": fold, "full_iteration": pass_index, **row} for row in rows)

    train_lowe_run = {
        i: v.apply_lowe_run_patch(train_lowe[i], ev, lowe_run_updates, args) for i, ev in enumerate(train)
    }
    target_lowe_run = [
        v.apply_lowe_run_patch(target_lowe[i], ev, lowe_run_updates, args) for i, ev in enumerate(target)
    ]

    seed_updates, rows = v.derive_seed_updates_regularized(train, train_lowe_run, args)
    update_rows.extend({"fold": fold, "full_iteration": pass_index, **row} for row in rows)

    train_final = [base.apply_seed(train_lowe_run[i], ev, seed_updates) for i, ev in enumerate(train)]
    target_final = [base.apply_seed(target_lowe_run[i], ev, seed_updates) for i, ev in enumerate(target)]
    return train_final, target_final, update_rows


def model_names(args: argparse.Namespace) -> list[str]:
    return ["hao_raw"] + [f"full_iter_{i}" for i in range(1, max(1, args.full_iterations) + 1)]


def fit_eval_rows(
    fold: str,
    sample: str,
    events: list[dict[str, object]],
    masses_by_model: dict[str, list[float]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, list[int]]] = [("pooled", list(range(len(events))))]
    for run in sorted({int(ev["run"]) for ev in events}):
        groups.append((f"run_{run}", [i for i, ev in enumerate(events) if int(ev["run"]) == run]))
    for eb in base.ENERGY_BINS:
        groups.append((f"emin_{base.bin_label(eb)}", [i for i, ev in enumerate(events) if base.energy_bin(float(ev["min_e"])) == eb]))
    if args.fit_detail == "run_energy":
        for eb in base.ENERGY_BINS:
            label = base.bin_label(eb)
            for run in sorted({int(ev["run"]) for ev in events}):
                groups.append(
                    (
                        f"run_{run}_emin_{label}",
                        [
                            i
                            for i, ev in enumerate(events)
                            if int(ev["run"]) == run and base.energy_bin(float(ev["min_e"])) == eb
                        ],
                    )
                )

    for group, idxs in groups:
        if not idxs:
            continue
        for model in model_names(args):
            vals = np.asarray([masses_by_model[model][i] for i in idxs], dtype=float)
            fit = fit_peak(vals, args.fit_window[0], args.fit_window[1])
            rows.append(
                {
                    "fold": fold,
                    "sample": sample,
                    "group": group,
                    "model": model,
                    "entries": len(vals),
                    "fit_entries": int(fit["entries"]),
                    "fit_ok": int(fit["fit_ok"]),
                    "fit_mu_mev": fit["mu"],
                    "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
                    "fit_sigma_mev": fit["sigma"],
                    "chi2_ndf": fit.get("chi2_ndf", math.nan),
                }
            )
    return rows


def evaluate_fold(
    fold_name: str,
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]]]:
    train_current = [float(ev["mass_mev"]) for ev in train]
    test_current = [float(ev["mass_mev"]) for ev in test]
    masses: dict[str, list[float]] = {"hao_raw": test_current[:]}
    update_rows: list[dict[str, object]] = []

    for pass_index in range(1, max(1, args.full_iterations) + 1):
        train_current, test_current, rows = apply_full_pass(
            pass_index, fold_name, train, train_current, test, test_current, args
        )
        update_rows.extend(rows)
        masses[f"full_iter_{pass_index}"] = test_current[:]

    fit_rows = fit_eval_rows(fold_name, "test", test, masses, args)
    return fit_rows, update_rows, masses


def plot_summary(rows: list[dict[str, object]], outdir: Path, args: argparse.Namespace) -> None:
    pdf = outdir / "hao_full_layer_iteration_comparison.pdf"
    groups = ["pooled"] + [f"emin_{base.bin_label(eb)}" for eb in base.ENERGY_BINS]
    labels = {
        "hao_raw": "Hao baseline",
        **{f"full_iter_{i}": f"full pass {i}" for i in range(1, max(1, args.full_iterations) + 1)},
    }
    colors = ["#777777", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    with PdfPages(pdf) as pp:
        for group in groups:
            fig, ax = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
            for idx, model in enumerate(model_names(args)):
                vals = [
                    r
                    for r in rows
                    if r["fold"] == "pooled_folds"
                    and r["sample"] == "test"
                    and r["group"] == group
                    and r["model"] == model
                    and int(r["fit_ok"]) == 1
                ]
                if not vals:
                    continue
                r = vals[0]
                ax[0].scatter([idx], [float(r["fit_mu_residual_mev"])], s=80, color=colors[idx % len(colors)], label=labels[model])
                ax[1].scatter([idx], [float(r["fit_sigma_mev"])], s=80, color=colors[idx % len(colors)], label=labels[model])
            ax[0].axhline(0.0, color="black", lw=0.8)
            ax[0].set_ylabel("Gaussian peak residual (MeV)")
            ax[1].set_ylabel("Gaussian sigma (MeV)")
            ax[1].set_xticks(range(len(model_names(args))))
            ax[1].set_xticklabels([labels[m] for m in model_names(args)], rotation=25, ha="right")
            ax[0].set_title(f"Full-layer iteration held-out fits: {group}")
            ax[0].legend(ncol=2, fontsize=8)
            pp.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, constrained_layout=True)
        e_groups = [f"emin_{base.bin_label(eb)}" for eb in base.ENERGY_BINS]
        xlabels = [g.replace("emin_", "") for g in e_groups]
        for idx, model in enumerate(model_names(args)):
            mus = []
            sigmas = []
            for group in e_groups:
                vals = [
                    r
                    for r in rows
                    if r["fold"] == "pooled_folds" and r["group"] == group and r["model"] == model and int(r["fit_ok"]) == 1
                ]
                mus.append(float(vals[0]["fit_mu_residual_mev"]) if vals else math.nan)
                sigmas.append(float(vals[0]["fit_sigma_mev"]) if vals else math.nan)
            ax[0].plot(xlabels, mus, "-o", label=labels[model], color=colors[idx % len(colors)])
            ax[1].plot(xlabels, sigmas, "-o", label=labels[model], color=colors[idx % len(colors)])
        ax[0].axhline(0.0, color="black", lw=0.8)
        ax[0].set_ylabel("Gaussian peak residual (MeV)")
        ax[0].set_title("Energy-bin closure across full-layer passes")
        ax[1].set_ylabel("Gaussian sigma (MeV)")
        ax[1].set_xlabel("min photon energy bin (GeV)")
        ax[0].legend(ncol=2, fontsize=8)
        pp.savefig(fig)
        plt.close(fig)

        for e_group in e_groups:
            fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True, constrained_layout=True)
            for idx, model in enumerate(model_names(args)):
                run_rows = []
                for r in rows:
                    group = str(r["group"])
                    if (
                        r["fold"] == "pooled_folds"
                        and group.startswith("run_")
                        and group.endswith(f"_{e_group}")
                        and r["model"] == model
                        and int(r["fit_ok"]) == 1
                    ):
                        parts = group.split("_")
                        run_rows.append((int(parts[1]), float(r["fit_mu_residual_mev"]), float(r["fit_sigma_mev"])))
                if not run_rows:
                    continue
                run_rows.sort()
                ax[0].plot([x[0] for x in run_rows], [x[1] for x in run_rows], "-o", ms=3, label=labels[model], color=colors[idx % len(colors)])
                ax[1].plot([x[0] for x in run_rows], [x[2] for x in run_rows], "-o", ms=3, label=labels[model], color=colors[idx % len(colors)])
            ax[0].axhline(0.0, color="black", lw=0.8)
            ax[0].set_ylabel("Gaussian peak residual (MeV)")
            ax[0].set_title(f"Run trend: {e_group.replace('emin_', '')} GeV")
            ax[1].set_ylabel("Gaussian sigma (MeV)")
            ax[1].set_xlabel("run")
            ax[0].legend(ncol=2, fontsize=8)
            pp.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = base.load_conversion_summary(Path(args.conversion_summary))
    events, qa = v.read_compact_events(summary, args)
    write_tsv(outdir / "sidecar_quality.tsv", qa)

    all_fit_rows: list[dict[str, object]] = []
    all_update_rows: list[dict[str, object]] = []
    pooled_events: list[dict[str, object]] = []
    pooled_masses: dict[str, list[float]] = {model: [] for model in model_names(args)}

    for test_fold in [0, 1]:
        train = [ev for ev in events if int(ev["fold"]) != test_fold]
        test = [ev for ev in events if int(ev["fold"]) == test_fold]
        fold_name = f"segment_parity_test_{test_fold}"
        fit_rows, update_rows, masses = evaluate_fold(fold_name, train, test, args)
        all_fit_rows.extend(fit_rows)
        all_update_rows.extend(update_rows)
        pooled_events.extend(test)
        for model in model_names(args):
            pooled_masses[model].extend(masses[model])

    all_fit_rows.extend(fit_eval_rows("pooled_folds", "test", pooled_events, pooled_masses, args))
    write_tsv(outdir / "heldout_fit_metrics.tsv", all_fit_rows)
    write_tsv(outdir / "fold_update_tables.tsv", all_update_rows)

    pooled_summary = [
        r
        for r in all_fit_rows
        if r["fold"] == "pooled_folds"
        and r["sample"] == "test"
        and (r["group"] == "pooled" or str(r["group"]).startswith("emin_"))
    ]
    write_tsv(outdir / "pooled_energy_summary.tsv", pooled_summary)

    if not args.skip_plots:
        plot_summary(all_fit_rows, outdir, args)

    lines = ["Hao-start full-layer iteration check", ""]
    for r in pooled_summary:
        if r["group"] == "pooled":
            lines.append(
                f"{r['model']:14s} pooled mu_res={float(r['fit_mu_residual_mev']):+.6f} MeV "
                f"sigma={float(r['fit_sigma_mev']):.6f} MeV fit_entries={r['fit_entries']}"
            )
    lines.append("")
    lines.append("Interpretation rule: promote a later full pass only if it improves Gaussian sigma and energy-bin closure without high-E drift.")
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(outdir / "summary.txt")


if __name__ == "__main__":
    main()
