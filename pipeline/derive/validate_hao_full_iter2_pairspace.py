#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import validate_hao_full_layer_iteration as fulliter
import validate_hao_period_energy_smoothed_seed as v
import validate_hao_start_layered_member_stack as base
from build_energy_correction_stack import M_PI0_MEV
from plot_period_fallback_mgg_fits import fit_peak


PAIR_BINS = [(0.6, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 2.5)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply pair-space residual layer on top of Hao-start full_iter2.")
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[105.0, 165.0])
    ap.add_argument("--min-fit-entries", type=int, default=200)
    ap.add_argument("--min-seed-events", type=int, default=20)
    ap.add_argument("--min-pairspace-entries", type=int, default=300)
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


def pair_bin(e: float) -> tuple[float, float] | None:
    for lo, hi in PAIR_BINS:
        if lo <= e < hi or (math.isclose(hi, 2.5) and lo <= e <= hi):
            return (lo, hi)
    return None


def pair_label(eb: tuple[float, float]) -> str:
    return f"{eb[0]:.1f}-{eb[1]:.1f}"


def event_pair_key(ev: dict[str, object]) -> str | None:
    e1 = float(ev["e1"])
    e2 = float(ev["e2"])
    b1 = pair_bin(min(e1, e2))
    b2 = pair_bin(max(e1, e2))
    if b1 is None or b2 is None:
        return None
    return f"{pair_label(b1)}|{pair_label(b2)}"


def read_events(summary_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    qa: list[dict[str, object]] = []
    for info in summary_rows:
        compact = Path(info["output_tsv"])
        total = selected = 0
        with compact.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                total += 1
                if int(row.get("nclusters", "0")) != 2:
                    continue
                e1 = float(row["e1"])
                e2 = float(row["e2"])
                min_e = min(e1, e2)
                if min_e < args.emin_range[0] or min_e > args.emin_range[1]:
                    continue
                if args.timing_center is not None and abs(float(row["dt12"]) - args.timing_center) > args.timing_window:
                    continue
                low_slot = 0 if e1 <= e2 else 1
                events.append(
                    {
                        "run": int(row["run"]),
                        "segment": int(info["segment"]),
                        "event": int(row["event"]),
                        "fold": int(info["segment"]) % 2,
                        "mass_mev": float(row["pair_m"]) * 1000.0,
                        "e1": e1,
                        "e2": e2,
                        "min_e": min_e,
                        "max_e": max(e1, e2),
                        "seed1_block": int(row["seed1_block"]),
                        "seed2_block": int(row["seed2_block"]),
                        "seed1_col": int(row["seed1_col"]),
                        "seed2_col": int(row["seed2_col"]),
                        "low_seed": int(row["seed1_block"]) if low_slot == 0 else int(row["seed2_block"]),
                        "low_col": int(row["seed1_col"]) if low_slot == 0 else int(row["seed2_col"]),
                        "members1": [],
                        "members2": [],
                    }
                )
                selected += 1
        qa.append({"run": info["run"], "segment": info["segment"], "total_rows": total, "selected_events": selected})
    return events, qa


def derive_full_iter2(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[list[float], list[float], list[dict[str, object]]]:
    train_current = [float(ev["mass_mev"]) for ev in train]
    test_current = [float(ev["mass_mev"]) for ev in test]
    all_rows: list[dict[str, object]] = []
    for pass_index in [1, 2]:
        train_current, test_current, rows = fulliter.apply_full_pass(
            pass_index,
            "pairspace_base",
            train,
            train_current,
            test,
            test_current,
            args,
        )
        all_rows.extend(rows)
    return train_current, test_current, all_rows


def derive_pairspace_updates(
    train: list[dict[str, object]],
    masses: list[float],
    candidate: str,
    run_min: int,
    damping: float,
    clip: float,
    key_prefix: str | None,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for ev, mass in zip(train, masses):
        if int(ev["run"]) < run_min:
            continue
        key = event_pair_key(ev)
        if key is None:
            continue
        if key_prefix is not None and not key.startswith(key_prefix):
            continue
        by_pair[key].append(float(mass))
    updates: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for key, vals in sorted(by_pair.items()):
        fit = fit_peak(np.asarray(vals, dtype=float), args.fit_window[0], args.fit_window[1])
        ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_pairspace_entries and math.isfinite(float(fit["mu"]))
        full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
        applied = base.clamp(full**damping, clip) if ok else 1.0
        rows.append(
            {
                "candidate": candidate,
                "run_min": run_min,
                "pair_bin": key,
                "entries": len(vals),
                "fit_entries": int(fit["entries"]),
                "fit_ok": int(fit["fit_ok"]),
                "fit_mu_mev": fit["mu"],
                "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
                "fit_sigma_mev": fit["sigma"],
                "full_update": full,
                "damping": damping,
                "clip_frac": clip,
                "key_prefix": key_prefix or "",
                "applied_update": applied,
            }
        )
        if ok:
            updates[key] = applied
    return updates, rows


def apply_pairspace(
    mass: float,
    ev: dict[str, object],
    updates: dict[str, float],
    run_min: int,
    edge_cols: set[int],
    key_prefix: str | None,
) -> float:
    if int(ev["run"]) < run_min:
        return mass
    if int(ev["seed1_col"]) in edge_cols or int(ev["seed2_col"]) in edge_cols:
        return mass
    key = event_pair_key(ev)
    if key is None:
        return mass
    if key_prefix is not None and not key.startswith(key_prefix):
        return mass
    return mass * updates.get(key, 1.0)


def model_names() -> list[str]:
    return [
        "full_iter2",
        "pair_late_d050",
        "pair_all_d025",
        "pair_all_d050",
        "pair_lowE_late_d050",
        "pair_lowE_all_d025",
        "pair_lowE_all_d050",
    ]


def fit_rows(
    fold: str,
    events: list[dict[str, object]],
    masses: dict[str, list[float]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    groups: list[tuple[str, list[int]]] = [("pooled", list(range(len(events))))]
    for run in sorted({int(ev["run"]) for ev in events}):
        groups.append((f"run_{run}", [i for i, ev in enumerate(events) if int(ev["run"]) == run]))
    for eb in base.ENERGY_BINS:
        groups.append((f"emin_{base.bin_label(eb)}", [i for i, ev in enumerate(events) if base.energy_bin(float(ev["min_e"])) == eb]))
        for run in sorted({int(ev["run"]) for ev in events}):
            groups.append(
                (
                    f"run_{run}_emin_{base.bin_label(eb)}",
                    [i for i, ev in enumerate(events) if int(ev["run"]) == run and base.energy_bin(float(ev["min_e"])) == eb],
                )
            )
    rows: list[dict[str, object]] = []
    for group, idxs in groups:
        if not idxs:
            continue
        for model in model_names():
            vals = np.asarray([masses[model][i] for i in idxs], dtype=float)
            fit = fit_peak(vals, args.fit_window[0], args.fit_window[1])
            rows.append(
                {
                    "fold": fold,
                    "sample": "test",
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


def plot_runrange(rows: list[dict[str, object]], outdir: Path) -> None:
    colors = {
        "full_iter2": "#444444",
        "pair_late_d050": "#1f77b4",
        "pair_all_d025": "#2ca02c",
        "pair_all_d050": "#d62728",
        "pair_lowE_late_d050": "#17becf",
        "pair_lowE_all_d025": "#bcbd22",
        "pair_lowE_all_d050": "#ff7f0e",
    }
    labels = {
        "full_iter2": "full iter2",
        "pair_late_d050": "late pair d0.50",
        "pair_all_d025": "all pair d0.25",
        "pair_all_d050": "all pair d0.50",
        "pair_lowE_late_d050": "lowE late pair d0.50",
        "pair_lowE_all_d025": "lowE all pair d0.25",
        "pair_lowE_all_d050": "lowE all pair d0.50",
    }
    markers = {
        "full_iter2": "o",
        "pair_late_d050": "s",
        "pair_all_d025": "^",
        "pair_all_d050": "D",
        "pair_lowE_late_d050": "P",
        "pair_lowE_all_d025": "X",
        "pair_lowE_all_d050": "v",
    }
    egroups = [f"emin_{base.bin_label(eb)}" for eb in base.ENERGY_BINS]
    pdf = outdir / "hao_full_iter2_pairspace_runrange.pdf"
    with PdfPages(pdf) as pp:
        fig, axs = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
        for ax, eg in zip(axs, egroups[:2]):
            for model in model_names():
                rr = [
                    r
                    for r in rows
                    if r["fold"] == "pooled_folds"
                    and str(r["group"]).startswith("run_")
                    and str(r["group"]).endswith(f"_{eg}")
                    and r["model"] == model
                    and int(r["fit_ok"]) == 1
                ]
                rr.sort(key=lambda r: int(str(r["group"]).split("_")[1]))
                ax.plot(
                    [int(str(r["group"]).split("_")[1]) for r in rr],
                    [float(r["fit_mu_residual_mev"]) for r in rr],
                    marker=markers[model],
                    ms=3,
                    lw=1.1,
                    color=colors[model],
                    label=labels[model],
                )
            ax.axhline(0.0, color="black", lw=0.8)
            ax.axhspan(-0.5, 0.5, color="grey", alpha=0.10)
            ax.set_ylabel("fit peak residual (MeV)")
            ax.set_title(f"{eg.replace('emin_', '')} GeV run-range residual")
        axs[0].legend(ncol=2, fontsize=8)
        axs[-1].set_xlabel("run")
        pp.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
        for ax, eg in zip(axs.flat, egroups):
            for model in model_names():
                rr = [
                    r
                    for r in rows
                    if r["fold"] == "pooled_folds"
                    and str(r["group"]).startswith("run_")
                    and str(r["group"]).endswith(f"_{eg}")
                    and r["model"] == model
                    and int(r["fit_ok"]) == 1
                ]
                rr.sort(key=lambda r: int(str(r["group"]).split("_")[1]))
                ax.plot(
                    [int(str(r["group"]).split("_")[1]) for r in rr],
                    [float(r["fit_mu_residual_mev"]) for r in rr],
                    marker=markers[model],
                    ms=2.5,
                    lw=1.0,
                    color=colors[model],
                    label=labels[model],
                )
            ax.axhline(0.0, color="black", lw=0.7)
            ax.axhspan(-0.5, 0.5, color="grey", alpha=0.10)
            ax.set_title(eg.replace("emin_", "") + " GeV")
            ax.set_ylabel("resid (MeV)")
        axs.flat[0].legend(ncol=2, fontsize=8)
        for ax in axs[-1, :]:
            ax.set_xlabel("run")
        pp.savefig(fig)
        plt.close(fig)


def summarize_run_residuals(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for eg in [f"emin_{base.bin_label(eb)}" for eb in base.ENERGY_BINS]:
        for model in model_names():
            vals = []
            for r in rows:
                if (
                    r["fold"] == "pooled_folds"
                    and str(r["group"]).startswith("run_")
                    and str(r["group"]).endswith(f"_{eg}")
                    and r["model"] == model
                    and int(r["fit_ok"]) == 1
                ):
                    vals.append(abs(float(r["fit_mu_residual_mev"])))
            vals.sort()
            if not vals:
                continue
            out.append(
                {
                    "energy_group": eg,
                    "model": model,
                    "nrun": len(vals),
                    "median_abs_mu_residual_mev": statistics.median(vals),
                    "p95_abs_mu_residual_mev": vals[int(0.95 * (len(vals) - 1))],
                    "max_abs_mu_residual_mev": max(vals),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = base.load_conversion_summary(Path(args.conversion_summary))
    events, qa = read_events(summary, args)
    write_tsv(outdir / "sidecar_quality.tsv", qa)

    all_fit_rows: list[dict[str, object]] = []
    all_update_rows: list[dict[str, object]] = []
    pooled_events: list[dict[str, object]] = []
    pooled_masses: dict[str, list[float]] = {model: [] for model in model_names()}

    candidates = {
        "pair_late_d050": {"run_min": 4551, "damping": 0.50, "clip": 0.003, "key_prefix": None},
        "pair_all_d025": {"run_min": 0, "damping": 0.25, "clip": 0.003, "key_prefix": None},
        "pair_all_d050": {"run_min": 0, "damping": 0.50, "clip": 0.003, "key_prefix": None},
        "pair_lowE_late_d050": {"run_min": 4551, "damping": 0.50, "clip": 0.003, "key_prefix": "0.6-0.8|"},
        "pair_lowE_all_d025": {"run_min": 0, "damping": 0.25, "clip": 0.003, "key_prefix": "0.6-0.8|"},
        "pair_lowE_all_d050": {"run_min": 0, "damping": 0.50, "clip": 0.003, "key_prefix": "0.6-0.8|"},
    }
    edge_cols = set(args.left_cols)
    for test_fold in [0, 1]:
        train = [ev for ev in events if int(ev["fold"]) != test_fold]
        test = [ev for ev in events if int(ev["fold"]) == test_fold]
        fold = f"segment_parity_test_{test_fold}"
        train_base, test_base, rows = derive_full_iter2(train, test, args)
        all_update_rows.extend({"fold": fold, "stage": "full_iter2", **row} for row in rows)

        masses: dict[str, list[float]] = {"full_iter2": test_base[:]}
        for name, cfg in candidates.items():
            updates, rows = derive_pairspace_updates(
                train,
                train_base,
                name,
                int(cfg["run_min"]),
                float(cfg["damping"]),
                float(cfg["clip"]),
                str(cfg["key_prefix"]) if cfg["key_prefix"] is not None else None,
                args,
            )
            all_update_rows.extend({"fold": fold, "stage": "pairspace", **row} for row in rows)
            masses[name] = [
                apply_pairspace(
                    test_base[i],
                    ev,
                    updates,
                    int(cfg["run_min"]),
                    edge_cols,
                    str(cfg["key_prefix"]) if cfg["key_prefix"] is not None else None,
                )
                for i, ev in enumerate(test)
            ]

        all_fit_rows.extend(fit_rows(fold, test, masses, args))
        pooled_events.extend(test)
        for model in model_names():
            pooled_masses[model].extend(masses[model])

    all_fit_rows.extend(fit_rows("pooled_folds", pooled_events, pooled_masses, args))
    write_tsv(outdir / "heldout_fit_metrics.tsv", all_fit_rows)
    write_tsv(outdir / "fold_update_tables.tsv", all_update_rows)
    run_summary = summarize_run_residuals(all_fit_rows)
    write_tsv(outdir / "run_residual_stability_summary.tsv", run_summary)
    plot_runrange(all_fit_rows, outdir)
    print(outdir / "heldout_fit_metrics.tsv")
    print(outdir / "run_residual_stability_summary.tsv")
    print(outdir / "hao_full_iter2_pairspace_runrange.pdf")


if __name__ == "__main__":
    main()
