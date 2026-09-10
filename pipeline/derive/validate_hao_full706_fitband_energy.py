#!/usr/bin/env python3
"""Hao-start full_iter2 plus Gaussian-fit run-band x energy-bin residual scan."""

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

import validate_hao_full_iter2_pairspace as pairspace
import validate_hao_period_energy_smoothed_seed as v
import validate_hao_start_layered_member_stack as base
from build_energy_correction_stack import M_PI0_MEV
from plot_period_fallback_mgg_fits import fit_peak


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


def candidate_specs() -> list[dict[str, object]]:
    return [
        {"model": "full_iter2", "band_mode": "none", "damping": 0.0},
        {"model": "fitBand_fixed4_d050", "band_mode": "fixed4", "damping": 0.50},
        {"model": "fitBand_fixed4_d075", "band_mode": "fixed4", "damping": 0.75},
        {"model": "fitBand_stat8_d050", "band_mode": "stat8", "damping": 0.50},
        {"model": "fitBand_stat8_d075", "band_mode": "stat8", "damping": 0.75},
        {"model": "fitBand_stat12_d075", "band_mode": "stat12", "damping": 0.75},
    ]


def model_names() -> list[str]:
    return [str(spec["model"]) for spec in candidate_specs()]


def fixed4_band(run: int) -> str:
    if run <= 4299:
        return "early_le4299"
    if run <= 4415:
        return "mid_4300_4415"
    if run <= 4450:
        return "transition_4417_4450"
    return "late_gt4450"


def build_band_map(train: list[dict[str, object]], mode: str) -> dict[int, str]:
    if mode == "fixed4":
        return {run: fixed4_band(run) for run in sorted({int(ev["run"]) for ev in train})}
    if mode.startswith("stat"):
        n_bands = int(mode[4:])
        return v.build_stat_run_bands(train, n_bands)
    raise ValueError(f"unknown band mode {mode}")


def nearest_band(run: int, band_map: dict[int, str]) -> str:
    if run in band_map:
        return band_map[run]
    nearest = min(band_map, key=lambda r: abs(r - run))
    return band_map[nearest]


def ebin_label(ev: dict[str, object]) -> str | None:
    eb = base.energy_bin(float(ev["min_e"]))
    return base.bin_label(eb) if eb is not None else None


def derive_fitband_updates(
    fold: str,
    train: list[dict[str, object]],
    train_masses: list[float],
    model: str,
    band_mode: str,
    damping: float,
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], dict[int, str], list[dict[str, object]]]:
    band_map = build_band_map(train, band_mode)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ev, mass in zip(train, train_masses):
        label = ebin_label(ev)
        if label is None:
            continue
        grouped[(nearest_band(int(ev["run"]), band_map), label)].append(float(mass))

    updates: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    for key, vals in sorted(grouped.items()):
        fit = fit_peak(np.asarray(vals, dtype=float), args.fit_window[0], args.fit_window[1])
        ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_band_fit_entries and math.isfinite(float(fit["mu"]))
        full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
        applied = base.clamp(full**damping, args.max_fitband_mass_update_frac) if ok else 1.0
        row = {
            "fold": fold,
            "model": model,
            "band_mode": band_mode,
            "band": key[0],
            "energy_bin": key[1],
            "entries": len(vals),
            "fit_entries": int(fit["entries"]),
            "fit_ok": int(fit["fit_ok"]),
            "support_pass": int(ok),
            "fit_mu_mev": fit["mu"],
            "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
            "fit_sigma_mev": fit["sigma"],
            "factor_full": full,
            "damping": damping,
            "clip_frac": args.max_fitband_mass_update_frac,
            "factor_applied": applied,
        }
        rows.append(row)
        if ok:
            updates[key] = applied
    return updates, band_map, rows


def apply_fitband(
    mass: float,
    ev: dict[str, object],
    updates: dict[tuple[str, str], float],
    band_map: dict[int, str],
) -> float:
    label = ebin_label(ev)
    if label is None or not band_map:
        return mass
    key = (nearest_band(int(ev["run"]), band_map), label)
    return float(mass) * updates.get(key, 1.0)


def fit_eval_rows(
    fold: str,
    events: list[dict[str, object]],
    masses: dict[str, list[float]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
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


def pooled_rows(events: list[dict[str, object]], masses: dict[str, list[float]], args: argparse.Namespace) -> list[dict[str, object]]:
    rows = fit_eval_rows("pooled_folds", events, masses, args)
    for row in rows:
        row["fold"] = "pooled_folds"
    return rows


def stability_summary(rows: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for eb in base.ENERGY_BINS:
        group_suffix = f"_emin_{base.bin_label(eb)}"
        energy_group = f"emin_{base.bin_label(eb)}"
        for model in model_names():
            vals = [
                abs(float(row["fit_mu_residual_mev"]))
                for row in rows
                if row["fold"] == "pooled_folds"
                and row["model"] == model
                and str(row["group"]).startswith("run_")
                and str(row["group"]).endswith(group_suffix)
                and int(row["fit_ok"]) == 1
                and int(row["fit_entries"]) >= args.min_fit_entries
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out.append(
                {
                    "energy_group": energy_group,
                    "model": model,
                    "nrun": len(vals),
                    "median_abs_mu_residual_mev": float(np.median(arr)),
                    "p95_abs_mu_residual_mev": float(np.percentile(arr, 95)),
                    "max_abs_mu_residual_mev": float(np.max(arr)),
                }
            )
    return out


def plot_pdf(rows: list[dict[str, object]], stability: list[dict[str, object]], outdir: Path) -> None:
    pdf = outdir / "hao_full706_fitband_energy_scan.pdf"
    colors = {
        "full_iter2": "#444444",
        "fitBand_fixed4_d050": "#4C78A8",
        "fitBand_fixed4_d075": "#F58518",
        "fitBand_stat8_d050": "#54A24B",
        "fitBand_stat8_d075": "#E45756",
        "fitBand_stat12_d075": "#B279A2",
    }
    labels = {
        "full_iter2": "full iter2",
        "fitBand_fixed4_d050": "fixed4 d0.50",
        "fitBand_fixed4_d075": "fixed4 d0.75",
        "fitBand_stat8_d050": "stat8 d0.50",
        "fitBand_stat8_d075": "stat8 d0.75",
        "fitBand_stat12_d075": "stat12 d0.75",
    }
    pooled = {
        row["model"]: row
        for row in rows
        if row["fold"] == "pooled_folds" and row["group"] == "pooled" and row["model"] in model_names()
    }
    with PdfPages(pdf) as pp:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        models = model_names()
        x = np.arange(len(models))
        ax[0].bar(x, [float(pooled[m]["fit_mu_residual_mev"]) for m in models], color=[colors[m] for m in models])
        ax[0].axhline(0, color="black", lw=1)
        ax[0].set_ylabel("fit mean residual (MeV)")
        ax[0].set_title("Pooled held-out mean")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([labels[m] for m in models], rotation=25, ha="right")
        ax[1].bar(x, [float(pooled[m]["fit_sigma_mev"]) for m in models], color=[colors[m] for m in models])
        ax[1].set_ylabel("fit sigma (MeV)")
        ax[1].set_title("Pooled held-out sigma")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels([labels[m] for m in models], rotation=25, ha="right")
        pp.savefig(fig)
        plt.close(fig)

        for energy_group in ["emin_0.6-0.8", "emin_0.8-1.0"]:
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            for model in models:
                vals = []
                runs = []
                for row in rows:
                    group = str(row["group"])
                    if (
                        row["fold"] == "pooled_folds"
                        and row["model"] == model
                        and group.startswith("run_")
                        and group.endswith("_" + energy_group)
                        and int(row["fit_ok"]) == 1
                    ):
                        runs.append(int(group.split("_")[1]))
                        vals.append(float(row["fit_mu_residual_mev"]))
                order = np.argsort(runs)
                ax.plot(np.asarray(runs)[order], np.asarray(vals)[order], marker="o", ms=2.5, lw=1, label=labels[model], color=colors[model])
            ax.axhline(0, color="black", lw=1)
            ax.axhline(1, color="gray", lw=0.8, ls="--")
            ax.axhline(-1, color="gray", lw=0.8, ls="--")
            ax.set_title(f"{energy_group} fitted peak residual vs run")
            ax.set_xlabel("run")
            ax.set_ylabel("fit mean residual (MeV)")
            ax.legend(fontsize=8, ncol=2)
            pp.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        sub = [row for row in stability if row["energy_group"] == "emin_0.6-0.8"]
        by = {row["model"]: row for row in sub}
        w = 0.25
        x = np.arange(len(models))
        ax.bar(x - w, [float(by[m]["median_abs_mu_residual_mev"]) for m in models], w, label="median")
        ax.bar(x, [float(by[m]["p95_abs_mu_residual_mev"]) for m in models], w, label="p95")
        ax.bar(x + w, [float(by[m]["max_abs_mu_residual_mev"]) for m in models], w, label="max")
        ax.set_title("0.6-0.8 run residual stability")
        ax.set_ylabel("|fit mean residual| (MeV)")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in models], rotation=25, ha="right")
        ax.legend()
        pp.savefig(fig)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = base.load_conversion_summary(Path(args.conversion_summary))
    events, qa = pairspace.read_events(summary_rows, args)
    write_tsv(outdir / "sidecar_quality.tsv", qa)

    all_test_events: list[dict[str, object]] = []
    all_masses: dict[str, list[float]] = {model: [] for model in model_names()}
    fit_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []

    for fold_value in [0, 1]:
        fold = f"segment_parity_test_{fold_value}"
        train = [ev for ev in events if int(ev["fold"]) != fold_value]
        test = [ev for ev in events if int(ev["fold"]) == fold_value]
        train_base, test_base, full_rows = pairspace.derive_full_iter2(train, test, args)
        update_rows.extend({"fold": fold, "model": "full_iter2", **row} for row in full_rows)

        masses: dict[str, list[float]] = {"full_iter2": test_base[:]}
        for spec in candidate_specs():
            model = str(spec["model"])
            if model == "full_iter2":
                continue
            updates, band_map, rows = derive_fitband_updates(
                fold, train, train_base, model, str(spec["band_mode"]), float(spec["damping"]), args
            )
            update_rows.extend(rows)
            masses[model] = [apply_fitband(mass, ev, updates, band_map) for ev, mass in zip(test, test_base)]

        fit_rows.extend(fit_eval_rows(fold, test, masses, args))
        all_test_events.extend(test)
        for model in model_names():
            all_masses[model].extend(masses[model])

    fit_rows.extend(pooled_rows(all_test_events, all_masses, args))
    stability = stability_summary(fit_rows, args)
    write_tsv(outdir / "fitband_update_table.tsv", update_rows)
    write_tsv(outdir / "heldout_fit_metrics.tsv", fit_rows)
    write_tsv(outdir / "run_residual_stability_summary.tsv", stability)
    plot_pdf(fit_rows, stability, outdir)
    print(outdir)
    print(outdir / "hao_full706_fitband_energy_scan.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
