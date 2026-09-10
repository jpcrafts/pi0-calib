#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

from build_energy_correction_stack import M_PI0_MEV, metrics
from validate_frozen_shared_time_block_granularity import load_coeff, load_time_slices
from validate_member_on_period_seed_fallback import (
    apply_stack,
    default_chunk_summaries,
    derive_member_updates,
    derive_seed_scales,
    event_time_scale,
    period_for_run,
    split_folds,
)
from validate_member_weighted_block_stack import (
    input_paths_from_summary,
    load_member_index,
    read_rows,
    rows_to_member_events,
    select_raw_rows,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Make slide-ready invariant-mass distributions with Gaussian-style peak fits for "
            "the current period/fallback seed policy and optional member residual layer."
        )
    )
    ap.add_argument("--chunk-summary", action="append")
    ap.add_argument("--output-dir", default="output/x60_4_partial_68/period_fallback_mgg_fit_reference")
    ap.add_argument("--coeff-tsv", default="output/x60_4_partial_68/sigma_window_2p0_working_exact2_t1p0_emin0p6/shared_curve/shared_curve_coeffs.tsv")
    ap.add_argument("--time-slice-tsv", default="output/x60_4_partial_68/correction_stack_working/time_slice_scales.tsv")
    ap.add_argument("--derive-mass-window", nargs=2, type=float, default=[0.115961, 0.138212])
    ap.add_argument("--plot-mass-window", nargs=2, type=float, default=[0.08, 0.20])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[115.0, 150.0])
    ap.add_argument("--hist-window", nargs=2, type=float, default=[80.0, 200.0])
    ap.add_argument("--timing-center", type=float, default=-0.176158)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--min-e-floor", type=float, default=0.6)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--min-seed-entries", type=int, default=50)
    ap.add_argument("--min-member-entries", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--damping", type=float, default=0.75)
    ap.add_argument("--max-update-frac", type=float, default=0.0)
    ap.add_argument("--left-fallback-cols", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rtol", type=float, default=1.0e-5)
    ap.add_argument("--atol", type=float, default=1.0e-5)
    ap.add_argument("--cache-npz", default=None, help="Optional mass-array cache. Defaults to output-dir/mgg_stage_masses.npz.")
    ap.add_argument("--force-rebuild-cache", action="store_true", help="Ignore an existing mass-array cache and rebuild from sidecars.")
    return ap.parse_args()


def clip_update(update: float, max_update_frac: float) -> float:
    if max_update_frac <= 0.0:
        return update
    return min(max(update, 1.0 - max_update_frac), 1.0 + max_update_frac)


def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0.0:
        return np.zeros_like(x)
    z = (x - mu) / sigma
    return amp * np.exp(-0.5 * z * z)


def gaussian_const(x: np.ndarray, amp: float, mu: float, sigma: float, const: float) -> np.ndarray:
    return gaussian(x, amp, mu, sigma) + const


def fit_peak(masses: np.ndarray, lo: float, hi: float) -> dict[str, float]:
    work = masses[(masses >= lo) & (masses <= hi)]
    if work.size < 50:
        return {"entries": float(work.size), "mu": math.nan, "sigma": math.nan, "amp": math.nan, "const": math.nan, "fit_ok": 0.0}

    fit_bins = 80
    hist, edges = np.histogram(work, bins=fit_bins, range=(lo, hi))
    fit_bin_width = (hi - lo) / fit_bins
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_bin = int(np.argmax(hist))
    edge_sample = np.r_[hist[:10], hist[-10:]]
    const0 = max(0.0, float(np.median(edge_sample))) if edge_sample.size else 0.0
    amp0 = max(1.0, float(hist[peak_bin]) - const0)
    mu0 = float(centers[peak_bin])
    sigma0 = 4.5

    try:
        popt, _ = curve_fit(
            gaussian_const,
            centers,
            hist.astype(float),
            p0=[amp0, mu0, sigma0, const0],
            sigma=np.sqrt(np.maximum(hist.astype(float), 1.0)),
            absolute_sigma=True,
            bounds=([0.0, lo, 1.0, 0.0], [np.inf, hi, 20.0, np.inf]),
            maxfev=20000,
        )
        amp, mu, sigma, const = [float(v) for v in popt]
        fit_ok = 1.0
    except Exception:
        # Fall back to a local moment estimate. The summary flags this as not a fit.
        above = np.maximum(hist.astype(float) - const0, 0.0)
        total = float(np.sum(above))
        if total > 0.0:
            mu = float(np.sum(centers * above) / total)
            sigma = float(np.sqrt(np.sum(above * (centers - mu) ** 2) / total))
        else:
            mu = mu0
            sigma = sigma0
        amp = amp0
        const = const0
        fit_ok = 0.0

    fit_vals = gaussian_const(centers, amp, mu, sigma, const)
    chi2 = float(np.sum((hist - fit_vals) ** 2 / np.maximum(hist, 1.0)))
    ndof = max(1, len(hist) - 4)
    return {
        "entries": float(work.size),
        "mu": mu,
        "sigma": sigma,
        "amp": amp,
        "const": const,
        "fit_bin_width": fit_bin_width,
        "chi2_ndf": chi2 / ndof,
        "fit_ok": fit_ok,
    }


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_events_with_raw(
    rows: list[dict[str, float]],
    coeffs: np.ndarray,
    member_index,
    *,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    events, closure = rows_to_member_events(rows, coeffs, member_index, rtol=rtol, atol=atol)
    if int(closure["usable_rows"]) != int(closure["selected_rows"]):
        raise SystemExit(f"member closure incomplete for mass-fit reference sample: {closure}")
    if len(events) != len(rows):
        raise SystemExit("internal alignment error: event count differs from row count")
    for event, row in zip(events, rows):
        event["raw_mass_mev"] = float(row["pair_m"]) * 1000.0
    return events, closure


def derive_fold_model(
    train: list[dict[str, object]],
    slices,
    *,
    min_seed_entries: int,
    min_member_entries: int,
    iterations: int,
    damping: float,
    max_update_frac: float,
    left_cols: set[int],
) -> tuple[dict[int, dict[str, float]], dict[str, dict[int, dict[str, float]]], dict[int, float]]:
    global_seed = derive_seed_scales(train, slices, min_seed_entries)
    period_seed = {
        "x60_4a2": derive_seed_scales([e for e in train if period_for_run(int(e["run"])) == "x60_4a2"], slices, min_seed_entries),
        "x60_4b": derive_seed_scales([e for e in train if period_for_run(int(e["run"])) == "x60_4b"], slices, min_seed_entries),
    }
    member_scales: dict[int, float] = {}
    for _ in range(iterations):
        updates = derive_member_updates(train, slices, global_seed, period_seed, member_scales, left_cols, min_member_entries)
        for block, row in sorted(updates.items()):
            applied = clip_update(float(row["update_scale_full"]) ** damping, max_update_frac)
            member_scales[block] = float(member_scales.get(block, 1.0)) * applied
    return global_seed, period_seed, member_scales


def collect_holdout_masses(
    train_events: list[dict[str, object]],
    plot_events: list[dict[str, object]],
    slices,
    args: argparse.Namespace,
) -> dict[str, list[float]]:
    left_cols = set(args.left_fallback_cols)
    global_seed, period_seed, member_scales = derive_fold_model(
        train_events,
        slices,
        min_seed_entries=args.min_seed_entries,
        min_member_entries=args.min_member_entries,
        iterations=args.iterations,
        damping=args.damping,
        max_update_frac=args.max_update_frac,
        left_cols=left_cols,
    )
    out = {"raw": [], "shared_time": [], "seed_policy": [], f"seed_plus_member_iter{args.iterations}": []}
    for event in plot_events:
        out["raw"].append(float(event["raw_mass_mev"]))
        out["shared_time"].append(float(event["shared_mass_mev"]) * event_time_scale(event, slices))
        seed_mass, _ = apply_stack(event, slices, global_seed, period_seed, {}, left_cols)
        member_mass, _ = apply_stack(event, slices, global_seed, period_seed, member_scales, left_cols)
        out["seed_policy"].append(seed_mass)
        out[f"seed_plus_member_iter{args.iterations}"].append(member_mass)
    return out


def plot_panel(ax, masses: np.ndarray, label: str, color: str, hist_range: tuple[float, float], fit_range: tuple[float, float]) -> dict[str, float]:
    hist, edges, _ = ax.hist(masses, bins=120, range=hist_range, color=color, alpha=0.78, label=label)
    fit = fit_peak(masses, fit_range[0], fit_range[1])
    xfit = np.linspace(fit_range[0], fit_range[1], 400)
    bin_width = (hist_range[1] - hist_range[0]) / 120.0
    amp = fit["amp"]
    if math.isfinite(amp):
        scale_to_display_bins = bin_width / float(fit.get("fit_bin_width", bin_width))
        ax.plot(
            xfit,
            gaussian_const(xfit, amp, fit["mu"], fit["sigma"], fit["const"]) * scale_to_display_bins,
            color="#1a202c",
            linewidth=2.0,
            label="local G + const fit",
        )
        ax.plot(
            xfit,
            np.full_like(xfit, fit["const"] * scale_to_display_bins),
            color="#1a202c",
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
        )
    ax.axvline(M_PI0_MEV, color="#111111", linestyle="--", linewidth=1.2, label=r"$m_{\pi^0}$")
    ax.axvline(fit["mu"], color="#c53030", linewidth=1.2, label=f"fit mean {fit['mu']:.2f}")
    ax.axvspan(fit_range[0], fit_range[1], color="#111111", alpha=0.04)
    ax.set_title(f"{label}\n$\\mu$={fit['mu']:.2f} MeV, $\\sigma$={fit['sigma']:.2f} MeV")
    ax.set_xlabel(r"$m_{\gamma\gamma}$ [MeV]")
    ax.set_ylabel("entries / bin")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fit["bin_width"] = bin_width
    fit["hist_entries"] = float(len(masses))
    return fit


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_npz = Path(args.cache_npz) if args.cache_npz else outdir / "mgg_stage_masses.npz"

    member_key = f"seed_plus_member_iter{args.iterations}"
    if cache_npz.exists() and not args.force_rebuild_cache:
        loaded = np.load(cache_npz)
        pooled = {key: loaded[key].astype(float).tolist() for key in ("raw", "shared_time", "seed_policy", member_key)}
        cache_note = f"loaded_cache: {cache_npz}"
        input_paths = []
        derive_rows = []
        plot_rows = []
        derive_closure = {"cache": "loaded"}
        plot_closure = {"cache": "loaded"}
    else:
        chunk_summaries = [Path(p) for p in args.chunk_summary] if args.chunk_summary else default_chunk_summaries()
        input_paths: list[Path] = []
        for summary in chunk_summaries:
            input_paths.extend(input_paths_from_summary(summary))
        input_paths = list(dict.fromkeys(input_paths))
        if not input_paths:
            raise SystemExit("no compact input paths found")

        coeff = load_coeff(Path(args.coeff_tsv))
        coeffs = np.asarray([coeff["c0"], coeff["c1"], coeff["c2"]], dtype=float)
        slices = load_time_slices(Path(args.time_slice_tsv))
        rows = read_rows(input_paths)
        member_index, missing = load_member_index(input_paths)
        if missing:
            raise SystemExit(f"missing member sidecars: {missing[:5]}")

        common = {
            "timing_window": args.timing_window,
            "timing_center": args.timing_center,
            "min_e_floor": args.min_e_floor,
            "nclusters_mode": "exact2",
            "emin_range": (args.emin_range[0], args.emin_range[1]),
        }
        derive_rows = select_raw_rows(
            rows,
            mass_window=(args.derive_mass_window[0], args.derive_mass_window[1]),
            **common,
        )
        plot_rows = select_raw_rows(
            rows,
            mass_window=(args.plot_mass_window[0], args.plot_mass_window[1]),
            **common,
        )
        derive_events, derive_closure = build_events_with_raw(derive_rows, coeffs, member_index, rtol=args.rtol, atol=args.atol)
        plot_events, plot_closure = build_events_with_raw(plot_rows, coeffs, member_index, rtol=args.rtol, atol=args.atol)

        derive_by_parity = {parity: events for parity, events in (("even", [e for e in derive_events if int(e["run"]) % 2 == 0]), ("odd", [e for e in derive_events if int(e["run"]) % 2 == 1]))}
        plot_by_parity = {parity: events for parity, events in (("even", [e for e in plot_events if int(e["run"]) % 2 == 0]), ("odd", [e for e in plot_events if int(e["run"]) % 2 == 1]))}

        pooled = {"raw": [], "shared_time": [], "seed_policy": [], member_key: []}
        for train_parity, test_parity in (("even", "odd"), ("odd", "even")):
            fold_masses = collect_holdout_masses(derive_by_parity[train_parity], plot_by_parity[test_parity], slices, args)
            for key, vals in fold_masses.items():
                pooled[key].extend(vals)
        np.savez_compressed(cache_npz, **{key: np.asarray(vals, dtype=float) for key, vals in pooled.items()})
        cache_note = f"wrote_cache: {cache_npz}"

    model_labels = {
        "raw": "raw-WF baseline",
        "shared_time": "shared + time",
        "seed_policy": "seed policy",
        f"seed_plus_member_iter{args.iterations}": f"seed + member iter {args.iterations}",
    }
    colors = {
        "raw": "#718096",
        "shared_time": "#4c78a8",
        "seed_policy": "#2f855a",
        f"seed_plus_member_iter{args.iterations}": "#c45a3c",
    }
    hist_range = (args.hist_window[0], args.hist_window[1])
    fit_range = (args.fit_window[0], args.fit_window[1])

    fit_rows: list[dict[str, object]] = []
    for key, vals in pooled.items():
        masses = np.asarray(vals, dtype=float)
        fit = fit_peak(masses, fit_range[0], fit_range[1])
        residuals = masses - M_PI0_MEV
        fit_rows.append(
            {
                "model": key,
                "label": model_labels[key],
                "hist_entries": len(masses),
                "fit_entries": int(fit["entries"]),
                "fit_mu_mev": fit["mu"],
                "fit_sigma_mev": fit["sigma"],
                "fit_bias_mev": fit["mu"] - M_PI0_MEV,
                **{f"all_{k}": v for k, v in metrics(residuals).items()},
            }
        )
    write_table(outdir / "mgg_fit_summary.tsv", fit_rows)

    with PdfPages(outdir / "period_fallback_mgg_fit_reference.pdf") as pdf:
        fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)
        for ax, key in zip(axs.flat, ("raw", "shared_time", "seed_policy", f"seed_plus_member_iter{args.iterations}")):
            plot_panel(ax, np.asarray(pooled[key], dtype=float), model_labels[key], colors[key], hist_range, fit_range)
        fig.suptitle("Held-out invariant-mass distributions: broad exact-2/timing/E selection", fontsize=15)
        pdf.savefig(fig)
        fig.savefig(outdir / "period_fallback_mgg_fit_reference.png", dpi=180)
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
        for ax, key in zip(axs.flat, ("seed_policy", f"seed_plus_member_iter{args.iterations}")):
            plot_panel(ax, np.asarray(pooled[key], dtype=float), model_labels[key], colors[key], hist_range, fit_range)
        fig.suptitle("Final candidates only", fontsize=15)
        pdf.savefig(fig)
        fig.savefig(outdir / "period_fallback_mgg_fit_candidates.png", dpi=180)
        plt.close(fig)

    summary = [
        "Invariant-mass Gaussian reference plots",
        cache_note,
        f"input_tsvs: {len(input_paths)}",
        f"derive_selected_rows: {len(derive_rows)}",
        f"derive_closure: {derive_closure}",
        f"plot_selected_rows: {len(plot_rows)}",
        f"plot_closure: {plot_closure}",
        f"derive_mass_window_gev: {args.derive_mass_window[0]} {args.derive_mass_window[1]}",
        f"plot_mass_window_gev: {args.plot_mass_window[0]} {args.plot_mass_window[1]}",
        f"fit_window_mev: {args.fit_window[0]} {args.fit_window[1]}",
        "Note: correction layers are derived on the working pi0 mass window, then plotted on a broader mass window.",
    ]
    (outdir / "period_fallback_mgg_fit_reference_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    print(outdir / "period_fallback_mgg_fit_reference.pdf")
    print(outdir / "mgg_fit_summary.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
