#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from timing_window_utils import parse_timing_center, passes_timing_window, resolve_timing_center


M_PI0_GEV = 0.1349766


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit a shared energy-dependent correction on a pi0-enriched calibration-tree sample.")
    ap.add_argument("input_tsv", nargs="+")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mass-window", nargs=2, type=float, default=[0.11, 0.16])
    ap.add_argument("--mm2-window", nargs=2, type=float)
    ap.add_argument("--timing-window", type=float, default=None)
    ap.add_argument("--timing-center", default="0.0", help="Timing-cut center in ns, or 'auto'/'peak'")
    ap.add_argument("--min-e-floor", type=float, default=0.6)
    ap.add_argument("--nclusters-mode", choices=["exact2", "ge2"], default="exact2")
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    return ap.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows = []
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                parsed: dict[str, float] = {}
                for k, v in row.items():
                    parsed[k] = float(v)
                rows.append(parsed)
    return rows


def cluster_mode_ok(nclusters: int, mode: str) -> bool:
    if mode == "exact2":
        return nclusters == 2
    if mode == "ge2":
        return nclusters >= 2
    raise ValueError(f"unsupported nclusters mode: {mode}")


def corr_model(coeffs: np.ndarray, e: np.ndarray) -> np.ndarray:
    ee = np.maximum(e, 1e-6)
    return np.exp(coeffs[0] + coeffs[1] / ee + coeffs[2] * np.log(ee))


def fit_shared_curve(rows: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, float], list[tuple[float, float, int, float, float]], np.ndarray, np.ndarray]:
    e1 = np.array([r["e1"] for r in rows], dtype=float)
    e2 = np.array([r["e2"] for r in rows], dtype=float)
    pair_m = np.array([r["pair_m"] for r in rows], dtype=float)

    target_log = np.log(M_PI0_GEV / np.maximum(pair_m, 1e-9))
    x = np.column_stack(
        [
            0.5 * np.ones_like(e1) + 0.5 * np.ones_like(e2),
            0.5 / np.maximum(e1, 1e-6) + 0.5 / np.maximum(e2, 1e-6),
            0.5 * np.log(np.maximum(e1, 1e-6)) + 0.5 * np.log(np.maximum(e2, 1e-6)),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(x, target_log, rcond=None)

    s1 = corr_model(coeffs, e1)
    s2 = corr_model(coeffs, e2)
    corrected = pair_m * np.sqrt(s1 * s2)

    summary = {
        "n_rows": float(len(rows)),
        "raw_mean_mev": float(np.mean(pair_m) * 1000.0),
        "corr_mean_mev": float(np.mean(corrected) * 1000.0),
        "raw_mae_mev": float(np.mean(np.abs(pair_m - M_PI0_GEV)) * 1000.0),
        "corr_mae_mev": float(np.mean(np.abs(corrected - M_PI0_GEV)) * 1000.0),
        "raw_rmse_mev": float(np.sqrt(np.mean((pair_m - M_PI0_GEV) ** 2)) * 1000.0),
        "corr_rmse_mev": float(np.sqrt(np.mean((corrected - M_PI0_GEV) ** 2)) * 1000.0),
    }

    bins = []
    edges = np.array([0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
    emin = np.minimum(e1, e2)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (emin >= lo) & (emin < hi)
        if not np.any(mask):
            continue
        bins.append((float(lo), float(hi), int(mask.sum()), float(np.mean(pair_m[mask]) * 1000.0), float(np.mean(corrected[mask]) * 1000.0)))

    return coeffs, summary, bins, pair_m, corrected


def main() -> int:
    args = parse_args()
    input_paths = [Path(p).resolve() for p in args.input_tsv]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_paths)
    mass_lo, mass_hi = args.mass_window
    emin_lo, emin_hi = args.emin_range
    timing_center_arg = parse_timing_center(args.timing_center)
    timing_center = resolve_timing_center(
        [
            r
            for r in rows
            if cluster_mode_ok(int(r["nclusters"]), args.nclusters_mode)
            and r["e1"] >= 0.6
            and r["e2"] >= 0.6
            and r["min_e"] >= args.min_e_floor
            and abs(r["x1"]) < 29.025
            and abs(r["x2"]) < 29.025
            and abs(r["y1"]) < 35.475
            and abs(r["y2"]) < 35.475
        ],
        timing_center_arg,
    )
    selected = []
    for r in rows:
        if not cluster_mode_ok(int(r["nclusters"]), args.nclusters_mode):
            continue
        if r["e1"] < 0.6 or r["e2"] < 0.6:
            continue
        if r["min_e"] < args.min_e_floor:
            continue
        if abs(r["x1"]) >= 29.025 or abs(r["x2"]) >= 29.025:
            continue
        if abs(r["y1"]) >= 35.475 or abs(r["y2"]) >= 35.475:
            continue
        if not passes_timing_window(r, args.timing_window, timing_center):
            continue
        if not (mass_lo <= r["pair_m"] <= mass_hi):
            continue
        if args.mm2_window and not (args.mm2_window[0] <= r["pair_mm2"] <= args.mm2_window[1]):
            continue
        if not (emin_lo <= r["min_e"] <= emin_hi):
            continue
        selected.append(r)

    coeffs, summary, bins, raw_pair_m, corrected_pair_m = fit_shared_curve(selected)

    (output_dir / "shared_curve_summary.txt").write_text(
        "\n".join(
            [
                *[f"input_tsv: {p}" for p in input_paths],
                f"mass_window: {mass_lo} {mass_hi}",
                f"nclusters_mode: {args.nclusters_mode}",
                f"min_e_floor: {args.min_e_floor}",
                f"timing_window: {args.timing_window}",
                f"timing_center_arg: {args.timing_center}",
                f"timing_center: {timing_center:.6f}",
                f"emin_range: {emin_lo} {emin_hi}",
                f"n_rows: {int(summary['n_rows'])}",
                f"log(scale) = {coeffs[0]:.12g} + ({coeffs[1]:.12g})/E + ({coeffs[2]:.12g})*ln(E)",
                f"raw_mean_mev: {summary['raw_mean_mev']:.6f}",
                f"corr_mean_mev: {summary['corr_mean_mev']:.6f}",
                f"raw_mae_mev: {summary['raw_mae_mev']:.6f}",
                f"corr_mae_mev: {summary['corr_mae_mev']:.6f}",
                f"raw_rmse_mev: {summary['raw_rmse_mev']:.6f}",
                f"corr_rmse_mev: {summary['corr_rmse_mev']:.6f}",
                "energy_bins:",
                *[
                    f"  {lo:.2f}-{hi:.2f} GeV: n={n} raw_mean_mev={raw:.6f} corr_mean_mev={corr:.6f}"
                    for lo, hi, n, raw, corr in bins
                ],
            ]
        )
        + "\n",
        encoding="ascii",
    )

    with (output_dir / "shared_curve_points.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["elo", "ehi", "entries", "raw_mean_mev", "corr_mean_mev"])
        for lo, hi, n, raw, corr in bins:
            writer.writerow([lo, hi, n, raw, corr])

    with (output_dir / "shared_curve_coeffs.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerow(["c0", coeffs[0]])
        writer.writerow(["c1", coeffs[1]])
        writer.writerow(["c2", coeffs[2]])
        writer.writerow(["mass_lo", mass_lo])
        writer.writerow(["mass_hi", mass_hi])
        writer.writerow(["timing_window", "" if args.timing_window is None else args.timing_window])
        writer.writerow(["timing_center", timing_center])
        writer.writerow(["emin_lo", emin_lo])
        writer.writerow(["emin_hi", emin_hi])

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(raw_pair_m * 1000.0, bins=120, range=(80.0, 180.0), histtype="step", linewidth=2.0, color="#4c78a8", label="Raw")
    ax.hist(corrected_pair_m * 1000.0, bins=120, range=(80.0, 180.0), histtype="step", linewidth=2.0, color="#c45a3c", label="Corrected")
    ax.axvline(M_PI0_GEV * 1000.0, color="#222222", linestyle="--", linewidth=1.5, label=r"$m_{\pi^0}$")
    ax.set_xlabel(r"$m_{\gamma\gamma}$ (MeV)")
    ax.set_ylabel("Events")
    ax.set_title("Shared-Curve Raw vs Corrected Pair Mass")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "shared_curve_mgg_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(corrected_pair_m * 1000.0, bins=120, range=(80.0, 180.0), color="#c45a3c", alpha=0.88)
    ax.axvline(M_PI0_GEV * 1000.0, color="#222222", linestyle="--", linewidth=1.5, label=r"$m_{\pi^0}$")
    ax.set_xlabel(r"$m_{\gamma\gamma}$ (MeV)")
    ax.set_ylabel("Events")
    ax.set_title("Shared-Curve Corrected Pair Mass")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "shared_curve_corrected_mgg_hist.png", dpi=180)
    plt.close(fig)

    print(f"selected_rows {len(selected)}")
    print(f"coeffs {coeffs[0]} {coeffs[1]} {coeffs[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
