#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from build_energy_correction_stack import M_PI0_MEV
from plot_period_fallback_mgg_fits import fit_peak, gaussian_const


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot raw and shared-curve-corrected Pi0 mass fits from compact pair TSVs."
    )
    ap.add_argument("input_tsv", nargs="+")
    ap.add_argument("--coeff-tsv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=None)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--hist-window", nargs=2, type=float, default=[80.0, 200.0])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[105.0, 165.0])
    return ap.parse_args()


def load_coeff(path: Path) -> tuple[float, float, float]:
    vals: dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vals[row["key"]] = float(row["value"])
    return vals["c0"], vals["c1"], vals["c2"]


def shared_scale(e_gev: np.ndarray, coeff: tuple[float, float, float]) -> np.ndarray:
    c0, c1, c2 = coeff
    safe_e = np.maximum(e_gev, 1.0e-6)
    return np.exp(c0 + c1 / safe_e + c2 * np.log(safe_e))


def read_events(paths: list[Path], coeff: tuple[float, float, float], timing_center: float | None, timing_window: float, emin_lo: float, emin_hi: float) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if int(row.get("nclusters", "0")) != 2:
                    continue
                e1 = float(row["e1"])
                e2 = float(row["e2"])
                min_e = min(e1, e2)
                if min_e < emin_lo or min_e > emin_hi:
                    continue
                if timing_center is not None:
                    dt = float(row["dt12"])
                    if abs(dt - timing_center) > timing_window:
                        continue
                raw_m = float(row["pair_m"]) * 1000.0
                scales = shared_scale(np.asarray([e1, e2], dtype=float), coeff)
                corr_m = raw_m * math.sqrt(float(scales[0] * scales[1]))
                events.append(
                    {
                        "run": float(row["run"]),
                        "min_e": min_e,
                        "raw_m": raw_m,
                        "shared_m": corr_m,
                    }
                )
    return events


def plot_fit(ax: plt.Axes, masses: np.ndarray, label: str, hist_window: tuple[float, float], fit_window: tuple[float, float]) -> dict[str, float]:
    hist, edges = np.histogram(masses, bins=100, range=hist_window)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax.step(centers, hist, where="mid", lw=1.1, label=label)
    fit = fit_peak(masses, fit_window[0], fit_window[1])
    if fit["fit_ok"] and math.isfinite(fit["mu"]):
        xfit = np.linspace(fit_window[0], fit_window[1], 300)
        yfit = gaussian_const(xfit, fit["amp"], fit["mu"], fit["sigma"], fit["const"])
        ax.plot(xfit, yfit, lw=1.3)
    ax.axvline(M_PI0_MEV, color="k", lw=0.8, alpha=0.5)
    ax.set_title(f"{label}: entries={len(masses)} mu={fit['mu']:.2f} sigma={fit['sigma']:.2f}")
    ax.set_xlabel("m_gg (MeV)")
    ax.set_ylabel(f"counts / {width:.1f} MeV")
    return fit


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    coeff = load_coeff(Path(args.coeff_tsv))
    events = read_events(
        [Path(p) for p in args.input_tsv],
        coeff,
        args.timing_center,
        args.timing_window,
        args.emin_range[0],
        args.emin_range[1],
    )

    groups: list[tuple[str, list[dict[str, float]]]] = [("pooled", events)]
    by_run: dict[int, list[dict[str, float]]] = defaultdict(list)
    for ev in events:
        by_run[int(ev["run"])].append(ev)
    for run in sorted(by_run):
        groups.append((f"run_{run}", by_run[run]))

    energy_bins = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 2.5)]
    for lo, hi in energy_bins:
        groups.append((f"emin_{lo:.1f}_{hi:.1f}", [ev for ev in events if lo <= ev["min_e"] < hi]))

    rows: list[dict[str, object]] = []
    pdf_path = outdir / "compact_shared_curve_mass_fits.pdf"
    with PdfPages(pdf_path) as pdf:
        for group, group_events in groups:
            if not group_events:
                continue
            fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
            for ax, model, key in [(axs[0], "raw", "raw_m"), (axs[1], "shared_curve", "shared_m")]:
                masses = np.asarray([ev[key] for ev in group_events], dtype=float)
                fit = plot_fit(ax, masses, f"{group} {model}", tuple(args.hist_window), tuple(args.fit_window))
                rows.append(
                    {
                        "group": group,
                        "model": model,
                        "entries": len(masses),
                        "fit_entries": int(fit["entries"]),
                        "fit_ok": int(fit["fit_ok"]),
                        "fit_mu_mev": fit["mu"],
                        "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(fit["mu"]) else math.nan,
                        "fit_sigma_mev": fit["sigma"],
                        "chi2_ndf": fit.get("chi2_ndf", math.nan),
                    }
                )
            pdf.savefig(fig)
            plt.close(fig)

    write_tsv(outdir / "compact_shared_curve_mass_fit_summary.tsv", rows)
    print(pdf_path)
    print(outdir / "compact_shared_curve_mass_fit_summary.tsv")


if __name__ == "__main__":
    main()
