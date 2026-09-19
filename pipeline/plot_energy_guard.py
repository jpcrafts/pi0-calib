#!/usr/bin/env python3
"""Plot legacy versus guarded TOTAL cluster correction for a frozen package."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nps_pi0_calibration import FrozenCalibration
from nps_pi0_calibration.calibration import upper_energy_weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--runs", required=True, help="Comma-separated runs")
    parser.add_argument("--seed", type=int, default=608)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    calibration = FrozenCalibration.load(args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    energies = [0.6 + i * 0.01 for i in range(441)]
    rows = []
    with PdfPages(args.output) as pdf:
        for run in map(int, args.runs.split(",")):
            period = calibration.run_period[run]
            old, guarded = [], []
            for energy in energies:
                # Legacy calculation for comparison ONLY; never write it to production.
                scale = (calibration._curve_scale(period, energy) * calibration.run_scales[run]
                         * calibration.seed_scales.get(args.seed, 1.0)
                         * calibration._lowe_scale(period, energy))
                result = calibration.correct(run, energy, args.seed)
                old.append(100 * (scale - 1))
                guarded.append(100 * (result.total_scale - 1))
                rows.append(dict(run=run, seed=args.seed, original_energy_gev=energy,
                                 weight=upper_energy_weight(energy), legacy_total_scale=scale,
                                 guarded_total_scale=result.total_scale))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
            for ax, limits in zip(axes, ((0.6, 5.0), (1.9, 2.6))):
                ax.plot(energies, old, color="#c44e52", ls="--", label="Legacy (unsafe above support)")
                ax.plot(energies, guarded, color="#0072b2", label="Guard + taper candidate")
                ax.axhline(0, color="black", lw=0.7)
                ax.axvspan(2, 2.5, color="#e69f00", alpha=0.12, label="2-2.5 GeV taper")
                ax.axvline(2.5, color="black", ls=":")
                ax.set(xlim=limits, xlabel="Original Hao cluster energy (GeV)",
                       ylabel="Total applied energy correction (%)")
                ax.grid(alpha=0.2)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"Run {run}, seed {args.seed}: all layers included\n"
                         "Diagnostic application policy, not a refitted or approved response")
            pdf.savefig(fig)
            if run == int(args.runs.split(",")[0]):
                fig.savefig(args.output.with_suffix(".png"), dpi=130)
            plt.close(fig)
    with args.output.with_suffix(".tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
