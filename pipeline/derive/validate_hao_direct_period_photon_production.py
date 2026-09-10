#!/usr/bin/env python3
"""Freeze and production-test the direct Hao-start period photon curve."""

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

import validate_hao_full_iter2_pairspace as pairspace
import validate_hao_period_energy_smoothed_seed as period_model
import validate_hao_start_layered_member_stack as base
import validate_hao_trusted_merged_energy_layer as direct
from pi0_mass_fit import DEFAULT_CONFIG, FitConfig, M_PI0_MEV, evaluate_model, fit_pi0_mass
from kinematic_calibration_config import read_run_period_lut


MODEL = "hao_period_photon_l10_d100_low045_k07_w150"
_RUN_PERIOD_MAP: dict[int, str] | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--production-pair-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--runs", default="4300,4417,4503")
    ap.add_argument("--timing-center", type=float, default=-0.213982)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--min-fit-entries", type=int, default=200)
    ap.add_argument("--min-photon-cell-entries", type=int, default=500)
    ap.add_argument("--fit-max-mu-err", type=float, default=0.50)
    ap.add_argument("--fit-max-chi2-ndf", type=float, default=math.inf)
    ap.add_argument("--fit-warn-chi2-ndf", type=float, default=3.0)
    ap.add_argument("--fit-max-window-shift", type=float, default=0.50)
    ap.add_argument("--smooth-lambda", type=float, default=10.0)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--low-positive-clip-frac", type=float, default=0.045)
    ap.add_argument("--max-update-frac", type=float, default=0.03)
    ap.add_argument("--low-constraint-weight", type=float, default=1.5)
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


def fit_config(args: argparse.Namespace, min_entries: int | None = None) -> FitConfig:
    return FitConfig(
        **{
            **DEFAULT_CONFIG.__dict__,
            "min_entries": min_entries if min_entries is not None else args.min_fit_entries,
            "max_mu_err": args.fit_max_mu_err,
            "max_chi2_ndf": args.fit_max_chi2_ndf,
            "warn_chi2_ndf": args.fit_warn_chi2_ndf,
            "max_window_shift": args.fit_max_window_shift,
        }
    )


def configure_run_period_lut(path: Path | None) -> dict[int, str] | None:
    global _RUN_PERIOD_MAP
    _RUN_PERIOD_MAP = read_run_period_lut(path) if path else None
    return _RUN_PERIOD_MAP


def derivation_args(args: argparse.Namespace) -> SimpleNamespace:
    mapping = configure_run_period_lut(getattr(args, "run_period_lut", None))
    return SimpleNamespace(
        timing_center=args.timing_center,
        timing_window=args.timing_window,
        emin_range=args.emin_range,
        period_mode="explicit" if mapping else "hao_lh2_cycles",
        _run_period_map=mapping,
        n_run_bands=3,
        min_photon_cell_entries=args.min_photon_cell_entries,
        max_period_photon_update_frac=args.max_update_frac,
        fit_max_mu_err=args.fit_max_mu_err,
        fit_max_chi2_ndf=args.fit_max_chi2_ndf,
        fit_warn_chi2_ndf=args.fit_warn_chi2_ndf,
        fit_max_window_shift=args.fit_max_window_shift,
    )


def period_for_run(run: int) -> str:
    if _RUN_PERIOD_MAP is not None:
        if run not in _RUN_PERIOD_MAP:
            raise ValueError(f"run {run} is absent from the configured run-period LUT")
        return _RUN_PERIOD_MAP[run]
    if run <= 4308:
        return "lh2_cycle12"
    if run <= 4404:
        return "lh2_cycle34"
    return "lh2_cycle56"


def read_production_pairs(pair_dir: Path, runs: set[int], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    qa: list[dict[str, object]] = []
    for path in sorted(pair_dir.glob("run*_production_exact2_pairs.tsv")):
        total = selected = 0
        with path.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"run", "segment", "event", "pair_m", "e1", "e2", "dt12"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"{path}: missing fields {sorted(missing)}")
            for row in reader:
                total += 1
                run = int(row["run"])
                if run not in runs:
                    continue
                e1 = float(row["e1"])
                e2 = float(row["e2"])
                min_e = min(e1, e2)
                if not (args.emin_range[0] <= min_e <= args.emin_range[1]):
                    continue
                if abs(float(row["dt12"]) - args.timing_center) > args.timing_window:
                    continue
                events.append(
                    {
                        "run": run,
                        "segment": int(row["segment"]),
                        "event": int(row["event"]),
                        "mass_mev": float(row["pair_m"]) * 1000.0,
                        "e1": e1,
                        "e2": e2,
                        "min_e": min_e,
                    }
                )
                selected += 1
        qa.append({"file": str(path), "total_rows": total, "selected_rows": selected})
    if not events:
        raise SystemExit("No production pairs passed the run, timing, and energy selection")
    return events, qa


def fit_row(group: str, model: str, values: list[float], args: argparse.Namespace) -> dict[str, object]:
    fit = fit_pi0_mass(np.asarray(values, dtype=float), fit_config(args))
    return {
        "group": group,
        "model": model,
        "entries": len(values),
        "fit_definition": "canonical_gaussian_linear_118_150",
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


def metric_groups(events: list[dict[str, object]]) -> list[tuple[str, list[int]]]:
    groups = [("pooled", list(range(len(events))))]
    for run in sorted({int(event["run"]) for event in events}):
        groups.append((f"run_{run}", [i for i, event in enumerate(events) if int(event["run"]) == run]))
    for energy_bin in base.ENERGY_BINS:
        label = base.bin_label(energy_bin)
        groups.append(
            (
                f"emin_{label}",
                [i for i, event in enumerate(events) if base.energy_bin(float(event["min_e"])) == energy_bin],
            )
        )
    return groups


def make_plots(path: Path, events: list[dict[str, object]], masses: dict[str, list[float]], metrics: list[dict[str, object]], args: argparse.Namespace) -> None:
    metric_map = {(str(row["group"]), str(row["model"])): row for row in metrics}
    labels = {"hao_baseline": "Hao production baseline", MODEL: "Hao + direct period photon curve (w=1.5)"}
    colors = {"hao_baseline": "#777777", MODEL: "#006d77"}
    models = ["hao_baseline", MODEL]
    with PdfPages(path) as pdf:
        for group, indexes in metric_groups(events):
            if len(indexes) < 50:
                continue
            fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
            bins = np.arange(105.0, 165.01, 1.0)
            centers = 0.5 * (bins[:-1] + bins[1:])
            for model in models:
                values = np.asarray([masses[model][i] for i in indexes], dtype=float)
                ax.hist(values, bins=bins, histtype="step", lw=1.8, color=colors[model], label=labels[model])
                row = metric_map[(group, model)]
                fit = fit_pi0_mass(values, fit_config(args))
                if int(fit["optimizer_ok"]):
                    ax.plot(centers, evaluate_model(centers, fit), color=colors[model], ls="--", lw=1.4)
                ax.plot([], [], color=colors[model], label=f"fit mu={float(row['fit_mu_mev']):.3f}, sigma={float(row['fit_sigma_mev']):.3f} MeV")
            ax.axvline(M_PI0_MEV, color="black", lw=1, ls=":", label="PDG pi0 mass")
            ax.set_title(f"Production-style invariant mass: {group}")
            ax.set_xlabel("m(gamma gamma) (MeV)")
            ax.set_ylabel("events / 1 MeV")
            ax.legend(fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
        run_groups = [group for group, _ in metric_groups(events) if group.startswith("run_")]
        for model in models:
            rows = [metric_map[(group, model)] for group in run_groups]
            runs = [int(str(row["group"]).split("_")[1]) for row in rows]
            axes[0].errorbar(runs, [float(row["fit_mu_residual_mev"]) for row in rows], yerr=[float(row["fit_mu_err_mev"]) for row in rows], marker="o", label=labels[model], color=colors[model])
            axes[1].errorbar(runs, [float(row["fit_sigma_mev"]) for row in rows], yerr=[float(row["fit_sigma_err_mev"]) for row in rows], marker="o", label=labels[model], color=colors[model])
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_ylabel("fit peak residual (MeV)")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("run")
        axes[1].set_ylabel("fit sigma (MeV)")
        axes[1].legend(fontsize=8)
        fig.suptitle("Production-style transfer check by run")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    package_dir = outdir / "frozen_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    runs = {int(item) for item in args.runs.split(",") if item.strip()}

    derive_args = derivation_args(args)
    summary_rows = base.load_conversion_summary(Path(args.conversion_summary))
    calibration_events, calibration_qa = pairspace.read_events(summary_rows, derive_args)
    curves, derivation_rows, knots = direct.derive_hao_direct_photon_curves(
        "full_sample",
        calibration_events,
        MODEL,
        args.damping,
        args.smooth_lambda,
        args.low_positive_clip_frac,
        args.low_constraint_weight,
        True,
        True,
        derive_args,
    )
    write_tsv(package_dir / "derivation_rows.tsv", derivation_rows)
    lut_rows = []
    for period, coeff in sorted(curves.items()):
        for energy, log_scale in zip(knots, coeff):
            lut_rows.append(
                {
                    "model": MODEL,
                    "period": period,
                    "energy_gev": energy,
                    "energy_scale": math.exp(float(log_scale)),
                    "log_energy_scale": float(log_scale),
                }
            )
    write_tsv(package_dir / "period_photon_curve.tsv", lut_rows)
    write_tsv(package_dir / "calibration_sidecar_quality.tsv", calibration_qa)
    (package_dir / "metadata.txt").write_text(
        "\n".join(
            [
                f"model: {MODEL}",
                f"conversion_summary: {Path(args.conversion_summary).resolve()}",
                "baseline: Hao CALO_calib_Pi0Coef energy",
                "application: each cluster independently receives s_period(E_cluster)",
                "periods: <=4308 cycle12; 4309-4404 cycle34; >=4405 cycle56",
                f"smooth_lambda: {args.smooth_lambda}",
                f"damping: {args.damping}",
                f"low_constraint_weight: {args.low_constraint_weight}",
                f"positive_low_energy_clip: {args.low_positive_clip_frac}",
                f"general_clip: {args.max_update_frac}",
                "target: pi0 mass 134.9766 MeV",
                "scope: LH2 x60_4 only",
                "",
            ]
        ),
        encoding="ascii",
    )

    production_events, production_qa = read_production_pairs(Path(args.production_pair_dir), runs, args)
    write_tsv(outdir / "production_pair_quality.tsv", production_qa)
    masses = {"hao_baseline": [], MODEL: []}
    event_rows = []
    for event in production_events:
        run = int(event["run"])
        period = period_for_run(run)
        coeff = curves.get(period, curves["global"])
        scale1 = direct.full_photon_energy_scale(coeff, float(event["e1"]), knots)
        scale2 = direct.full_photon_energy_scale(coeff, float(event["e2"]), knots)
        mass_before = float(event["mass_mev"])
        mass_after = mass_before * math.sqrt(scale1 * scale2)
        masses["hao_baseline"].append(mass_before)
        masses[MODEL].append(mass_after)
        event_rows.append(
            {
                **event,
                "period": period,
                "scale1": scale1,
                "scale2": scale2,
                "e1_corrected": float(event["e1"]) * scale1,
                "e2_corrected": float(event["e2"]) * scale2,
                "mass_corrected_mev": mass_after,
            }
        )
    write_tsv(outdir / "production_event_comparison.tsv", event_rows)

    metric_rows = []
    for group, indexes in metric_groups(production_events):
        if not indexes:
            continue
        for model in ["hao_baseline", MODEL]:
            metric_rows.append(fit_row(group, model, [masses[model][i] for i in indexes], args))
    write_tsv(outdir / "production_fit_metrics.tsv", metric_rows)
    make_plots(outdir / "production_direct_period_photon_validation.pdf", production_events, masses, metric_rows, args)

    pooled = {(row["group"], row["model"]): row for row in metric_rows}
    lines = [
        "Hao-start direct period photon production-style validation",
        f"candidate: {MODEL}",
        f"runs: {','.join(str(run) for run in sorted(runs))}",
        f"selected_events: {len(production_events)}",
        "diagnostic selection: exact-2 pair table, timing and min-energy cuts, no final mass-window cut",
        "",
    ]
    for model in ["hao_baseline", MODEL]:
        row = pooled[("pooled", model)]
        lines.append(
            f"{model}: mu={float(row['fit_mu_mev']):.6f} MeV; "
            f"residual={float(row['fit_mu_residual_mev']):+.6f} MeV; "
            f"sigma={float(row['fit_sigma_mev']):.6f} MeV; fit_ok={row['fit_ok']}"
        )
    lines.append("")
    for run in sorted(runs):
        before = pooled[(f"run_{run}", "hao_baseline")]
        after = pooled[(f"run_{run}", MODEL)]
        lines.append(
            f"run {run}: residual {float(before['fit_mu_residual_mev']):+.6f} -> "
            f"{float(after['fit_mu_residual_mev']):+.6f} MeV; sigma "
            f"{float(before['fit_sigma_mev']):.6f} -> {float(after['fit_sigma_mev']):.6f} MeV"
        )
    low_before = pooled[("emin_0.6-0.8", "hao_baseline")]
    low_after = pooled[("emin_0.6-0.8", MODEL)]
    lines.extend(
        [
            "",
            "0.6-0.8 GeV: residual "
            f"{float(low_before['fit_mu_residual_mev']):+.6f} -> "
            f"{float(low_after['fit_mu_residual_mev']):+.6f} MeV; sigma "
            f"{float(low_before['fit_sigma_mev']):.6f} -> {float(low_after['fit_sigma_mev']):.6f} MeV",
            "decision: do not promote standalone; energy closure improves, but run-level closure worsens for 4300 and 4417.",
        ]
    )
    lines.extend(
        [
            "",
            "Interpretation: this checks transfer through the independent production replay path.",
            "It is not statistically independent of calibration runs because the frozen curve uses the full calibration sample.",
        ]
    )
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
