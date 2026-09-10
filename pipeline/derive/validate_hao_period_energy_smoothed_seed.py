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

import validate_hao_start_layered_member_stack as base
from build_energy_correction_stack import M_PI0_MEV
from plot_compact_shared_curve_mass_fits import plot_fit
from plot_period_fallback_mgg_fits import fit_peak


BASE_MODELS = [
    "hao_raw",
    "hao_energy",
    "hao_period_energy",
    "hao_period_energy_lowe_patch",
]


def model_names(args: argparse.Namespace) -> list[str]:
    models = BASE_MODELS[:]
    models.extend(f"hao_period_energy_seed_iter_{i}" for i in range(1, max(1, args.seed_iterations) + 1))
    models.append("hao_period_energy_seed")
    if not args.no_member:
        models.extend(
            f"hao_period_energy_seed_member_iter_{i}" for i in range(1, max(1, args.member_iterations) + 1)
        )
        models.append("hao_period_energy_seed_member")
    return models


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Held-out Hao-start validation with a smoothed period x energy correction "
            "before seed/member residual layers."
        )
    )
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=None)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[105.0, 165.0])
    ap.add_argument("--hist-window", nargs=2, type=float, default=[80.0, 200.0])
    ap.add_argument("--min-fit-entries", type=int, default=200)
    ap.add_argument("--min-seed-events", type=int, default=20)
    ap.add_argument("--min-member-weight", type=float, default=20.0)
    ap.add_argument("--energy-damping", type=float, default=0.75)
    ap.add_argument("--period-energy-damping", type=float, default=0.75)
    ap.add_argument(
        "--energy-bin-damping",
        action="append",
        default=[],
        metavar="EMIN:EMAX:DAMPING",
        help="Override global energy damping for matching energy bins.",
    )
    ap.add_argument(
        "--period-energy-bin-damping",
        action="append",
        default=[],
        metavar="EMIN:EMAX:DAMPING",
        help="Override period-energy damping for matching energy bins.",
    )
    ap.add_argument("--seed-damping", type=float, default=0.50)
    ap.add_argument(
        "--seed-iterations",
        type=int,
        default=1,
        help="Iterate seed-block residual updates after energy/period/lowE layers. Iteration 1 is the current one-pass seed layer.",
    )
    ap.add_argument("--member-damping", type=float, default=0.10)
    ap.add_argument(
        "--member-iterations",
        type=int,
        default=1,
        help="Iterate member/deposit residual updates after the final seed layer.",
    )
    ap.add_argument("--max-energy-mass-update-frac", type=float, default=0.03)
    ap.add_argument("--max-period-energy-mass-update-frac", type=float, default=0.015)
    ap.add_argument("--max-seed-energy-update-frac", type=float, default=0.01)
    ap.add_argument("--max-member-energy-update-frac", type=float, default=0.0015)
    ap.add_argument("--left-cols", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument(
        "--period-mode",
        choices=["hao_lh2_cycles", "stat3", "stat_n"],
        default="hao_lh2_cycles",
        help="Run period definition used for the smoothed period-energy layer.",
    )
    ap.add_argument("--n-run-bands", type=int, default=6, help="Number of stat-balanced run bands for --period-mode stat_n.")
    ap.add_argument("--lowe-patch-bands", type=int, default=0, help="If >0, add stat-balanced low-E residual runband patch after period-energy.")
    ap.add_argument("--lowe-patch-emin", type=float, default=0.0, help="Apply low-E patch to min_pair_E bins at or above this lower edge.")
    ap.add_argument("--lowe-patch-emax", type=float, default=1.0, help="Apply low-E patch to min_pair_E bins below this upper edge.")
    ap.add_argument("--lowe-patch-damping", type=float, default=0.75)
    ap.add_argument("--max-lowe-patch-mass-update-frac", type=float, default=0.0075)
    ap.add_argument(
        "--lowe-bin-patch",
        action="append",
        default=[],
        metavar="EMIN:EMAX:BANDS:DAMPING:CLIP",
        help=(
            "Add one independent low-E patch. Example: 0.6:0.8:10:0.35:0.0035. "
            "If provided, these bin-specific patches replace the single --lowe-patch-* patch."
        ),
    )
    ap.add_argument(
        "--lowe-run-patch",
        action="append",
        default=[],
        metavar="EMIN:EMAX:RUNLO:RUNHI:DAMPING:CLIP",
        help=(
            "Add one explicit run-range low-E residual patch after lowe-bin-patch. "
            "Example: 0.6:0.8:4253:4259:0.50:0.005."
        ),
    )
    ap.add_argument(
        "--smooth-degree",
        type=int,
        default=2,
        help="Polynomial degree for smoothing log(period-energy mass factors) vs log(min pair E).",
    )
    ap.add_argument(
        "--no-member",
        action="store_true",
        help="Skip the diagnostic member layer. Seed-only output is still produced.",
    )
    ap.add_argument(
        "--fit-detail",
        choices=["summary", "run_energy"],
        default="summary",
        help="summary fits pooled/per-run/per-energy groups. run_energy also fits every run x energy bin.",
    )
    ap.add_argument("--skip-plots", action="store_true", help="Write TSV metrics only.")
    ap.add_argument(
        "--compact-only",
        action="store_true",
        help="Read only compact pair TSVs. Valid for seed-only scans; member layer requires sidecars.",
    )
    ap.add_argument(
        "--seed-regularization",
        choices=["legacy", "centered_shrink"],
        default="legacy",
        help="legacy uses raw seed median residuals. centered_shrink removes global eligible-seed median and shrinks low-stat seeds.",
    )
    ap.add_argument(
        "--seed-shrink-events",
        type=float,
        default=200.0,
        help="Pseudo-event scale for centered_shrink seed residual shrinkage toward identity.",
    )
    ap.add_argument(
        "--member-regularization",
        choices=["legacy", "centered_shrink"],
        default="legacy",
        help="legacy uses raw weighted member residuals. centered_shrink removes global weighted mean and shrinks low-weight blocks.",
    )
    ap.add_argument(
        "--member-shrink-weight",
        type=float,
        default=200.0,
        help="Pseudo-weight scale for centered_shrink member residual shrinkage toward identity.",
    )
    return ap.parse_args()


def period_key(run: int, events: list[dict[str, object]], mode: str) -> str:
    if mode == "hao_lh2_cycles":
        if run <= 4308:
            return "lh2_cycle12"
        if run <= 4404:
            return "lh2_cycle34"
        return "lh2_cycle56"

    runs = sorted({int(ev["run"]) for ev in events})
    if not runs:
        return "period_unknown"
    idx = runs.index(run) if run in runs else min(range(len(runs)), key=lambda i: abs(runs[i] - run))
    if mode == "stat_n":
        return "statn_placeholder"
    third = max(1, math.ceil(len(runs) / 3))
    return f"stat3_{min(2, idx // third)}"


def build_stat_run_bands(events: list[dict[str, object]], n_bands: int) -> dict[int, str]:
    by_run: dict[int, int] = defaultdict(int)
    for ev in events:
        by_run[int(ev["run"])] += 1
    runs = sorted(by_run)
    total = sum(by_run.values())
    target = max(1.0, total / max(1, n_bands))
    out: dict[int, str] = {}
    band = 0
    acc = 0
    for i, run in enumerate(runs):
        if band < n_bands - 1 and acc >= target and acc > 0:
            band += 1
            acc = 0
        out[run] = f"stat{n_bands}_{band:02d}"
        acc += by_run[run]
    return out


def event_period(run: int, events: list[dict[str, object]], args: argparse.Namespace) -> str:
    if args.period_mode == "explicit":
        mapping = getattr(args, "_run_period_map", None)
        if mapping is None or run not in mapping:
            raise ValueError(f"run {run} is absent from the explicit run-period map")
        return str(mapping[run])
    if args.period_mode == "stat_n":
        band_map = getattr(args, "_run_band_map", None)
        if band_map is None:
            band_map = build_stat_run_bands(events, args.n_run_bands)
            setattr(args, "_run_band_map", band_map)
        return band_map.get(run, f"stat{args.n_run_bands}_unknown")
    return period_key(run, events, args.period_mode)


def lowe_patch_period(run: int, events: list[dict[str, object]], args: argparse.Namespace) -> str:
    band_map = getattr(args, "_lowe_run_band_map", None)
    if band_map is None:
        low_events = [ev for ev in events if float(ev["min_e"]) < args.lowe_patch_emax]
        band_map = build_stat_run_bands(low_events if low_events else events, args.lowe_patch_bands)
        setattr(args, "_lowe_run_band_map", band_map)
    return band_map.get(run, f"lowe{args.lowe_patch_bands}_unknown")


def parse_lowe_bin_patches(args: argparse.Namespace) -> list[dict[str, float]]:
    patches = []
    for idx, item in enumerate(args.lowe_bin_patch):
        parts = item.split(":")
        if len(parts) != 5:
            raise SystemExit(f"Bad --lowe-bin-patch {item!r}; expected EMIN:EMAX:BANDS:DAMPING:CLIP")
        emin, emax, bands, damping, clip = parts
        patches.append(
            {
                "idx": float(idx),
                "emin": float(emin),
                "emax": float(emax),
                "bands": float(int(bands)),
                "damping": float(damping),
                "clip": float(clip),
            }
        )
    if patches:
        return patches
    if args.lowe_patch_bands <= 0:
        return []
    return [
        {
            "idx": 0.0,
            "emin": float(args.lowe_patch_emin),
            "emax": float(args.lowe_patch_emax),
            "bands": float(args.lowe_patch_bands),
            "damping": float(args.lowe_patch_damping),
            "clip": float(args.max_lowe_patch_mass_update_frac),
        }
    ]


def parse_lowe_run_patches(args: argparse.Namespace) -> list[dict[str, float]]:
    patches = []
    for idx, item in enumerate(args.lowe_run_patch):
        parts = item.split(":")
        if len(parts) != 6:
            raise SystemExit(f"Bad --lowe-run-patch {item!r}; expected EMIN:EMAX:RUNLO:RUNHI:DAMPING:CLIP")
        emin, emax, runlo, runhi, damping, clip = parts
        patches.append(
            {
                "idx": float(idx),
                "emin": float(emin),
                "emax": float(emax),
                "runlo": float(int(runlo)),
                "runhi": float(int(runhi)),
                "damping": float(damping),
                "clip": float(clip),
            }
        )
    return patches


def lowe_bin_patch_period(run: int, events: list[dict[str, object]], args: argparse.Namespace, patch: dict[str, float]) -> str:
    attr = f"_lowe_bin_patch_{int(patch['idx'])}_run_band_map"
    band_map = getattr(args, attr, None)
    if band_map is None:
        low_events = [
            ev
            for ev in events
            if float(ev["min_e"]) >= patch["emin"] and float(ev["min_e"]) < patch["emax"]
        ]
        band_map = build_stat_run_bands(low_events if low_events else events, int(patch["bands"]))
        setattr(args, attr, band_map)
    return band_map.get(run, f"lowe{int(patch['idx'])}_unknown")


def read_compact_events(summary_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    qa: list[dict[str, object]] = []
    for info in summary_rows:
        compact = Path(info["output_tsv"])
        total = selected = 0
        with compact.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
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
                        "low_seed": int(row["seed1_block"]) if low_slot == 0 else int(row["seed2_block"]),
                        "low_col": int(row["seed1_col"]) if low_slot == 0 else int(row["seed2_col"]),
                        "members1": [],
                        "members2": [],
                    }
                )
                selected += 1
        qa.append(
            {
                "run": info["run"],
                "segment": int(info["segment"]),
                "total_rows": total,
                "selected_events": selected,
                "missing_member_events": "not_checked_compact_only",
            }
        )
    return events, qa


def energy_center(eb: tuple[float, float]) -> float:
    return 0.5 * (eb[0] + eb[1])


def parse_bin_damping(items: list[str]) -> list[tuple[float, float, float]]:
    out = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise SystemExit(f"Bad bin damping {item!r}; expected EMIN:EMAX:DAMPING")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def damping_for_bin(default: float, overrides: list[str], eb: tuple[float, float]) -> float:
    for lo, hi, damping in parse_bin_damping(overrides):
        if eb[0] >= lo and eb[1] <= hi:
            return damping
    return default


def derive_energy_updates(
    train: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    updates: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for eb in base.ENERGY_BINS:
        label = base.bin_label(eb)
        vals = [float(ev["mass_mev"]) for ev in train if base.energy_bin(float(ev["min_e"])) == eb]
        fit = fit_peak(np.asarray(vals, dtype=float), args.fit_window[0], args.fit_window[1])
        ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_fit_entries and math.isfinite(float(fit["mu"]))
        full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
        damping = damping_for_bin(args.energy_damping, args.energy_bin_damping, eb)
        applied = base.clamp(full**damping, args.max_energy_mass_update_frac) if ok else 1.0
        rows.append(
            {
                "layer": "energy",
                "key": label,
                "entries": len(vals),
                "fit_entries": int(fit["entries"]),
                "fit_ok": int(fit["fit_ok"]),
                "fit_mu_mev": fit["mu"],
                "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
                "fit_sigma_mev": fit["sigma"],
                "factor_full": full,
                "damping": damping,
                "factor_applied": applied,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[label] = applied
    return updates, rows


def derive_period_energy_updates(
    train: list[dict[str, object]],
    base_masses: dict[int, float],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], list[dict[str, object]]]:
    by_period_bin: dict[tuple[str, tuple[float, float]], list[float]] = defaultdict(list)
    for i, ev in enumerate(train):
        eb = base.energy_bin(float(ev["min_e"]))
        if eb is None:
            continue
        pk = event_period(int(ev["run"]), train, args)
        by_period_bin[(pk, eb)].append(base_masses[i])

    raw_rows: list[dict[str, object]] = []
    raw_factors: dict[tuple[str, tuple[float, float]], tuple[float, int, float, float]] = {}
    periods = sorted({event_period(int(ev["run"]), train, args) for ev in train})
    for pk in periods:
        for eb in base.ENERGY_BINS:
            label = base.bin_label(eb)
            vals = np.asarray(by_period_bin.get((pk, eb), []), dtype=float)
            fit = fit_peak(vals, args.fit_window[0], args.fit_window[1])
            ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_fit_entries and math.isfinite(float(fit["mu"]))
            full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
            damping = damping_for_bin(args.period_energy_damping, args.period_energy_bin_damping, eb)
            damped = full**damping if ok else 1.0
            clipped = base.clamp(damped, args.max_period_energy_mass_update_frac)
            raw_rows.append(
                {
                    "layer": "period_energy_raw",
                    "period": pk,
                    "key": label,
                    "entries": len(vals),
                    "fit_entries": int(fit["entries"]),
                    "fit_ok": int(fit["fit_ok"]),
                    "fit_mu_mev": fit["mu"],
                    "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
                    "fit_sigma_mev": fit["sigma"],
                    "factor_full": full,
                    "damping": damping,
                    "factor_damped": damped,
                    "factor_clipped": clipped,
                    "support_pass": int(ok),
                }
            )
            if ok:
                raw_factors[(pk, eb)] = (clipped, int(fit["entries"]), float(fit["mu"]), float(fit["sigma"]))

    updates: dict[tuple[str, str], float] = {}
    smooth_rows: list[dict[str, object]] = []
    for pk in periods:
        points = []
        for eb in base.ENERGY_BINS:
            item = raw_factors.get((pk, eb))
            if item is None:
                continue
            factor, entries, _mu, _sigma = item
            points.append((math.log(energy_center(eb)), math.log(factor), max(1.0, math.sqrt(entries)), eb, factor, entries))

        if len(points) >= 2:
            deg = min(args.smooth_degree, len(points) - 1)
            x = np.asarray([p[0] for p in points], dtype=float)
            y = np.asarray([p[1] for p in points], dtype=float)
            w = np.asarray([p[2] for p in points], dtype=float)
            coeff = np.polyfit(x, y, deg=deg, w=w)
            poly = np.poly1d(coeff)
            mode = f"poly{deg}"
        else:
            poly = None
            mode = "identity_or_single_bin"

        for eb in base.ENERGY_BINS:
            label = base.bin_label(eb)
            raw = raw_factors.get((pk, eb), (1.0, 0, math.nan, math.nan))[0]
            if poly is not None:
                smooth = math.exp(float(poly(math.log(energy_center(eb)))))
            elif (pk, eb) in raw_factors:
                smooth = raw
            else:
                smooth = 1.0
            smooth = base.clamp(smooth, args.max_period_energy_mass_update_frac)
            updates[(pk, label)] = smooth
            smooth_rows.append(
                {
                    "layer": "period_energy_smoothed",
                    "period": pk,
                    "key": label,
                    "entries": raw_factors.get((pk, eb), (1.0, 0, math.nan, math.nan))[1],
                    "raw_factor": raw,
                    "factor_applied": smooth,
                    "smooth_mode": mode,
                    "support_pass": int((pk, eb) in raw_factors),
                }
            )

    return updates, raw_rows + smooth_rows


def apply_period_energy(
    mass: float,
    ev: dict[str, object],
    events_for_periods: list[dict[str, object]],
    updates: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> float:
    eb = base.energy_bin(float(ev["min_e"]))
    if eb is None:
        return mass
    pk = event_period(int(ev["run"]), events_for_periods, args)
    return mass * updates.get((pk, base.bin_label(eb)), 1.0)


def derive_lowe_patch_updates(
    train: list[dict[str, object]],
    base_masses: dict[int, float],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], list[dict[str, object]]]:
    patches = parse_lowe_bin_patches(args)
    if not patches:
        return {}, []

    by_band_bin: dict[tuple[int, str, tuple[float, float]], list[float]] = defaultdict(list)
    for i, ev in enumerate(train):
        eb = base.energy_bin(float(ev["min_e"]))
        if eb is None:
            continue
        for patch in patches:
            if eb[0] < patch["emin"] or eb[1] > patch["emax"]:
                continue
            pk = lowe_bin_patch_period(int(ev["run"]), train, args, patch)
            by_band_bin[(int(patch["idx"]), pk, eb)].append(base_masses[i])

    updates: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    for patch in patches:
        patch_id = int(patch["idx"])
        periods = sorted(
            {
                lowe_bin_patch_period(int(ev["run"]), train, args, patch)
                for ev in train
                if float(ev["min_e"]) >= patch["emin"] and float(ev["min_e"]) < patch["emax"]
            }
        )
        for pk in periods:
            for eb in base.ENERGY_BINS:
                if eb[0] < patch["emin"] or eb[1] > patch["emax"]:
                    continue
                label = base.bin_label(eb)
                vals = np.asarray(by_band_bin.get((patch_id, pk, eb), []), dtype=float)
                fit = fit_peak(vals, args.fit_window[0], args.fit_window[1])
                ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_fit_entries and math.isfinite(float(fit["mu"]))
                full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
                applied = base.clamp(full**patch["damping"], patch["clip"])
                rows.append(
                    {
                        "layer": "lowe_patch",
                        "patch_id": patch_id,
                        "patch_emin": patch["emin"],
                        "patch_emax": patch["emax"],
                        "patch_bands": int(patch["bands"]),
                        "period": pk,
                        "key": label,
                        "entries": len(vals),
                        "fit_entries": int(fit["entries"]),
                        "fit_ok": int(fit["fit_ok"]),
                        "fit_mu_mev": fit["mu"],
                        "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
                        "fit_sigma_mev": fit["sigma"],
                        "factor_full": full,
                        "factor_applied": applied,
                        "support_pass": int(ok),
                    }
                )
                if ok:
                    updates[(pk, label)] = applied
    return updates, rows


def apply_lowe_patch(
    mass: float,
    ev: dict[str, object],
    events_for_periods: list[dict[str, object]],
    updates: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> float:
    patches = parse_lowe_bin_patches(args)
    if not patches:
        return mass
    eb = base.energy_bin(float(ev["min_e"]))
    if eb is None:
        return mass
    out = mass
    for patch in patches:
        if eb[0] < patch["emin"] or eb[1] > patch["emax"]:
            continue
        pk = lowe_bin_patch_period(int(ev["run"]), events_for_periods, args, patch)
        out *= updates.get((pk, base.bin_label(eb)), 1.0)
    return out


def derive_lowe_run_patch_updates(
    train: list[dict[str, object]],
    base_masses: dict[int, float],
    args: argparse.Namespace,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    patches = parse_lowe_run_patches(args)
    if not patches:
        return {}, []
    updates: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for patch in patches:
        vals = []
        for i, ev in enumerate(train):
            eb = base.energy_bin(float(ev["min_e"]))
            run = int(ev["run"])
            if eb is None:
                continue
            if eb[0] < patch["emin"] or eb[1] > patch["emax"]:
                continue
            if run < int(patch["runlo"]) or run > int(patch["runhi"]):
                continue
            vals.append(base_masses[i])
        arr = np.asarray(vals, dtype=float)
        fit = fit_peak(arr, args.fit_window[0], args.fit_window[1])
        ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_fit_entries and math.isfinite(float(fit["mu"]))
        full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
        applied = base.clamp(full**patch["damping"], patch["clip"])
        patch_id = int(patch["idx"])
        rows.append(
            {
                "layer": "lowe_run_patch",
                "patch_id": patch_id,
                "patch_emin": patch["emin"],
                "patch_emax": patch["emax"],
                "runlo": int(patch["runlo"]),
                "runhi": int(patch["runhi"]),
                "entries": len(vals),
                "fit_entries": int(fit["entries"]),
                "fit_ok": int(fit["fit_ok"]),
                "fit_mu_mev": fit["mu"],
                "fit_mu_residual_mev": fit["mu"] - M_PI0_MEV if math.isfinite(float(fit["mu"])) else math.nan,
                "fit_sigma_mev": fit["sigma"],
                "factor_full": full,
                "factor_applied": applied,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[patch_id] = applied
    return updates, rows


def apply_lowe_run_patch(
    mass: float,
    ev: dict[str, object],
    updates: dict[int, float],
    args: argparse.Namespace,
) -> float:
    out = mass
    eb = base.energy_bin(float(ev["min_e"]))
    if eb is None:
        return out
    for patch in parse_lowe_run_patches(args):
        run = int(ev["run"])
        if eb[0] < patch["emin"] or eb[1] > patch["emax"]:
            continue
        if run < int(patch["runlo"]) or run > int(patch["runhi"]):
            continue
        out *= updates.get(int(patch["idx"]), 1.0)
    return out


def derive_seed_updates_regularized(
    train: list[dict[str, object]],
    base_masses: dict[int, float],
    args: argparse.Namespace,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    if args.seed_regularization == "legacy":
        return base.derive_seed_updates(train, base_masses, args)

    left_cols = set(args.left_cols)
    by_seed: dict[int, list[float]] = defaultdict(list)
    eligible_residuals: list[float] = []
    for i, ev in enumerate(train):
        if int(ev["low_col"]) in left_cols:
            continue
        resid = base_masses[i] - M_PI0_MEV
        by_seed[int(ev["low_seed"])].append(resid)
        eligible_residuals.append(resid)

    global_center = float(np.median(np.asarray(eligible_residuals, dtype=float))) if eligible_residuals else 0.0
    updates: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for seed, vals in sorted(by_seed.items()):
        residuals = np.asarray(vals, dtype=float)
        median_residual = float(np.median(residuals)) if residuals.size else math.nan
        centered = median_residual - global_center if math.isfinite(median_residual) else math.nan
        shrink = len(vals) / (len(vals) + args.seed_shrink_events) if len(vals) > 0 else 0.0
        regularized = centered * shrink if math.isfinite(centered) else math.nan
        ok = len(vals) >= args.min_seed_events and math.isfinite(regularized)
        full_energy = (M_PI0_MEV / (M_PI0_MEV + regularized)) ** 2 if ok else 1.0
        applied_energy = base.clamp(full_energy**args.seed_damping, args.max_seed_energy_update_frac)
        rows.append(
            {
                "layer": "seed_regularized",
                "key": seed,
                "entries": len(vals),
                "median_residual_mev": median_residual,
                "global_center_residual_mev": global_center,
                "centered_residual_mev": centered,
                "shrink_factor": shrink,
                "regularized_residual_mev": regularized,
                "factor_full": full_energy,
                "factor_applied": applied_energy,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[seed] = applied_energy
    return updates, rows


def derive_member_updates_regularized(
    train: list[dict[str, object]],
    base_masses: dict[int, float],
    args: argparse.Namespace,
) -> tuple[dict[int, float], list[dict[str, object]]]:
    if args.member_regularization == "legacy":
        return base.derive_member_updates(train, base_masses, args)

    left_cols = set(args.left_cols)
    weighted_resid: dict[int, list[tuple[float, float]]] = defaultdict(list)
    global_num = 0.0
    global_den = 0.0
    for i, ev in enumerate(train):
        resid = base_masses[i] - M_PI0_MEV
        for cluster_key in ("members1", "members2"):
            for member in ev[cluster_key]:  # type: ignore[index]
                if int(member["col"]) in left_cols:
                    continue
                w = max(0.0, float(member["member_fraction"]))
                if w <= 0.0:
                    continue
                weighted_resid[int(member["block"])].append((resid, w))
                global_num += resid * w
                global_den += w
    global_center = global_num / global_den if global_den > 0 else 0.0

    updates: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for block, vals in sorted(weighted_resid.items()):
        wsum = sum(w for _r, w in vals)
        mean = sum(r * w for r, w in vals) / wsum if wsum > 0 else math.nan
        centered = mean - global_center if math.isfinite(mean) else math.nan
        shrink = wsum / (wsum + args.member_shrink_weight) if wsum > 0 else 0.0
        regularized = centered * shrink if math.isfinite(centered) else math.nan
        ok = wsum >= args.min_member_weight and math.isfinite(regularized)
        full_energy = (M_PI0_MEV / (M_PI0_MEV + regularized)) ** 2 if ok else 1.0
        applied_energy = base.clamp(full_energy**args.member_damping, args.max_member_energy_update_frac)
        rows.append(
            {
                "layer": "member_regularized",
                "key": block,
                "entries": len(vals),
                "weight_sum": wsum,
                "weighted_mean_residual_mev": mean,
                "global_center_residual_mev": global_center,
                "centered_residual_mev": centered,
                "shrink_factor": shrink,
                "regularized_residual_mev": regularized,
                "factor_full": full_energy,
                "factor_applied": applied_energy,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[block] = applied_energy
    return updates, rows


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
        label = f"emin_{base.bin_label(eb)}"
        groups.append((label, [i for i, ev in enumerate(events) if base.energy_bin(float(ev["min_e"])) == eb]))
    if args.fit_detail == "run_energy":
        for eb in base.ENERGY_BINS:
            e_label = base.bin_label(eb)
            for run in sorted({int(ev["run"]) for ev in events}):
                groups.append(
                    (
                        f"run_{run}_emin_{e_label}",
                        [i for i, ev in enumerate(events) if int(ev["run"]) == run and base.energy_bin(float(ev["min_e"])) == eb],
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
    name: str,
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    all_events: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]]]:
    update_rows: list[dict[str, object]] = []
    if hasattr(args, "_run_band_map"):
        delattr(args, "_run_band_map")
    if hasattr(args, "_lowe_run_band_map"):
        delattr(args, "_lowe_run_band_map")
    for key in list(vars(args)):
        if key.startswith("_lowe_bin_patch_"):
            delattr(args, key)

    energy_updates, rows = derive_energy_updates(train, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_energy = {i: base.apply_energy(float(ev["mass_mev"]), ev, energy_updates) for i, ev in enumerate(train)}
    period_energy_updates, rows = derive_period_energy_updates(train, train_energy, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_period_energy = {
        i: apply_period_energy(train_energy[i], ev, train, period_energy_updates, args) for i, ev in enumerate(train)
    }
    lowe_patch_updates, rows = derive_lowe_patch_updates(train, train_period_energy, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_lowe_patch = {
        i: apply_lowe_patch(train_period_energy[i], ev, train, lowe_patch_updates, args) for i, ev in enumerate(train)
    }
    lowe_run_patch_updates, rows = derive_lowe_run_patch_updates(train, train_lowe_patch, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_lowe_run_patch = {
        i: apply_lowe_run_patch(train_lowe_patch[i], ev, lowe_run_patch_updates, args) for i, ev in enumerate(train)
    }
    seed_updates: dict[int, float] = {}
    seed_updates_by_iter: dict[int, dict[int, float]] = {}
    seed_iterations = max(1, args.seed_iterations)
    for iteration in range(1, seed_iterations + 1):
        train_seed_input = {i: base.apply_seed(train_lowe_run_patch[i], ev, seed_updates) for i, ev in enumerate(train)}
        seed_delta, rows = derive_seed_updates_regularized(train, train_seed_input, args)
        update_rows.extend({"fold": name, "iteration": iteration, **row} for row in rows)
        for block, scale in seed_delta.items():
            seed_updates[block] = seed_updates.get(block, 1.0) * scale
        seed_updates_by_iter[iteration] = dict(seed_updates)

    train_seed = {i: base.apply_seed(train_lowe_run_patch[i], ev, seed_updates) for i, ev in enumerate(train)}
    member_updates_by_iter: dict[int, dict[int, float]] = {}
    if args.no_member:
        member_updates: dict[int, float] = {}
    else:
        member_updates = {}
        member_iterations = max(1, args.member_iterations)
        for iteration in range(1, member_iterations + 1):
            train_member_input = {
                i: base.apply_member(train_seed[i], ev, member_updates) for i, ev in enumerate(train)
            }
            member_delta, rows = derive_member_updates_regularized(train, train_member_input, args)
            update_rows.extend({"fold": name, "iteration": iteration, **row} for row in rows)
            for block, scale in member_delta.items():
                member_updates[block] = member_updates.get(block, 1.0) * scale
            member_updates_by_iter[iteration] = dict(member_updates)

    masses: dict[str, list[float]] = {model: [] for model in model_names(args)}
    for ev in test:
        raw = float(ev["mass_mev"])
        energy = base.apply_energy(raw, ev, energy_updates)
        period_energy = apply_period_energy(energy, ev, train, period_energy_updates, args)
        lowe_patch = apply_lowe_patch(period_energy, ev, train, lowe_patch_updates, args)
        lowe_run_patch = apply_lowe_run_patch(lowe_patch, ev, lowe_run_patch_updates, args)
        seed = lowe_run_patch
        for iteration in range(1, seed_iterations + 1):
            seed = base.apply_seed(lowe_run_patch, ev, seed_updates_by_iter[iteration])
            masses[f"hao_period_energy_seed_iter_{iteration}"].append(seed)
        member = seed
        if not args.no_member:
            for iteration in range(1, max(1, args.member_iterations) + 1):
                member = base.apply_member(seed, ev, member_updates_by_iter[iteration])
                masses[f"hao_period_energy_seed_member_iter_{iteration}"].append(member)
        masses["hao_raw"].append(raw)
        masses["hao_energy"].append(energy)
        masses["hao_period_energy"].append(period_energy)
        masses["hao_period_energy_lowe_patch"].append(lowe_run_patch)
        masses["hao_period_energy_seed"].append(seed)
        if not args.no_member:
            masses["hao_period_energy_seed_member"].append(member)

    fit_rows = fit_eval_rows(name, "test", test, masses, args)
    return fit_rows, update_rows, masses


def plot_trends(rows: list[dict[str, object]], outdir: Path) -> None:
    pdf = outdir / "hao_period_energy_smoothed_seed_trends.pdf"
    groups = ["pooled"] + [f"emin_{base.bin_label(eb)}" for eb in base.ENERGY_BINS]
    with PdfPages(pdf) as pp:
        for group in groups:
            fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
            for model in model_names(args):
                vals = [
                    row
                    for row in rows
                    if row["sample"] == "test" and row["group"] == group and row["model"] == model and int(row["fit_ok"]) == 1
                ]
                if not vals:
                    continue
                labels = [str(row["fold"]) for row in vals]
                axs[0].plot(labels, [float(row["fit_mu_residual_mev"]) for row in vals], "-o", label=model)
                axs[1].plot(labels, [float(row["fit_sigma_mev"]) for row in vals], "-o", label=model)
            axs[0].axhline(0.0, color="black", lw=0.9)
            axs[0].set_ylabel("fit peak - PDG (MeV)")
            axs[0].set_title(f"Hao-start period-energy smoothing held-out fits: {group}")
            axs[1].set_ylabel("fit sigma (MeV)")
            axs[1].set_xlabel("held-out fold")
            for ax in axs:
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def plot_run_energy_trends(rows: list[dict[str, object]], outdir: Path) -> None:
    pdf = outdir / "hao_period_energy_smoothed_seed_run_energy_trends.pdf"
    with PdfPages(pdf) as pp:
        for eb in base.ENERGY_BINS:
            e_label = base.bin_label(eb)
            fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
            for model in model_names(args):
                points = []
                for row in rows:
                    group = str(row["group"])
                    if (
                        row["fold"] == "pooled_folds"
                        and row["sample"] == "test"
                        and row["model"] == model
                        and group.endswith(f"_emin_{e_label}")
                        and group.startswith("run_")
                        and int(row["fit_ok"]) == 1
                    ):
                        run = int(group.split("_")[1])
                        points.append((run, float(row["fit_mu_residual_mev"]), float(row["fit_sigma_mev"]), int(row["fit_entries"])))
                points.sort()
                points = [p for p in points if p[3] >= 80]
                if not points:
                    continue
                axs[0].plot([p[0] for p in points], [p[1] for p in points], "-o", ms=4, lw=1.2, label=model)
                axs[1].plot([p[0] for p in points], [p[2] for p in points], "-o", ms=4, lw=1.2, label=model)
            axs[0].axhline(0.0, color="black", lw=0.9)
            axs[0].set_ylabel("fit peak - PDG (MeV)")
            axs[0].set_title(f"Hao-start smoothed period-energy: run trend, min_pair_E {e_label} GeV")
            axs[1].set_ylabel("fit sigma (MeV)")
            axs[1].set_xlabel("run")
            for ax in axs:
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def plot_period_energy_updates(rows: list[dict[str, object]], outdir: Path) -> None:
    pdf = outdir / "hao_period_energy_smoothed_update_curves.pdf"
    smooth = [r for r in rows if r.get("layer") == "period_energy_smoothed"]
    raw = [r for r in rows if r.get("layer") == "period_energy_raw"]
    with PdfPages(pdf) as pp:
        for fold in sorted({str(r["fold"]) for r in smooth}):
            fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
            for period in sorted({str(r["period"]) for r in smooth if str(r["fold"]) == fold}):
                srows = [r for r in smooth if str(r["fold"]) == fold and str(r["period"]) == period]
                rrows = [r for r in raw if str(r["fold"]) == fold and str(r["period"]) == period]
                x = [energy_center(tuple(map(float, str(r["key"]).split("-")))) for r in srows]
                y = [float(r["factor_applied"]) for r in srows]
                ax.plot(x, y, "-o", label=f"{period} smoothed")
                rx = [energy_center(tuple(map(float, str(r["key"]).split("-")))) for r in rrows if int(r["support_pass"]) == 1]
                ry = [float(r["factor_clipped"]) for r in rrows if int(r["support_pass"]) == 1]
                ax.scatter(rx, ry, s=22, alpha=0.55)
            ax.axhline(1.0, color="black", lw=0.9)
            ax.set_xlabel("min pair photon energy bin center (GeV)")
            ax.set_ylabel("mass correction factor")
            ax.set_title(f"Smoothed period-energy correction curves: {fold}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def plot_mass_pages(events: list[dict[str, object]], masses_by_model: dict[str, list[float]], outdir: Path, args: argparse.Namespace) -> None:
    pdf = outdir / "hao_period_energy_smoothed_seed_mass_fits.pdf"
    with PdfPages(pdf) as pp:
        groups = [("pooled", list(range(len(events))))]
        groups.extend(
            (f"emin_{base.bin_label(eb)}", [i for i, ev in enumerate(events) if base.energy_bin(float(ev["min_e"])) == eb])
            for eb in base.ENERGY_BINS
        )
        for group, idxs in groups:
            if not idxs:
                continue
            models = model_names(args)
            fig, axs = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.8), constrained_layout=True)
            if len(models) == 1:
                axs = [axs]
            for ax, model in zip(axs, models):
                masses = np.asarray([masses_by_model[model][i] for i in idxs], dtype=float)
                plot_fit(ax, masses, f"{group} {model}", tuple(args.hist_window), tuple(args.fit_window))
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.compact_only and not args.no_member:
        raise SystemExit("--compact-only requires --no-member")
    summary = base.load_conversion_summary(Path(args.conversion_summary))
    if args.compact_only:
        events, qa = read_compact_events(summary, args)
    else:
        events, qa = base.read_events(summary, args)
    base.write_tsv(outdir / "sidecar_quality.tsv", qa)

    all_fit_rows: list[dict[str, object]] = []
    all_update_rows: list[dict[str, object]] = []
    pooled_test_events: list[dict[str, object]] = []
    pooled_masses: dict[str, list[float]] = {model: [] for model in model_names(args)}

    for test_fold in [0, 1]:
        train = [ev for ev in events if int(ev["fold"]) != test_fold]
        test = [ev for ev in events if int(ev["fold"]) == test_fold]
        fit_rows, update_rows, masses = evaluate_fold(f"segment_parity_test_{test_fold}", train, test, events, args)
        all_fit_rows.extend(fit_rows)
        all_update_rows.extend(update_rows)
        pooled_test_events.extend(test)
        for model in model_names(args):
            pooled_masses[model].extend(masses[model])

    all_fit_rows.extend(fit_eval_rows("pooled_folds", "test", pooled_test_events, pooled_masses, args))
    base.write_tsv(outdir / "heldout_fit_metrics.tsv", all_fit_rows)
    base.write_tsv(outdir / "fold_update_tables.tsv", all_update_rows)
    if not args.skip_plots:
        plot_trends(all_fit_rows, outdir)
        if args.fit_detail == "run_energy":
            plot_run_energy_trends(all_fit_rows, outdir)
        plot_period_energy_updates(all_update_rows, outdir)
        plot_mass_pages(pooled_test_events, pooled_masses, outdir, args)
    print(outdir / "heldout_fit_metrics.tsv")
    print(outdir / "fold_update_tables.tsv")


if __name__ == "__main__":
    main()
