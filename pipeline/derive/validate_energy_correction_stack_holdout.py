#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_energy_correction_stack import (
    M_PI0_MEV,
    cluster_mode_ok,
    input_paths_from_summary,
    load_coeffs,
    metrics,
    normalize_path,
    shared_scale,
)
from timing_window_utils import parse_timing_center, passes_timing_window, resolve_timing_center


SEGMENT_RE = re.compile(r"nps_production_(\d+)_(\d+)_wf\.tsv$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Held-out validation of the shared + time-slice + block correction stack.")
    ap.add_argument("coeff_tsv")
    ap.add_argument("input_tsv", nargs="*")
    ap.add_argument("--input-summary")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mass-window", nargs=2, type=float, required=True)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--timing-center", default="auto")
    ap.add_argument("--min-e-floor", type=float, default=0.6)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--nclusters-mode", choices=["exact2", "ge2"], default="exact2")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--min-block-entries", type=int, default=50)
    ap.add_argument("--split-mode", choices=["segment_parity", "run_parity"], default="segment_parity")
    ap.add_argument("--damping", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return ap.parse_args()


def parse_segment(path: Path) -> tuple[int | None, int]:
    match = SEGMENT_RE.search(path.name)
    if not match:
        return None, 0
    return int(match.group(1)), int(match.group(2))


def read_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in paths:
        _, segment = parse_segment(path)
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                item = {key: float(value) for key, value in row.items()}
                item["_segment"] = float(segment)
                rows.append(item)
    return rows


def lower_energy_seed(row: dict[str, float]) -> int:
    if row["e1"] <= row["e2"]:
        return int(row.get("seed1_block", -1.0))
    return int(row.get("seed2_block", -1.0))


def select_events(
    rows: list[dict[str, float]],
    coeff: dict[str, float],
    *,
    mass_window: tuple[float, float],
    timing_window: float | None,
    timing_center: float,
    min_e_floor: float,
    nclusters_mode: str,
    emin_range: tuple[float, float],
) -> list[dict[str, float]]:
    mass_lo, mass_hi = mass_window
    emin_lo, emin_hi = emin_range
    selected: list[dict[str, float]] = []
    for row in rows:
        if not cluster_mode_ok(int(row["nclusters"]), nclusters_mode):
            continue
        if row["e1"] < 0.6 or row["e2"] < 0.6:
            continue
        if row["min_e"] < min_e_floor:
            continue
        if abs(row["x1"]) >= 29.025 or abs(row["x2"]) >= 29.025:
            continue
        if abs(row["y1"]) >= 35.475 or abs(row["y2"]) >= 35.475:
            continue
        if not passes_timing_window(row, timing_window, timing_center):
            continue
        if not (mass_lo <= row["pair_m"] <= mass_hi):
            continue
        if not (emin_lo <= row["min_e"] <= emin_hi):
            continue
        seed1 = int(row.get("seed1_block", -1.0))
        seed2 = int(row.get("seed2_block", -1.0))
        low_seed = lower_energy_seed(row)
        if seed1 < 0 or seed2 < 0 or low_seed < 0:
            continue
        shared_mass = row["pair_m"] * math.sqrt(shared_scale(coeff, row["e1"]) * shared_scale(coeff, row["e2"])) * 1000.0
        selected.append(
            {
                "run": int(row["run"]),
                "event": int(row["event"]),
                "segment": int(row["_segment"]),
                "seed1_block": seed1,
                "seed2_block": seed2,
                "low_seed_block": low_seed,
                "shared_mass_mev": shared_mass,
            }
        )
    selected.sort(key=lambda r: (int(r["run"]), int(r["segment"]), int(r["event"])))
    return selected


def fold_events(events: list[dict[str, float]], split_mode: str) -> list[tuple[str, list[dict[str, float]], list[dict[str, float]]]]:
    out = []
    for train_mod in (0, 1):
        if split_mode == "segment_parity":
            name = "segment_even_train" if train_mod == 0 else "segment_odd_train"
            train = [e for e in events if int(e["segment"]) % 2 == train_mod]
            test = [e for e in events if int(e["segment"]) % 2 != train_mod]
        elif split_mode == "run_parity":
            name = "run_even_train" if train_mod == 0 else "run_odd_train"
            train = [e for e in events if int(e["run"]) % 2 == train_mod]
            test = [e for e in events if int(e["run"]) % 2 != train_mod]
        else:
            raise ValueError(split_mode)
        out.append((name, train, test))
    return out


def build_slices(events: list[dict[str, float]], n_slices: int) -> list[dict[str, object]]:
    by_run: dict[int, int] = defaultdict(int)
    for event in events:
        by_run[int(event["run"])] += 1
    runs = sorted(by_run)
    if not runs:
        return []

    n_slices = max(1, min(n_slices, len(runs)))
    target = max(1.0, len(events) / n_slices)
    slices: list[list[int]] = []
    current: list[int] = []
    current_n = 0
    remaining_slices = n_slices
    for i, run in enumerate(runs):
        current.append(run)
        current_n += by_run[run]
        remaining_runs = len(runs) - i - 1
        if current_n >= target and remaining_runs >= remaining_slices - 1:
            slices.append(current)
            current = []
            current_n = 0
            remaining_slices -= 1
    if current:
        slices.append(current)

    out: list[dict[str, object]] = []
    for idx, slice_runs in enumerate(slices):
        run_set = set(slice_runs)
        members = [event for event in events if int(event["run"]) in run_set]
        mean_mass = float(np.mean([float(e["shared_mass_mev"]) for e in members])) if members else float("nan")
        out.append(
            {
                "slice": idx,
                "run_min": min(slice_runs),
                "run_max": max(slice_runs),
                "runs": slice_runs,
                "events": len(members),
                "time_slice_scale": M_PI0_MEV / mean_mass if math.isfinite(mean_mass) else 1.0,
                "mean_mass_mev": mean_mass,
            }
        )
    return out


def time_scale_for_run(slices: list[dict[str, object]], run: int) -> tuple[float, int, str]:
    for sl in slices:
        if int(sl["run_min"]) <= run <= int(sl["run_max"]):
            return float(sl["time_slice_scale"]), int(sl["slice"]), "range"
    nearest = min(slices, key=lambda sl: min(abs(run - int(sl["run_min"])), abs(run - int(sl["run_max"]))))
    return float(nearest["time_slice_scale"]), int(nearest["slice"]), "nearest"


def derive_block_scales(events: list[dict[str, float]], slices: list[dict[str, object]], min_entries: int) -> dict[int, dict[str, float]]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for event in events:
        time_scale, _, _ = time_scale_for_run(slices, int(event["run"]))
        residual = float(event["shared_mass_mev"]) * time_scale - M_PI0_MEV
        by_block[int(event["low_seed_block"])].append(residual)

    out: dict[int, dict[str, float]] = {}
    for block, vals in by_block.items():
        if len(vals) < min_entries:
            continue
        residual = float(np.mean(vals))
        out[block] = {
            "entries": len(vals),
            "mean_residual_mev": residual,
            "block_scale_full": (M_PI0_MEV / max(1e-6, M_PI0_MEV + residual)) ** 2,
        }
    return out


def evaluate(
    events: list[dict[str, float]],
    slices: list[dict[str, object]],
    block_scales: dict[int, dict[str, float]],
    damping_values: list[float],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    residuals: dict[str, list[float]] = {"shared": [], "shared_time": []}
    for damping in damping_values:
        residuals[f"shared_time_block_damp_{damping:g}"] = []

    counters = defaultdict(int)
    for event in events:
        time_scale, _, mode = time_scale_for_run(slices, int(event["run"]))
        counters[f"time_scale_{mode}"] += 1
        shared_mass = float(event["shared_mass_mev"])
        shared_time_mass = shared_mass * time_scale
        residuals["shared"].append(shared_mass - M_PI0_MEV)
        residuals["shared_time"].append(shared_time_mass - M_PI0_MEV)

        block1 = int(event["seed1_block"])
        block2 = int(event["seed2_block"])
        has1 = block1 in block_scales
        has2 = block2 in block_scales
        if has1 and has2:
            counters["both_block_scales"] += 1
        elif has1 or has2:
            counters["one_block_scale"] += 1
        else:
            counters["no_block_scale"] += 1
        full1 = float(block_scales[block1]["block_scale_full"]) if has1 else 1.0
        full2 = float(block_scales[block2]["block_scale_full"]) if has2 else 1.0
        for damping in damping_values:
            mass = shared_time_mass * math.sqrt(math.exp(damping * math.log(full1)) * math.exp(damping * math.log(full2)))
            residuals[f"shared_time_block_damp_{damping:g}"].append(mass - M_PI0_MEV)
    return residuals, dict(counters)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    input_paths = [normalize_path(p) for p in args.input_tsv]
    if args.input_summary:
        input_paths.extend(input_paths_from_summary(Path(args.input_summary)))
    if not input_paths:
        raise SystemExit("no input TSVs provided")

    rows = read_rows(input_paths)
    coeff = load_coeffs(Path(args.coeff_tsv))
    timing_basis = [
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
    ]
    timing_center = resolve_timing_center(timing_basis, parse_timing_center(args.timing_center))
    events = select_events(
        rows,
        coeff,
        mass_window=(args.mass_window[0], args.mass_window[1]),
        timing_window=args.timing_window,
        timing_center=timing_center,
        min_e_floor=args.min_e_floor,
        nclusters_mode=args.nclusters_mode,
        emin_range=(args.emin_range[0], args.emin_range[1]),
    )

    damping_values = sorted(set(args.damping))
    fold_metric_rows = []
    fold_summary_rows = []
    pooled_test: dict[str, list[float]] = defaultdict(list)

    with (outdir / "fold_time_slice_scales.tsv").open("w", newline="") as f_time, (
        outdir / "fold_block_scales.tsv"
    ).open("w", newline="") as f_block:
        time_writer = csv.writer(f_time, delimiter="\t")
        time_writer.writerow(["fold", "slice", "run_min", "run_max", "runs", "events", "mean_mass_mev", "time_slice_scale"])
        block_writer = csv.writer(f_block, delimiter="\t")
        block_writer.writerow(["fold", "seed_block", "entries", "mean_residual_mev", "block_scale_full"])

        for fold, train, test in fold_events(events, args.split_mode):
            slices = build_slices(train, args.n_slices)
            block_scales = derive_block_scales(train, slices, args.min_block_entries)
            for sl in slices:
                time_writer.writerow(
                    [
                        fold,
                        sl["slice"],
                        sl["run_min"],
                        sl["run_max"],
                        len(sl["runs"]),  # type: ignore[arg-type]
                        sl["events"],
                        sl["mean_mass_mev"],
                        sl["time_slice_scale"],
                    ]
                )
            for block, row in sorted(block_scales.items()):
                block_writer.writerow([fold, block, row["entries"], row["mean_residual_mev"], row["block_scale_full"]])

            for sample_name, sample in [("train", train), ("test", test)]:
                residuals, counters = evaluate(sample, slices, block_scales, damping_values)
                if sample_name == "test":
                    for model, vals in residuals.items():
                        pooled_test[model].extend(vals)
                fold_summary_rows.append(
                    [
                        fold,
                        sample_name,
                        len(sample),
                        len(slices),
                        len(block_scales),
                        counters.get("time_scale_range", 0),
                        counters.get("time_scale_nearest", 0),
                        counters.get("both_block_scales", 0),
                        counters.get("one_block_scale", 0),
                        counters.get("no_block_scale", 0),
                    ]
                )
                for model, vals in residuals.items():
                    row = metrics(vals)
                    fold_metric_rows.append(
                        [fold, sample_name, model, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]]
                    )

    with (outdir / "fold_metrics.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["fold", "sample", "model", "entries", "mean_mev", "mae_mev", "rmse_mev", "sigma_mev"])
        writer.writerows(fold_metric_rows)

    with (outdir / "fold_summary.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "fold",
                "sample",
                "events",
                "time_slices",
                "block_scales",
                "events_with_time_scale_by_range",
                "events_with_time_scale_by_nearest",
                "events_with_both_block_scales",
                "events_with_one_block_scale",
                "events_without_block_scale",
            ]
        )
        writer.writerows(fold_summary_rows)

    with (outdir / "pooled_test_metrics.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["model", "entries", "mean_mev", "mae_mev", "rmse_mev", "sigma_mev"])
        for model, vals in pooled_test.items():
            row = metrics(vals)
            writer.writerow([model, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]])

    summary = [
        "Held-out correction-stack validation",
        f"input_tsvs: {len(input_paths)}",
        f"selected_events: {len(events)}",
        f"split_mode: {args.split_mode}",
        f"mass_window: {args.mass_window[0]:.6f} {args.mass_window[1]:.6f}",
        f"timing_window: {args.timing_window}",
        f"timing_center_arg: {args.timing_center}",
        f"timing_center_resolved: {timing_center:.6f}",
        f"min_e_floor: {args.min_e_floor:.6f}",
        f"emin_range: {args.emin_range[0]:.6f} {args.emin_range[1]:.6f}",
        f"nclusters_mode: {args.nclusters_mode}",
        f"n_slices: {args.n_slices}",
        f"min_block_entries: {args.min_block_entries}",
        "",
        "This validates the time-slice and block layers with the existing shared curve held fixed.",
        "For final constants, repeat after refitting the shared curve inside each training fold.",
    ]
    (outdir / "holdout_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
