#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from timing_window_utils import parse_timing_center, passes_timing_window, resolve_timing_center


M_PI0_MEV = 134.9766


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prototype the correction stack "
            "E_corr = E_raw * shared_curve(E) * time_slice_scale(run) * block_scale(block)."
        )
    )
    ap.add_argument("coeff_tsv", help="Shared-curve coefficient TSV.")
    ap.add_argument("time_slice_tsv", help="common_mode_slice_scales.tsv from the time-sliced diagnostic.")
    ap.add_argument("block_residual_tsv", help="common_mode_block_residuals.tsv from the time-sliced diagnostic.")
    ap.add_argument("input_tsv", nargs="*", help="Raw-WF pair TSV inputs.")
    ap.add_argument("--input-summary", help="File containing input_tsv: lines.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mass-window", nargs=2, type=float, required=True)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--timing-center", default="auto")
    ap.add_argument("--min-e-floor", type=float, default=0.6)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--nclusters-mode", choices=["exact2", "ge2"], default="exact2")
    ap.add_argument("--min-block-entries", type=int, default=50)
    ap.add_argument(
        "--damping",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Log-space damping values for the block correction. 0 means shared+time only.",
    )
    return ap.parse_args()


def normalize_path(path: str) -> Path:
    return Path(path).expanduser()


def input_paths_from_summary(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("input_tsv:"):
            paths.append(normalize_path(line.split(":", 1)[1].strip()))
    return paths


def load_coeffs(path: Path) -> dict[str, float]:
    coeff: dict[str, float] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("value", "") != "":
                coeff[row["key"]] = float(row["value"])
    return coeff


def shared_scale(coeff: dict[str, float], e: float) -> float:
    ee = max(e, 1e-6)
    return math.exp(coeff["c0"] + coeff["c1"] / ee + coeff["c2"] * math.log(ee))


def load_time_scales(path: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append(
                {
                    "slice": int(row["slice"]),
                    "run_min": int(row["run_min"]),
                    "run_max": int(row["run_max"]),
                    "scale": float(row["mass_scale_to_pi0"]),
                    "events": int(row["events"]),
                }
            )
    return out


def time_scale_for_run(time_scales: list[dict[str, float]], run: int) -> tuple[float, int | None]:
    for row in time_scales:
        if int(row["run_min"]) <= run <= int(row["run_max"]):
            return float(row["scale"]), int(row["slice"])
    return 1.0, None


def load_block_scales(path: Path, *, min_entries: int) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entries = int(row["entries"])
            if entries < min_entries:
                continue
            residual = float(row["mean_residual_mev"])
            full_scale = (M_PI0_MEV / max(1e-6, M_PI0_MEV + residual)) ** 2
            block = int(row["seed_block"])
            out[block] = {
                "row": int(row["row"]),
                "col": int(row["col"]),
                "entries": entries,
                "mean_residual_mev": residual,
                "block_scale_full": full_scale,
            }
    return out


def cluster_mode_ok(nclusters: int, mode: str) -> bool:
    if mode == "exact2":
        return nclusters == 2
    if mode == "ge2":
        return nclusters >= 2
    raise ValueError(f"unsupported nclusters mode: {mode}")


def metrics(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"entries": 0, "mean_mev": float("nan"), "mae_mev": float("nan"), "rmse_mev": float("nan"), "sigma_mev": float("nan")}
    return {
        "entries": int(arr.size),
        "mean_mev": float(np.mean(arr)),
        "mae_mev": float(np.mean(np.abs(arr))),
        "rmse_mev": float(np.sqrt(np.mean(arr**2))),
        "sigma_mev": float(np.std(arr)),
    }


def read_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append({key: float(value) for key, value in row.items()})
    return rows


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    input_paths = [normalize_path(p) for p in args.input_tsv]
    if args.input_summary:
        input_paths.extend(input_paths_from_summary(Path(args.input_summary)))
    if not input_paths:
        raise SystemExit("no input TSVs provided")

    coeff = load_coeffs(Path(args.coeff_tsv))
    time_scales = load_time_scales(Path(args.time_slice_tsv))
    block_scales = load_block_scales(Path(args.block_residual_tsv), min_entries=args.min_block_entries)
    rows = read_rows(input_paths)

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
    mass_lo, mass_hi = args.mass_window
    emin_lo, emin_hi = args.emin_range

    residuals_by_model: dict[str, list[float]] = {"shared": [], "shared_time": []}
    damping_values = sorted(set(args.damping))
    for damping in damping_values:
        residuals_by_model[f"shared_time_block_damp_{damping:g}"] = []

    selected = 0
    events_with_both_block_scales = 0
    events_with_one_block_scale = 0
    events_without_block_scale = 0
    runs_without_time_scale: set[int] = set()
    slices_seen: set[int] = set()

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
        if not (emin_lo <= r["min_e"] <= emin_hi):
            continue

        run = int(r["run"])
        tscale, slice_id = time_scale_for_run(time_scales, run)
        if slice_id is None:
            runs_without_time_scale.add(run)
        else:
            slices_seen.add(slice_id)

        shared_mass = r["pair_m"] * math.sqrt(shared_scale(coeff, r["e1"]) * shared_scale(coeff, r["e2"])) * 1000.0
        shared_time_mass = shared_mass * tscale
        residuals_by_model["shared"].append(shared_mass - M_PI0_MEV)
        residuals_by_model["shared_time"].append(shared_time_mass - M_PI0_MEV)

        block1 = int(r.get("seed1_block", -1.0))
        block2 = int(r.get("seed2_block", -1.0))
        has1 = block1 in block_scales
        has2 = block2 in block_scales
        if has1 and has2:
            events_with_both_block_scales += 1
        elif has1 or has2:
            events_with_one_block_scale += 1
        else:
            events_without_block_scale += 1

        full1 = block_scales[block1]["block_scale_full"] if has1 else 1.0
        full2 = block_scales[block2]["block_scale_full"] if has2 else 1.0
        for damping in damping_values:
            b1 = math.exp(damping * math.log(full1))
            b2 = math.exp(damping * math.log(full2))
            mass = shared_time_mass * math.sqrt(b1 * b2)
            residuals_by_model[f"shared_time_block_damp_{damping:g}"].append(mass - M_PI0_MEV)
        selected += 1

    with (outdir / "correction_stack_metrics.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["model", "entries", "mean_mev", "mae_mev", "rmse_mev", "sigma_mev"])
        for model, vals in residuals_by_model.items():
            row = metrics(vals)
            writer.writerow([model, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]])

    with (outdir / "block_scales.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seed_block", "row", "col", "entries", "mean_residual_mev", "block_scale_full"])
        for block, row in sorted(block_scales.items()):
            writer.writerow(
                [
                    block,
                    int(row["row"]),
                    int(row["col"]),
                    int(row["entries"]),
                    row["mean_residual_mev"],
                    row["block_scale_full"],
                ]
            )

    with (outdir / "time_slice_scales.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["slice", "run_min", "run_max", "events", "time_slice_scale"])
        for row in time_scales:
            writer.writerow([row["slice"], row["run_min"], row["run_max"], row["events"], row["scale"]])

    summary = [
        "Correction stack prototype",
        f"input_tsvs: {len(input_paths)}",
        f"selected_events: {selected}",
        f"mass_window: {mass_lo:.6f} {mass_hi:.6f}",
        f"timing_window: {args.timing_window}",
        f"timing_center_arg: {args.timing_center}",
        f"timing_center_resolved: {timing_center:.6f}",
        f"min_e_floor: {args.min_e_floor:.6f}",
        f"emin_range: {emin_lo:.6f} {emin_hi:.6f}",
        f"nclusters_mode: {args.nclusters_mode}",
        f"time_slices_loaded: {len(time_scales)}",
        f"time_slices_used: {len(slices_seen)}",
        f"runs_without_time_scale: {len(runs_without_time_scale)}",
        f"block_scales_loaded: {len(block_scales)}",
        f"min_block_entries: {args.min_block_entries}",
        f"events_with_both_block_scales: {events_with_both_block_scales}",
        f"events_with_one_block_scale: {events_with_one_block_scale}",
        f"events_without_block_scale: {events_without_block_scale}",
        "",
        "Block scale convention:",
        "  full block_scale = (m_pi0 / (m_pi0 + mean_residual_after_shared_and_time))^2",
        "  damping is applied in log space: applied_scale = full_scale^damping",
        "  event pair mass receives sqrt(block_scale(seed1) * block_scale(seed2)).",
        "",
        "Use this as a closure/prototype table, not final constants, until tested on independent runs.",
    ]
    (outdir / "correction_stack_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
