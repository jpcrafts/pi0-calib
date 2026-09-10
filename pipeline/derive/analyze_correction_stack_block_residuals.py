#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from build_energy_correction_stack import normalize_path, shared_scale
from timing_window_utils import parse_timing_center, passes_timing_window, resolve_timing_center


M_PI0_GEV = 0.1349766
DEFAULT_EDGES = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate post-correction-stack residuals by lower-energy seed block and energy bin."
        )
    )
    ap.add_argument("coeff_tsv")
    ap.add_argument("time_slice_tsv")
    ap.add_argument("block_scale_tsv")
    ap.add_argument("input_tsv", nargs="*")
    ap.add_argument("--input-summary")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mass-window", nargs=2, type=float, required=True)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--timing-center", default="auto")
    ap.add_argument("--min-e-floor", type=float, default=0.6)
    ap.add_argument("--nclusters-mode", choices=["exact2", "ge2"], default="exact2")
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--min-block-entries", type=int, default=4)
    ap.add_argument("--min-bin-entries", type=int, default=2)
    ap.add_argument("--damping", type=float, default=1.0)
    return ap.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, float]]:
    rows = []
    for path in paths:
        with path.open() as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_key_value_coeffs(path: Path) -> dict[str, float]:
    coeff = {}
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row.get("value", "") != "":
                coeff[row["key"]] = float(row["value"])
    return coeff


def load_time_scales(path: Path) -> list[dict[str, float]]:
    scales = []
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            scales.append(
                {
                    "run_min": int(row["run_min"]),
                    "run_max": int(row["run_max"]),
                    "scale": float(row["mass_scale_to_pi0"]),
                }
            )
    return scales


def load_block_scales(path: Path) -> dict[int, float]:
    out = {}
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            out[int(row["seed_block"])] = float(row["block_scale_full"])
    return out


def cluster_mode_ok(nclusters: int, mode: str) -> bool:
    if mode == "exact2":
        return nclusters == 2
    if mode == "ge2":
        return nclusters >= 2
    raise ValueError(f"unsupported nclusters mode: {mode}")


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def assign_bin(emin: float, edges: list[float]) -> tuple[float, float] | None:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= emin < hi:
            return lo, hi
    if edges and math.isclose(emin, edges[-1]):
        return edges[-2], edges[-1]
    return None


def lower_energy_seed(row: dict[str, float]) -> int:
    if row["e1"] <= row["e2"]:
        return int(row.get("seed1_block", -1.0))
    return int(row.get("seed2_block", -1.0))


def time_scale_for_run(scales: list[dict[str, float]], run: int) -> float:
    for row in scales:
        if row["run_min"] <= run <= row["run_max"]:
            return float(row["scale"])
    return 1.0


def damp_scale(full_scale: float, damping: float) -> float:
    return math.exp(damping * math.log(max(full_scale, 1e-12)))


def main() -> int:
    args = parse_args()
    coeff = load_key_value_coeffs(Path(args.coeff_tsv))
    time_scales = load_time_scales(Path(args.time_slice_tsv))
    block_scales_full = load_block_scales(Path(args.block_scale_tsv))
    applied_block_scales = {block: damp_scale(scale, args.damping) for block, scale in block_scales_full.items()}

    input_paths = [normalize_path(p) for p in args.input_tsv]
    if args.input_summary:
        for line in Path(args.input_summary).read_text(errors="replace").splitlines():
            if line.startswith("input_tsv:"):
                input_paths.append(normalize_path(line.split(":", 1)[1].strip()))
    input_paths = list(dict.fromkeys(input_paths))
    if not input_paths:
        raise SystemExit("no input TSVs provided")

    rows = load_rows(input_paths)
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mass_lo, mass_hi = args.mass_window
    emin_lo, emin_hi = args.emin_range
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
        parse_timing_center(args.timing_center),
    )

    edges = [e for e in DEFAULT_EDGES if emin_lo <= e <= emin_hi]
    if not edges:
        edges = [emin_lo, emin_hi]
    if edges[0] > emin_lo:
        edges = [emin_lo] + edges
    if edges[-1] < emin_hi:
        edges = edges + [emin_hi]

    block_all: dict[int, list[float]] = defaultdict(list)
    block_bins: dict[tuple[int, float, float], list[float]] = defaultdict(list)
    selected_rows = 0

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
        emin = r["min_e"]
        if not (emin_lo <= emin <= emin_hi):
            continue

        shared_mass = r["pair_m"] * math.sqrt(shared_scale(coeff, r["e1"]) * shared_scale(coeff, r["e2"]))
        tscale = time_scale_for_run(time_scales, int(r["run"]))
        block1 = int(r.get("seed1_block", -1.0))
        block2 = int(r.get("seed2_block", -1.0))
        if block1 < 0 or block2 < 0:
            continue
        b1 = applied_block_scales.get(block1, 1.0)
        b2 = applied_block_scales.get(block2, 1.0)
        corrected_mass = shared_mass * tscale * math.sqrt(b1 * b2)
        residual = (corrected_mass - M_PI0_GEV) * 1000.0

        seed_block = lower_energy_seed(r)
        if seed_block < 0:
            continue

        selected_rows += 1
        block_all[seed_block].append(residual)
        eb = assign_bin(emin, edges)
        if eb is not None:
            block_bins[(seed_block, eb[0], eb[1])].append(residual)

    with (outdir / "block_summary.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed_block", "entries", "mean_residual_mev"])
        for block in sorted(block_all):
            vals = block_all[block]
            if len(vals) < args.min_block_entries:
                continue
            w.writerow([block, len(vals), mean(vals)])

    with (outdir / "block_bin_residuals.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["seed_block", "elo", "ehi", "entries", "mean_residual_mev"])
        for (block, lo, hi) in sorted(block_bins):
            vals = block_bins[(block, lo, hi)]
            if len(vals) < args.min_bin_entries:
                continue
            w.writerow([block, lo, hi, len(vals), mean(vals)])

    blocks_with_stats = [block for block, vals in block_all.items() if len(vals) >= args.min_block_entries]
    lines = [
        "Post-correction-stack seed-block residual summary",
        f"selected_rows: {selected_rows}",
        f"mass_window: {mass_lo} {mass_hi}",
        f"nclusters_mode: {args.nclusters_mode}",
        f"min_e_floor: {args.min_e_floor}",
        f"timing_window: {args.timing_window}",
        f"timing_center_arg: {args.timing_center}",
        f"timing_center_resolved: {timing_center:.6f}",
        f"emin_range: {emin_lo} {emin_hi}",
        f"damping: {args.damping}",
        f"blocks_with_stats: {len(blocks_with_stats)}",
    ]
    if blocks_with_stats:
        block_means = [mean(block_all[b]) for b in blocks_with_stats]
        lines.append(f"block_mean_spread_mev: {max(block_means) - min(block_means):.6f}")
    lines.append(
        f"block_energy_bins_with_stats: {sum(1 for vals in block_bins.values() if len(vals) >= args.min_bin_entries)}"
    )
    (outdir / "block_residual_summary.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
