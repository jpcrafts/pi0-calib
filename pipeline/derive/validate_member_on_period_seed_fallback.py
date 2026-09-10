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

from build_energy_correction_stack import M_PI0_MEV, metrics
from validate_energy_correction_stack_holdout import time_scale_for_run
from validate_frozen_shared_time_block_granularity import (
    ENERGY_BINS,
    block_row_col,
    load_coeff,
    load_time_slices,
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
            "Held-out test of member residuals on top of the current seed-only policy: "
            "fixed broad shared/time, period seed maps in bulk, global seed fallback on left edge."
        )
    )
    ap.add_argument("--chunk-summary", action="append")
    ap.add_argument("--output-dir", default="output/x60_4_partial_68/member_on_period_seed_fallback_run_parity")
    ap.add_argument("--coeff-tsv", default="output/x60_4_partial_68/sigma_window_2p0_working_exact2_t1p0_emin0p6/shared_curve/shared_curve_coeffs.tsv")
    ap.add_argument("--time-slice-tsv", default="output/x60_4_partial_68/correction_stack_working/time_slice_scales.tsv")
    ap.add_argument("--mass-window", nargs=2, type=float, default=[0.115961, 0.138212])
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
    return ap.parse_args()


def default_chunk_summaries() -> list[Path]:
    return sorted(Path("manifests/run_range_chunks").glob("x60_4_chunk6_*_member_sidecars.summary.txt"))


def period_for_run(run: int) -> str | None:
    if 4253 <= run <= 4415:
        return "x60_4a2"
    if 4417 <= run <= 4568:
        return "x60_4b"
    return None


def event_time_scale(event: dict[str, object], slices: list[dict[str, object]]) -> float:
    run = int(event["run"])
    for sl in slices:
        if int(sl["run_min"]) <= run <= int(sl["run_max"]):
            return float(sl.get("time_slice_scale", sl.get("scale", 1.0)))
    nearest = min(slices, key=lambda sl: min(abs(run - int(sl["run_min"])), abs(run - int(sl["run_max"]))))
    return float(nearest.get("time_slice_scale", nearest.get("scale", 1.0)))


def shared_time_mass(event: dict[str, object], slices: list[dict[str, object]]) -> float:
    return float(event["shared_mass_mev"]) * event_time_scale(event, slices)


def derive_seed_scales(events: list[dict[str, object]], slices: list[dict[str, object]], min_entries: int) -> dict[int, dict[str, float]]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for event in events:
        residual = shared_time_mass(event, slices) - M_PI0_MEV
        by_block[int(event["low_seed_block"])].append(residual)
    out: dict[int, dict[str, float]] = {}
    for block, vals in by_block.items():
        if len(vals) < min_entries:
            continue
        mean = float(np.mean(vals))
        rr, cc = block_row_col(block)
        out[block] = {
            "row": float(rr),
            "col": float(cc),
            "entries": float(len(vals)),
            "mean_residual_mev": mean,
            "block_scale_full": (M_PI0_MEV / max(1e-6, M_PI0_MEV + mean)) ** 2,
        }
    return out


def choose_seed_scale(
    block: int,
    period: str,
    global_scales: dict[int, dict[str, float]],
    period_scales: dict[str, dict[int, dict[str, float]]],
    left_cols: set[int],
) -> tuple[float, str]:
    _, col = block_row_col(block)
    if col in left_cols and block in global_scales:
        return float(global_scales[block]["block_scale_full"]), "global_left_fallback"
    if block in period_scales.get(period, {}):
        return float(period_scales[period][block]["block_scale_full"]), "period_bulk"
    if block in global_scales:
        return float(global_scales[block]["block_scale_full"]), "global_missing_period_fallback"
    return 1.0, "identity_missing"


def cluster_member_scale(weights: dict[int, float], member_scales: dict[int, float]) -> float:
    if not weights:
        return 1.0
    return sum(float(weight) * float(member_scales.get(int(block), 1.0)) for block, weight in weights.items())


def apply_stack(
    event: dict[str, object],
    slices: list[dict[str, object]],
    global_seed: dict[int, dict[str, float]],
    period_seed: dict[str, dict[int, dict[str, float]]],
    member_scales: dict[int, float],
    left_cols: set[int],
) -> tuple[float, list[str]]:
    period = period_for_run(int(event["run"]))
    if period is None:
        period = "unknown"
    s1, src1 = choose_seed_scale(int(event["seed1_block"]), period, global_seed, period_seed, left_cols)
    s2, src2 = choose_seed_scale(int(event["seed2_block"]), period, global_seed, period_seed, left_cols)
    m1 = cluster_member_scale(event["weights1"], member_scales)  # type: ignore[arg-type]
    m2 = cluster_member_scale(event["weights2"], member_scales)  # type: ignore[arg-type]
    mass = shared_time_mass(event, slices) * math.sqrt(max(1e-12, s1 * s2 * m1 * m2))
    return mass, [src1, src2]


def derive_member_updates(
    events: list[dict[str, object]],
    slices: list[dict[str, object]],
    global_seed: dict[int, dict[str, float]],
    period_seed: dict[str, dict[int, dict[str, float]]],
    member_scales: dict[int, float],
    left_cols: set[int],
    min_entries: int,
) -> dict[int, dict[str, float]]:
    by_block: dict[int, dict[str, float]] = defaultdict(lambda: {"entries": 0.0, "weight_sum": 0.0, "weighted_residual_sum": 0.0})
    for event in events:
        mass, _ = apply_stack(event, slices, global_seed, period_seed, member_scales, left_cols)
        residual = mass - M_PI0_MEV
        low_weights = event["low_weights"]  # type: ignore[assignment]
        for block, weight in low_weights.items():  # type: ignore[union-attr]
            row = by_block[int(block)]
            row["entries"] += 1.0
            row["weight_sum"] += float(weight)
            row["weighted_residual_sum"] += float(weight) * residual
    out: dict[int, dict[str, float]] = {}
    for block, row in by_block.items():
        if int(row["entries"]) < min_entries or not (row["weight_sum"] > 0.0):
            continue
        mean_residual = row["weighted_residual_sum"] / row["weight_sum"]
        update_full = (M_PI0_MEV / max(1e-6, M_PI0_MEV + mean_residual)) ** 2
        out[block] = {
            "entries": int(row["entries"]),
            "weight_sum": row["weight_sum"],
            "mean_residual_mev": mean_residual,
            "update_scale_full": update_full,
        }
    return out


def clip_update(update: float, max_update_frac: float) -> float:
    if max_update_frac <= 0.0:
        return update
    return min(max(update, 1.0 - max_update_frac), 1.0 + max_update_frac)


def split_folds(events: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]], list[dict[str, object]]]]:
    even = [event for event in events if int(event["run"]) % 2 == 0]
    odd = [event for event in events if int(event["run"]) % 2 == 1]
    return [("even_train_odd_test", even, odd), ("odd_train_even_test", odd, even)]


def stat_rows_by_region(events: list[dict[str, object]], residuals: list[float], model: str, fold: str) -> list[dict[str, object]]:
    by_region: dict[str, list[float]] = defaultdict(list)
    for event, residual in zip(events, residuals):
        _, col = block_row_col(int(event["low_seed_block"]))
        by_region["all"].append(residual)
        if col <= 2:
            by_region["left_cols_0_2"].append(residual)
        else:
            by_region["bulk_cols_ge3"].append(residual)
        if col == 1:
            by_region["col1"].append(residual)
        if col == 2:
            by_region["col2"].append(residual)
    return [{"fold": fold, "model": model, "region": region, **metrics(vals)} for region, vals in sorted(by_region.items())]


def energy_bin(min_e: float) -> tuple[float, float] | None:
    for lo, hi in ENERGY_BINS:
        if lo <= min_e < hi:
            return lo, hi
    return None


def stat_rows_by_energy(events: list[dict[str, object]], residuals: list[float], model: str, fold: str) -> list[dict[str, object]]:
    by_bin: dict[tuple[float, float], list[float]] = defaultdict(list)
    for event, residual in zip(events, residuals):
        eb = energy_bin(float(event["min_e"]))
        if eb is not None:
            by_bin[eb].append(residual)
    return [{"fold": fold, "model": model, "emin_lo": lo, "emin_hi": hi, **metrics(vals)} for (lo, hi), vals in sorted(by_bin.items())]


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def block_residual_rows(events: list[dict[str, object]], residuals: list[float], model: str) -> list[dict[str, object]]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for event, residual in zip(events, residuals):
        by_block[int(event["low_seed_block"])].append(residual)
    rows = []
    for block, vals in sorted(by_block.items()):
        rr, cc = block_row_col(block)
        rows.append({"model": model, "seed_block": block, "row": rr, "col": cc, **metrics(vals)})
    return rows


def plot_block_maps(outdir: Path, rows: list[dict[str, object]], min_entries: int) -> None:
    models = ["seed_baseline", "member_iter_1", "member_iter_2", "member_iter_3"]
    grids = {}
    vals = []
    for model in models:
        grid = np.full((36, 30), np.nan)
        for row in rows:
            if row["model"] != model or int(row["entries"]) < min_entries:
                continue
            grid[int(row["row"]), int(row["col"])] = float(row["mean_mev"])
        grids[model] = grid
        vals.extend(grid[np.isfinite(grid)].tolist())
    if not vals:
        return
    vmax = max(0.5, float(np.nanpercentile(np.abs(vals), 95)))
    with PdfPages(outdir / "seed_vs_member_period_fallback_residual_maps.pdf") as pdf:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, model in zip(axs.flat, models):
            im = ax.imshow(grids[model], origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(model)
            ax.set_xlabel("column")
            ax.set_ylabel("row")
            ax.axvspan(-0.5, 2.5, color="black", alpha=0.08)
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.85, label="held-out mean residual (MeV)")
        fig.suptitle("Held-out seed baseline vs member residual on period/fallback seed policy")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_summaries = [Path(p) for p in args.chunk_summary] if args.chunk_summary else default_chunk_summaries()
    input_paths: list[Path] = []
    for summary in chunk_summaries:
        input_paths.extend(input_paths_from_summary(summary))
    input_paths = list(dict.fromkeys(input_paths))
    if not input_paths:
        raise SystemExit("no input paths from chunk summaries")

    coeff = load_coeff(Path(args.coeff_tsv))
    coeffs = np.asarray([coeff["c0"], coeff["c1"], coeff["c2"]], dtype=float)
    slices = load_time_slices(Path(args.time_slice_tsv))
    rows = read_rows(input_paths)
    selected_rows = select_raw_rows(
        rows,
        mass_window=(args.mass_window[0], args.mass_window[1]),
        timing_window=args.timing_window,
        timing_center=args.timing_center,
        min_e_floor=args.min_e_floor,
        nclusters_mode="exact2",
        emin_range=(args.emin_range[0], args.emin_range[1]),
    )
    member_index, missing = load_member_index(input_paths)
    if missing:
        raise SystemExit(f"missing member files: {missing[:5]}")
    events, closure = rows_to_member_events(selected_rows, coeffs, member_index, rtol=args.rtol, atol=args.atol)
    if int(closure["usable_rows"]) != int(closure["selected_rows"]):
        raise SystemExit(f"member closure incomplete: {closure}")

    left_cols = set(args.left_fallback_cols)
    pooled: dict[str, list[float]] = defaultdict(list)
    pooled_events: list[dict[str, object]] = []
    fold_rows = []
    update_rows = []
    energy_rows = []
    region_rows = []
    block_rows = []

    for fold, train, test in split_folds(events):
        global_seed = derive_seed_scales(train, slices, args.min_seed_entries)
        period_seed = {
            "x60_4a2": derive_seed_scales([e for e in train if period_for_run(int(e["run"])) == "x60_4a2"], slices, args.min_seed_entries),
            "x60_4b": derive_seed_scales([e for e in train if period_for_run(int(e["run"])) == "x60_4b"], slices, args.min_seed_entries),
        }
        member_scales: dict[int, float] = {}

        seed_resids = []
        for event in test:
            mass, _ = apply_stack(event, slices, global_seed, period_seed, {}, left_cols)
            seed_resids.append(mass - M_PI0_MEV)
        pooled["seed_baseline"].extend(seed_resids)
        pooled_events.extend(test)
        fold_rows.append({"fold": fold, "iteration": 0, "model": "seed_baseline", **metrics(seed_resids)})
        energy_rows.extend(stat_rows_by_energy(test, seed_resids, "seed_baseline", fold))
        region_rows.extend(stat_rows_by_region(test, seed_resids, "seed_baseline", fold))
        block_rows.extend(block_residual_rows(test, seed_resids, "seed_baseline"))

        for iteration in range(1, args.iterations + 1):
            updates = derive_member_updates(
                train,
                slices,
                global_seed,
                period_seed,
                member_scales,
                left_cols,
                args.min_member_entries,
            )
            for block, row in sorted(updates.items()):
                applied = clip_update(float(row["update_scale_full"]) ** args.damping, args.max_update_frac)
                prev = float(member_scales.get(block, 1.0))
                member_scales[block] = prev * applied
                update_rows.append(
                    {
                        "fold": fold,
                        "iteration": iteration,
                        "block": block,
                        "entries": row["entries"],
                        "weight_sum": row["weight_sum"],
                        "mean_residual_mev": row["mean_residual_mev"],
                        "update_scale_full": row["update_scale_full"],
                        "applied_update_scale": applied,
                        "previous_scale": prev,
                        "new_scale": member_scales[block],
                    }
                )
            resids = []
            for event in test:
                mass, _ = apply_stack(event, slices, global_seed, period_seed, member_scales, left_cols)
                resids.append(mass - M_PI0_MEV)
            model = f"member_iter_{iteration}"
            pooled[model].extend(resids)
            fold_rows.append({"fold": fold, "iteration": iteration, "model": model, **metrics(resids)})
            energy_rows.extend(stat_rows_by_energy(test, resids, model, fold))
            region_rows.extend(stat_rows_by_region(test, resids, model, fold))
            block_rows.extend(block_residual_rows(test, resids, model))

    pooled_rows = [{"model": model, **metrics(vals)} for model, vals in sorted(pooled.items())]
    write_table(outdir / "pooled_test_metrics.tsv", pooled_rows)
    write_table(outdir / "fold_metrics.tsv", fold_rows)
    write_table(outdir / "fold_update_table.tsv", update_rows)
    write_table(outdir / "energy_bin_metrics.tsv", energy_rows)
    write_table(outdir / "region_metrics.tsv", region_rows)
    write_table(outdir / "block_residuals.tsv", block_rows)
    plot_block_maps(outdir, block_rows, args.min_seed_entries)

    best = min(pooled_rows, key=lambda r: float(r["rmse_mev"]))
    summary = [
        "Member residual on period/fallback seed policy",
        f"input_tsvs: {len(input_paths)}",
        f"selected_rows: {len(selected_rows)}",
        f"usable_member_events: {len(events)}",
        f"left_fallback_cols: {','.join(str(c) for c in sorted(left_cols))}",
        f"iterations: {args.iterations}",
        f"damping: {args.damping}",
        f"max_update_frac: {args.max_update_frac}",
        f"best_model: {best['model']}",
        f"best_rmse_mev: {float(best['rmse_mev']):.6f}",
        "",
        "Method:",
        "  fixed broad shared curve and full-timeline time scale",
        "  seed policy derived on train fold: x60_4a2/x60_4b bulk with global left fallback",
        "  member residual updates derived on train fold only",
        "  evaluation on held-out run parity fold only",
    ]
    (outdir / "member_on_period_seed_fallback_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
