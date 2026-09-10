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

from build_energy_correction_stack import M_PI0_MEV, input_paths_from_summary, normalize_path
from timing_window_utils import passes_timing_window


ENERGY_BINS = [(0.6, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 2.5)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Validate seed-block residual granularity after freezing broad shared(E) "
            "and full-timeline time_scale(run). This tests whether block maps need "
            "global, x60_4a/b, or chunk-local granularity without letting chunks refit "
            "the broad layers."
        )
    )
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--input-summary")
    ap.add_argument("--coeff-tsv", default="output/x60_4_partial_68/sigma_window_2p0_working_exact2_t1p0_emin0p6/shared_curve/shared_curve_coeffs.tsv")
    ap.add_argument("--time-slice-tsv", default="output/x60_4_partial_68/correction_stack_working/time_slice_scales.tsv")
    ap.add_argument("--chunk-map-tsv", default="output/x60_4_partial_68/frozen_chunk6_seed_hybrid_export/tables/chunk_map.tsv")
    ap.add_argument("--output-dir", default="output/x60_4_partial_68/frozen_shared_time_block_granularity")
    ap.add_argument("--mass-window", nargs=2, type=float, default=[0.115961, 0.138212])
    ap.add_argument("--timing-center", type=float, default=-0.176158)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--min-block-entries", type=int, default=50)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--left-fallback-cols", nargs="+", type=int, default=[0, 1, 2])
    return ap.parse_args()


def read_tsv(path: Path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def load_coeff(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_tsv(path):
        out[row["key"]] = float(row["value"])
    return out


def shared_scale(coeff: dict[str, float], energy: float) -> float:
    ee = max(float(energy), 1e-6)
    return math.exp(coeff["c0"] + coeff["c1"] / ee + coeff["c2"] * math.log(ee))


def load_time_slices(path: Path) -> list[dict[str, float]]:
    out = []
    for row in read_tsv(path):
        out.append(
            {
                "slice": int(row["slice"]),
                "run_min": int(row["run_min"]),
                "run_max": int(row["run_max"]),
                "scale": float(row.get("time_slice_scale", row.get("mass_scale_to_pi0", 1.0))),
            }
        )
    return out


def time_scale_for_run(slices: list[dict[str, float]], run: int) -> float:
    for row in slices:
        if int(row["run_min"]) <= run <= int(row["run_max"]):
            return float(row["scale"])
    if not slices:
        return 1.0
    nearest = min(slices, key=lambda r: min(abs(run - int(r["run_min"])), abs(run - int(r["run_max"]))))
    return float(nearest["scale"])


def load_chunk6(path: Path) -> list[dict[str, object]]:
    chunks = []
    for row in read_tsv(path):
        chunks.append(
            {
                "scheme": "chunk6",
                "group": row["chunk_id"],
                "run_min": int(row["run_min"]),
                "run_max": int(row["run_max"]),
            }
        )
    return chunks


def build_groups(chunk_map: Path) -> dict[str, list[dict[str, object]]]:
    return {
        "global": [{"scheme": "global", "group": "x60_4_all", "run_min": 4253, "run_max": 4568}],
        "x60_4ab": [
            {"scheme": "x60_4ab", "group": "x60_4a2", "run_min": 4253, "run_max": 4415},
            {"scheme": "x60_4ab", "group": "x60_4b", "run_min": 4417, "run_max": 4568},
        ],
        "chunk6": load_chunk6(chunk_map),
    }


def group_for_run(groups: list[dict[str, object]], run: int) -> dict[str, object] | None:
    for group in groups:
        if int(group["run_min"]) <= run <= int(group["run_max"]):
            return group
    return None


def block_row_col(block: int) -> tuple[int, int]:
    return block // 30, block % 30


def lower_seed(row: dict[str, float]) -> int:
    return int(row["seed1_block"] if row["e1"] <= row["e2"] else row["seed2_block"])


def select_row(row: dict[str, float], args: argparse.Namespace) -> bool:
    if int(row["nclusters"]) != 2:
        return False
    if row["e1"] < 0.6 or row["e2"] < 0.6:
        return False
    if abs(row["x1"]) >= 29.025 or abs(row["x2"]) >= 29.025:
        return False
    if abs(row["y1"]) >= 35.475 or abs(row["y2"]) >= 35.475:
        return False
    if not passes_timing_window(row, args.timing_window, args.timing_center):
        return False
    if not (args.mass_window[0] <= row["pair_m"] <= args.mass_window[1]):
        return False
    if not (args.emin_range[0] <= row["min_e"] <= args.emin_range[1]):
        return False
    if int(row.get("seed1_block", -1)) < 0 or int(row.get("seed2_block", -1)) < 0:
        return False
    return True


def load_events(paths: list[Path], coeff: dict[str, float], slices: list[dict[str, float]], args: argparse.Namespace) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for path in paths:
        for raw in read_tsv(path):
            row = {key: float(value) for key, value in raw.items()}
            if not select_row(row, args):
                continue
            run = int(row["run"])
            seed1 = int(row["seed1_block"])
            seed2 = int(row["seed2_block"])
            shared_mass = row["pair_m"] * 1000.0 * math.sqrt(shared_scale(coeff, row["e1"]) * shared_scale(coeff, row["e2"]))
            shared_time_mass = shared_mass * time_scale_for_run(slices, run)
            events.append(
                {
                    "run": run,
                    "seed1": seed1,
                    "seed2": seed2,
                    "low_seed": lower_seed(row),
                    "shared_mass": shared_mass,
                    "shared_time_mass": shared_time_mass,
                    "shared_time_resid": shared_time_mass - M_PI0_MEV,
                    "min_e": float(row["min_e"]),
                }
            )
    return events


def stat(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"entries": 0, "mean_mev": float("nan"), "mae_mev": float("nan"), "rmse_mev": float("nan"), "sigma_mev": float("nan")}
    return {
        "entries": int(arr.size),
        "mean_mev": float(np.mean(arr)),
        "mae_mev": float(np.mean(np.abs(arr))),
        "rmse_mev": float(np.sqrt(np.mean(arr * arr))),
        "sigma_mev": float(np.std(arr)),
    }


def derive_block_scales(events: list[dict[str, float]], min_entries: int, damping: float) -> dict[int, dict[str, float]]:
    vals: dict[int, list[float]] = defaultdict(list)
    for event in events:
        vals[int(event["low_seed"])].append(float(event["shared_time_resid"]))
    scales: dict[int, dict[str, float]] = {}
    for block, residuals in vals.items():
        if len(residuals) < min_entries:
            continue
        mean_resid = float(np.mean(residuals))
        full = (M_PI0_MEV / max(1e-6, M_PI0_MEV + mean_resid)) ** 2
        applied = full ** damping
        rr, cc = block_row_col(block)
        scales[block] = {
            "entries": float(len(residuals)),
            "mean_residual_mev": mean_resid,
            "block_scale_full": full,
            "applied_block_scale": applied,
            "row": float(rr),
            "col": float(cc),
        }
    return scales


def apply_block(event: dict[str, float], scales: dict[int, dict[str, float]]) -> tuple[float, bool, bool]:
    s1 = scales.get(int(event["seed1"]))
    s2 = scales.get(int(event["seed2"]))
    scale1 = float(s1["applied_block_scale"]) if s1 else 1.0
    scale2 = float(s2["applied_block_scale"]) if s2 else 1.0
    mass = float(event["shared_time_mass"]) * math.sqrt(max(1e-12, scale1 * scale2))
    return mass - M_PI0_MEV, bool(s1), bool(s2)


def apply_block_with_fallback(
    event: dict[str, float],
    primary_scales: dict[int, dict[str, float]],
    fallback_scales: dict[int, dict[str, float]],
    fallback_cols: set[int],
) -> tuple[float, bool, bool, bool]:
    used_fallback = False
    scales = []
    hits = []
    for seed in [int(event["seed1"]), int(event["seed2"])]:
        _, col = block_row_col(seed)
        scale_row = None
        if col in fallback_cols:
            scale_row = fallback_scales.get(seed)
            used_fallback = used_fallback or scale_row is not None
        if scale_row is None:
            scale_row = primary_scales.get(seed)
        scales.append(float(scale_row["applied_block_scale"]) if scale_row else 1.0)
        hits.append(scale_row is not None)
    mass = float(event["shared_time_mass"]) * math.sqrt(max(1e-12, scales[0] * scales[1]))
    return mass - M_PI0_MEV, hits[0], hits[1], used_fallback


def evaluate(events: list[dict[str, float]], scales: dict[int, dict[str, float]]) -> tuple[dict[str, float], dict[str, int], dict[int, list[float]]]:
    residuals = []
    block_resids: dict[int, list[float]] = defaultdict(list)
    support = defaultdict(int)
    for event in events:
        resid, has1, has2 = apply_block(event, scales)
        residuals.append(resid)
        block_resids[int(event["low_seed"])].append(resid)
        if has1 and has2:
            support["both"] += 1
        elif has1 or has2:
            support["one"] += 1
        else:
            support["none"] += 1
    return stat(residuals), dict(support), block_resids


def energy_bin(min_e: float) -> tuple[float, float] | None:
    for lo, hi in ENERGY_BINS:
        if lo <= min_e < hi:
            return lo, hi
    return None


def block_regions(block: int) -> list[str]:
    _, col = block_row_col(block)
    out = ["all"]
    if col == 0:
        out.append("col0")
    if col == 1:
        out.append("col1")
    if 0 <= col <= 2:
        out.append("left_cols_0_2")
    if col >= 3:
        out.append("bulk_cols_ge3")
    return out


def split_events(events: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    train = [e for e in events if int(e["run"]) % 2 == 0]
    test = [e for e in events if int(e["run"]) % 2 == 1]
    return train, test


def write_outputs(
    outdir: Path,
    scheme_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    update_rows: list[dict[str, object]],
    block_rows: list[dict[str, object]],
    energy_rows: list[dict[str, object]],
    left_edge_rows: list[dict[str, object]],
) -> None:
    def write(name: str, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (outdir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write("scheme_pooled_metrics.tsv", scheme_rows)
    write("group_fold_metrics.tsv", group_rows)
    write("block_update_tables.tsv", update_rows)
    write("test_block_residuals.tsv", block_rows)
    write("energy_bin_metrics.tsv", energy_rows)
    write("left_edge_region_metrics.tsv", left_edge_rows)


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def residual_map(block_rows: list[dict[str, object]], scheme: str, group: str, min_entries: int) -> np.ndarray:
    grid = np.full((36, 30), np.nan)
    for row in block_rows:
        if row["scheme"] != scheme or row["group"] != group or int(row["entries"]) < min_entries:
            continue
        grid[int(row["row"]), int(row["col"])] = float(row["mean_mev"])
    return grid


def plot_maps(outdir: Path, block_rows: list[dict[str, object]], groups_by_scheme: dict[str, list[dict[str, object]]], min_entries: int) -> None:
    pdf_path = outdir / "frozen_shared_time_block_granularity_maps.pdf"
    with PdfPages(pdf_path) as pdf:
        for scheme, groups in groups_by_scheme.items():
            n = len(groups)
            cols = min(3, n)
            rows = int(math.ceil(n / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(11, 3.4 * rows), squeeze=False, constrained_layout=True)
            vals = []
            grids = []
            for group in groups:
                grid = residual_map(block_rows, scheme, str(group["group"]), min_entries)
                grids.append(grid)
                vals.extend(grid[np.isfinite(grid)].tolist())
            vmax = max(0.5, float(np.nanpercentile(np.abs(vals), 95)) if vals else 1.0)
            for ax, group, grid in zip(axs.flat, groups, grids):
                im = ax.imshow(grid, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_title(f"{group['group']} ({group['run_min']}-{group['run_max']})", fontsize=9)
                ax.set_xlabel("column")
                ax.set_ylabel("row")
                ax.plot([0], [17], marker=">", color="black", ms=5)
            for ax in axs.flat[n:]:
                ax.axis("off")
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.86, label="held-out mean residual (MeV)")
            fig.suptitle(f"{scheme}: held-out residual maps after fixed shared+time + group block map", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)


def evaluate_left_edge_fallback(
    events: list[dict[str, float]],
    groups_by_scheme: dict[str, list[dict[str, object]]],
    *,
    min_entries: int,
    damping: float,
    fallback_cols: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    global_group = groups_by_scheme["global"][0]
    global_events = [e for e in events if int(global_group["run_min"]) <= int(e["run"]) <= int(global_group["run_max"])]
    global_train, _ = split_events(global_events)
    global_scales = derive_block_scales(global_train, min_entries, damping)

    policy_resids: dict[tuple[str, str], list[float]] = defaultdict(list)
    policy_energy: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    support = defaultdict(int)

    for group in groups_by_scheme["x60_4ab"]:
        group_events = [e for e in events if int(group["run_min"]) <= int(e["run"]) <= int(group["run_max"])]
        train, test = split_events(group_events)
        period_scales = derive_block_scales(train, min_entries, damping)
        for event in test:
            resid, has1, has2, used_fallback = apply_block_with_fallback(event, period_scales, global_scales, fallback_cols)
            policy_resids[("x60_4ab_bulk_global_left", "shared_time_group_block")].append(resid)
            for region in block_regions(int(event["low_seed"])):
                policy_resids[("x60_4ab_bulk_global_left", region)].append(resid)
            eb = energy_bin(float(event["min_e"]))
            if eb is not None:
                policy_energy[("x60_4ab_bulk_global_left", eb[0], eb[1])].append(resid)
            if has1 and has2:
                support["both"] += 1
            elif has1 or has2:
                support["one"] += 1
            else:
                support["none"] += 1
            if used_fallback:
                support["used_fallback"] += 1

    rows = [
        {
            "scheme": "x60_4ab_bulk_global_left",
            "model": "shared_time_group_block",
            **stat(policy_resids[("x60_4ab_bulk_global_left", "shared_time_group_block")]),
            "events_with_both_seed_scales": support.get("both", 0),
            "events_with_one_seed_scale": support.get("one", 0),
            "events_without_seed_scale": support.get("none", 0),
            "events_using_left_fallback": support.get("used_fallback", 0),
        }
    ]
    for region in ["all", "col0", "col1", "left_cols_0_2", "bulk_cols_ge3"]:
        rows.append(
            {
                "scheme": "x60_4ab_bulk_global_left",
                "model": "shared_time_group_block",
                "region": region,
                **stat(policy_resids[("x60_4ab_bulk_global_left", region)]),
            }
        )

    energy_rows = []
    for (scheme, lo, hi), vals in sorted(policy_energy.items()):
        energy_rows.append({"scheme": scheme, "model": "shared_time_group_block", "emin_lo": lo, "emin_hi": hi, **stat(vals)})
    return rows, energy_rows


def evaluate_fallback_scan(
    events: list[dict[str, float]],
    groups_by_scheme: dict[str, list[dict[str, object]]],
    *,
    min_entries: int,
    damping: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scan_defs = {
        "fallback_none": set(),
        "fallback_col1": {1},
        "fallback_cols0_1": {0, 1},
        "fallback_cols0_2": {0, 1, 2},
    }
    metric_rows = []
    region_rows = []
    for label, cols in scan_defs.items():
        rows, _ = evaluate_left_edge_fallback(
            events,
            groups_by_scheme,
            min_entries=min_entries,
            damping=damping,
            fallback_cols=cols,
        )
        main = rows[0].copy()
        main["fallback_policy"] = label
        main["fallback_cols"] = ",".join(str(c) for c in sorted(cols)) or "none"
        metric_rows.append(main)
        for row in rows[1:]:
            out = row.copy()
            out["fallback_policy"] = label
            out["fallback_cols"] = ",".join(str(c) for c in sorted(cols)) or "none"
            region_rows.append(out)
    return metric_rows, region_rows


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    coeff = load_coeff(Path(args.coeff_tsv))
    slices = load_time_slices(Path(args.time_slice_tsv))
    paths: list[Path] = []
    if args.input_summary:
        paths.extend(input_paths_from_summary(Path(args.input_summary)))
    if args.input_dir:
        paths.extend(sorted(Path(args.input_dir).glob("nps_production_*_wf.tsv")))
    paths = [normalize_path(str(p)) for p in dict.fromkeys(paths)]
    events = load_events(paths, coeff, slices, args)
    groups_by_scheme = build_groups(Path(args.chunk_map_tsv))

    scheme_resids: dict[tuple[str, str], list[float]] = defaultdict(list)
    energy_resids: dict[tuple[str, str, str, float, float], list[float]] = defaultdict(list)
    left_edge_resids: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    group_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []

    for scheme, groups in groups_by_scheme.items():
        for group in groups:
            group_events = [e for e in events if int(group["run_min"]) <= int(e["run"]) <= int(group["run_max"])]
            train, test = split_events(group_events)
            scales = derive_block_scales(train, args.min_block_entries, args.damping)
            test_metrics, support, block_resids = evaluate(test, scales)
            shared_time_metrics = stat([float(e["shared_time_resid"]) for e in test])

            for model, row in [("shared_time", shared_time_metrics), ("shared_time_group_block", test_metrics)]:
                group_rows.append(
                    {
                        "scheme": scheme,
                        "group": group["group"],
                        "run_min": group["run_min"],
                        "run_max": group["run_max"],
                        "model": model,
                        **row,
                    }
                )
            for event in test:
                resid, _, _ = apply_block(event, scales)
                shared_time_resid = float(event["shared_time_resid"])
                scheme_resids[(scheme, "shared_time_group_block")].append(resid)
                scheme_resids[(scheme, "shared_time")].append(shared_time_resid)
                eb = energy_bin(float(event["min_e"]))
                if eb is not None:
                    energy_resids[(scheme, str(group["group"]), "shared_time", eb[0], eb[1])].append(shared_time_resid)
                    energy_resids[(scheme, str(group["group"]), "shared_time_group_block", eb[0], eb[1])].append(resid)
                for region in block_regions(int(event["low_seed"])):
                    left_edge_resids[(scheme, str(group["group"]), "shared_time", region)].append(shared_time_resid)
                    left_edge_resids[(scheme, str(group["group"]), "shared_time_group_block", region)].append(resid)

            for block, scale in sorted(scales.items()):
                update_rows.append(
                    {
                        "scheme": scheme,
                        "group": group["group"],
                        "seed_block": block,
                        "row": int(scale["row"]),
                        "col": int(scale["col"]),
                        "train_entries": int(scale["entries"]),
                        "train_mean_residual_mev": scale["mean_residual_mev"],
                        "block_scale_full": scale["block_scale_full"],
                        "applied_block_scale": scale["applied_block_scale"],
                    }
                )
            for block, vals in sorted(block_resids.items()):
                rr, cc = block_row_col(block)
                row = stat(vals)
                block_rows.append(
                    {
                        "scheme": scheme,
                        "group": group["group"],
                        "seed_block": block,
                        "row": rr,
                        "col": cc,
                        **row,
                    }
                )

            group_rows[-1]["events_with_both_seed_scales"] = support.get("both", 0)
            group_rows[-1]["events_with_one_seed_scale"] = support.get("one", 0)
            group_rows[-1]["events_without_seed_scale"] = support.get("none", 0)

    scheme_rows: list[dict[str, object]] = []
    for scheme in groups_by_scheme:
        for model in ["shared_time", "shared_time_group_block"]:
            row = stat(scheme_resids[(scheme, model)])
            scheme_rows.append({"scheme": scheme, "model": model, **row})

    energy_rows: list[dict[str, object]] = []
    for (scheme, group, model, lo, hi), vals in sorted(energy_resids.items()):
        energy_rows.append({"scheme": scheme, "group": group, "model": model, "emin_lo": lo, "emin_hi": hi, **stat(vals)})

    left_edge_rows: list[dict[str, object]] = []
    for (scheme, group, model, region), vals in sorted(left_edge_resids.items()):
        left_edge_rows.append({"scheme": scheme, "group": group, "model": model, "region": region, **stat(vals)})

    fallback_rows, fallback_energy_rows = evaluate_left_edge_fallback(
        events,
        groups_by_scheme,
        min_entries=args.min_block_entries,
        damping=args.damping,
        fallback_cols=set(args.left_fallback_cols),
    )
    fallback_scan_rows, fallback_scan_region_rows = evaluate_fallback_scan(
        events,
        groups_by_scheme,
        min_entries=args.min_block_entries,
        damping=args.damping,
    )
    scheme_rows.append(fallback_rows[0])
    for row in fallback_rows[1:]:
        left_edge_rows.append({"group": "x60_4ab_with_global_left_cols_0_2", **row})
    for row in fallback_energy_rows:
        energy_rows.append({"group": "x60_4ab_with_global_left_cols_0_2", **row})

    write_outputs(outdir, scheme_rows, group_rows, update_rows, block_rows, energy_rows, left_edge_rows)
    write_table(outdir / "heldout_fallback_column_scan_metrics.tsv", fallback_scan_rows)
    write_table(outdir / "heldout_fallback_column_scan_region_metrics.tsv", fallback_scan_region_rows)
    plot_maps(outdir, block_rows, groups_by_scheme, args.min_block_entries)

    best = min(
        (row for row in scheme_rows if row["model"] == "shared_time_group_block"),
        key=lambda r: float(r["rmse_mev"]),
    )
    summary = [
        "Frozen shared/time block-granularity validation",
        f"input_tsvs: {len(paths)}",
        f"selected_events: {len(events)}",
        f"mass_window: {args.mass_window[0]:.6f} {args.mass_window[1]:.6f}",
        f"timing_center: {args.timing_center:.6f}",
        f"timing_window: {args.timing_window:.3f}",
        f"emin_range: {args.emin_range[0]:.3f} {args.emin_range[1]:.3f}",
        f"min_block_entries: {args.min_block_entries}",
        f"damping: {args.damping:.3f}",
        "",
        "Method:",
        "  broad shared(E) curve and full-timeline time scales are fixed for all schemes",
        "  only seed-block residual maps change granularity",
        "  even runs train block maps; odd runs are held out for evaluation inside each group",
        "",
        f"best_scheme_by_pooled_rmse: {best['scheme']}",
        f"best_rmse_mev: {float(best['rmse_mev']):.6f}",
    ]
    (outdir / "frozen_shared_time_block_granularity_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
