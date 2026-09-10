#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_correction_stack_block_residuals import DEFAULT_EDGES
from build_energy_correction_stack import M_PI0_MEV, cluster_mode_ok, input_paths_from_summary, metrics, normalize_path, shared_scale
from fit_calib_shared_curve import fit_shared_curve
from timing_window_utils import parse_timing_center, passes_timing_window, resolve_timing_center
from validate_energy_correction_stack_holdout import build_slices, derive_block_scales, fold_events, parse_segment, time_scale_for_run


PAIR_KEY_FIELDS = ("t_entry", "wf_entry", "run", "event", "pair_i", "pair_j")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Held-out validation of a member-weighted analog of the current seed-block "
            "correction stack. The shared curve and time-slice layers are unchanged; "
            "the block term is distributed through actual cluster member deposits."
        )
    )
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
    ap.add_argument(
        "--split-mode",
        choices=["segment_parity", "run_parity", "x60_4a2_4b", "x60_4a2_4b_stratified_run_parity"],
        default="run_parity",
        help=(
            "Held-out split. x60_4a2_4b trains/tests across the two x60_4 sub-kinematic "
            "run ranges found in the raw-WF manifest: 4253-4372 vs 4397-4568. "
            "x60_4a2_4b_stratified_run_parity keeps both sub-kinematics in each parity fold."
        ),
    )
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument(
        "--member-base",
        choices=["shared_time", "seed_block"],
        default="shared_time",
        help=(
            "Base stack for member residual updates. shared_time reproduces the existing member-weighted analog; "
            "seed_block tests a hybrid seed-block plus member residual layer."
        ),
    )
    ap.add_argument(
        "--max-update-frac",
        type=float,
        default=0.0,
        help=(
            "Optional fractional clip on each damped member update scale. "
            "0 preserves current behavior; 0.02 clamps applied updates to [0.98, 1.02]."
        ),
    )
    ap.add_argument(
        "--shuffle-member-blocks",
        action="store_true",
        help="Deterministically shuffle member block identities as a negative-control test.",
    )
    ap.add_argument("--shuffle-seed", type=int, default=314159)
    ap.add_argument("--rtol", type=float, default=1.0e-5)
    ap.add_argument("--atol", type=float, default=1.0e-5)
    ap.add_argument("--min-bin-entries", type=int, default=2)
    return ap.parse_args()


def close(a: float, b: float, *, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


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


def member_path_for_compact(path: Path) -> Path:
    if path.name.endswith("_wf.tsv"):
        return path.with_name(path.name[:-4] + ".members.tsv")
    return path.with_name(path.name + ".members.tsv")


def pair_key_from_compact(row: dict[str, float]) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(row["_segment"]),
        int(row["t_entry"]),
        int(row["wf_entry"]),
        int(row["run"]),
        int(row["event"]),
        int(row["pair_i"]),
        int(row["pair_j"]),
    )


def pair_key_from_member(row: dict[str, str], segment: int) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(segment),
        int(row["t_entry"]),
        int(row["wf_entry"]),
        int(row["run"]),
        int(row["event"]),
        int(row["pair_i"]),
        int(row["pair_j"]),
    )


def load_member_index(paths: list[Path]) -> tuple[dict[tuple[int, int, int, int, int, int, int], dict[int, dict[str, object]]], list[Path]]:
    pair_slots: dict[tuple[int, int, int, int, int, int, int], dict[int, dict[str, object]]] = defaultdict(dict)
    missing_files: list[Path] = []
    for compact_path in paths:
        member_path = member_path_for_compact(compact_path)
        if not member_path.exists():
            missing_files.append(member_path)
            continue
        _, segment = parse_segment(compact_path)
        with member_path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = pair_key_from_member(row, segment)
                slot = int(row["pair_cluster_slot"])
                slot_data = pair_slots[key].setdefault(
                    slot,
                    {
                        "cluster_e": float(row["cluster_e"]),
                        "cluster_size": int(row["cluster_size"]),
                        "member_rows": 0,
                        "energy_sum": 0.0,
                        "blocks": defaultdict(float),
                    },
                )
                slot_data["member_rows"] = int(slot_data["member_rows"]) + 1
                energy = float(row["energy"])
                slot_data["energy_sum"] = float(slot_data["energy_sum"]) + energy
                cast_blocks = slot_data["blocks"]
                assert isinstance(cast_blocks, defaultdict)
                cast_blocks[int(row["block"])] += energy
    return pair_slots, missing_files


def shuffled_member_block_map(
    member_index: dict[tuple[int, int, int, int, int, int, int], dict[int, dict[str, object]]],
    *,
    seed: int,
) -> dict[int, int]:
    blocks: set[int] = set()
    for slots in member_index.values():
        for slot_data in slots.values():
            slot_blocks = slot_data["blocks"]
            assert isinstance(slot_blocks, defaultdict)
            blocks.update(int(block) for block in slot_blocks)
    src = sorted(blocks)
    dst = src[:]
    rng = random.Random(seed)
    rng.shuffle(dst)
    return dict(zip(src, dst))


def lower_energy_seed(row: dict[str, float]) -> int:
    if row["e1"] <= row["e2"]:
        return int(row.get("seed1_block", -1.0))
    return int(row.get("seed2_block", -1.0))


def lower_energy_slot(row: dict[str, float]) -> int:
    return 1 if row["e1"] <= row["e2"] else 2


def cluster_mode_ok_row(row: dict[str, float], mode: str) -> bool:
    return cluster_mode_ok(int(row["nclusters"]), mode)


def select_raw_rows(
    rows: list[dict[str, float]],
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
        if not cluster_mode_ok_row(row, nclusters_mode):
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
        if int(row.get("seed1_block", -1.0)) < 0 or int(row.get("seed2_block", -1.0)) < 0 or lower_energy_seed(row) < 0:
            continue
        selected.append(row)
    selected.sort(key=lambda r: (int(r["run"]), int(r["_segment"]), int(r["event"])))
    return selected


def fit_fold_shared_curve(rows: list[dict[str, float]]) -> tuple[np.ndarray, dict[str, float]]:
    coeffs, summary, _, _, _ = fit_shared_curve(rows)
    return coeffs, summary


def build_member_weights(blocks: defaultdict[int, float], norm_e: float, block_map: dict[int, int] | None = None) -> dict[int, float]:
    if not (norm_e > 0.0):
        return {}
    out: dict[int, float] = defaultdict(float)
    for block, energy in blocks.items():
        if energy <= 0.0:
            continue
        mapped_block = block_map.get(int(block), int(block)) if block_map is not None else int(block)
        out[mapped_block] += float(energy) / norm_e
    return dict(out)


def rows_to_member_events(
    rows: list[dict[str, float]],
    coeffs: np.ndarray,
    member_index: dict[tuple[int, int, int, int, int, int, int], dict[int, dict[str, object]]],
    *,
    rtol: float,
    atol: float,
    member_block_map: dict[int, int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    coeff = {"c0": float(coeffs[0]), "c1": float(coeffs[1]), "c2": float(coeffs[2])}
    events: list[dict[str, object]] = []
    closure = {
        "selected_rows": 0,
        "usable_rows": 0,
        "missing_member_pairs": 0,
        "missing_slot_rows": 0,
        "energy_mismatch_rows": 0,
        "size_mismatch_rows": 0,
        "examples": [],
    }
    for row in rows:
        closure["selected_rows"] = int(closure["selected_rows"]) + 1
        key = pair_key_from_compact(row)
        slots = member_index.get(key)
        if slots is None:
            closure["missing_member_pairs"] = int(closure["missing_member_pairs"]) + 1
            if len(closure["examples"]) < 10:
                closure["examples"].append(("missing_pair", key))
            continue
        slot1 = slots.get(1)
        slot2 = slots.get(2)
        if slot1 is None or slot2 is None:
            closure["missing_slot_rows"] = int(closure["missing_slot_rows"]) + 1
            if len(closure["examples"]) < 10:
                closure["examples"].append(("missing_slot", key))
            continue

        expected = {
            1: (float(row["e1"]), int(row["size1"])),
            2: (float(row["e2"]), int(row["size2"])),
        }
        bad = False
        for slot, slot_data in ((1, slot1), (2, slot2)):
            exp_e, exp_size = expected[slot]
            sum_e = float(slot_data["energy_sum"])
            cluster_e = float(slot_data["cluster_e"])
            member_rows = int(slot_data["member_rows"])
            if not close(sum_e, exp_e, rtol=rtol, atol=atol) or not close(cluster_e, exp_e, rtol=rtol, atol=atol):
                closure["energy_mismatch_rows"] = int(closure["energy_mismatch_rows"]) + 1
                if len(closure["examples"]) < 10:
                    closure["examples"].append(("energy_mismatch", key, slot, exp_e, sum_e, cluster_e))
                bad = True
                break
            if member_rows != exp_size:
                closure["size_mismatch_rows"] = int(closure["size_mismatch_rows"]) + 1
                if len(closure["examples"]) < 10:
                    closure["examples"].append(("size_mismatch", key, slot, exp_size, member_rows))
                bad = True
                break
        if bad:
            continue

        shared_mass = float(row["pair_m"]) * math.sqrt(shared_scale(coeff, float(row["e1"])) * shared_scale(coeff, float(row["e2"]))) * 1000.0
        weights1 = build_member_weights(slot1["blocks"], float(row["e1"]), member_block_map)  # type: ignore[arg-type]
        weights2 = build_member_weights(slot2["blocks"], float(row["e2"]), member_block_map)  # type: ignore[arg-type]
        low_slot = lower_energy_slot(row)
        low_weights = weights1 if low_slot == 1 else weights2
        events.append(
            {
                "run": int(row["run"]),
                "event": int(row["event"]),
                "segment": int(row["_segment"]),
                "seed1_block": int(row["seed1_block"]),
                "seed2_block": int(row["seed2_block"]),
                "low_seed_block": lower_energy_seed(row),
                "shared_mass_mev": shared_mass,
                "min_e": float(row["min_e"]),
                "weights1": weights1,
                "weights2": weights2,
                "low_weights": low_weights,
            }
        )
        closure["usable_rows"] = int(closure["usable_rows"]) + 1
    return events, closure


def fold_member_events(
    events: list[dict[str, object]], split_mode: str
) -> list[tuple[str, list[dict[str, object]], list[dict[str, object]]]]:
    if split_mode not in ("x60_4a2_4b", "x60_4a2_4b_stratified_run_parity"):
        return fold_events(events, split_mode)  # type: ignore[return-value]

    a2_runs = set(range(4253, 4373))
    b_runs = set(range(4397, 4569))
    a2 = [event for event in events if int(event["run"]) in a2_runs]
    b = [event for event in events if int(event["run"]) in b_runs]
    other = [event for event in events if int(event["run"]) not in a2_runs and int(event["run"]) not in b_runs]
    if other:
        runs = sorted({int(event["run"]) for event in other})
        raise ValueError(f"x60_4a2_4b split saw runs outside known ranges: {runs[:20]}")
    if split_mode == "x60_4a2_4b_stratified_run_parity":
        return [
            (
                "x60_4a2_4b_even_train_odd_test",
                [event for event in a2 + b if int(event["run"]) % 2 == 0],
                [event for event in a2 + b if int(event["run"]) % 2 == 1],
            ),
            (
                "x60_4a2_4b_odd_train_even_test",
                [event for event in a2 + b if int(event["run"]) % 2 == 1],
                [event for event in a2 + b if int(event["run"]) % 2 == 0],
            ),
        ]
    return [
        ("x60_4a2_train_x60_4b_test", a2, b),
        ("x60_4b_train_x60_4a2_test", b, a2),
    ]


def cluster_scale(weights: dict[int, float], block_scales: dict[int, float]) -> float:
    if not weights:
        return 1.0
    return sum(float(w) * float(block_scales.get(block, 1.0)) for block, w in weights.items())


def seed_cluster_scales(event: dict[str, object], seed_block_scales: dict[int, dict[str, float]]) -> tuple[float, float]:
    block1 = int(event["seed1_block"])
    block2 = int(event["seed2_block"])
    full1 = float(seed_block_scales[block1]["block_scale_full"]) if block1 in seed_block_scales else 1.0
    full2 = float(seed_block_scales[block2]["block_scale_full"]) if block2 in seed_block_scales else 1.0
    return full1, full2


def apply_member_stack(
    event: dict[str, object],
    slices: list[dict[str, object]],
    member_scales: dict[int, float],
    seed_block_scales: dict[int, dict[str, float]] | None = None,
) -> tuple[float, str]:
    time_scale, _, mode = time_scale_for_run(slices, int(event["run"]))
    shared_time_mass = float(event["shared_mass_mev"]) * time_scale
    c1 = cluster_scale(event["weights1"], member_scales)  # type: ignore[arg-type]
    c2 = cluster_scale(event["weights2"], member_scales)  # type: ignore[arg-type]
    if seed_block_scales is not None:
        s1, s2 = seed_cluster_scales(event, seed_block_scales)
        c1 *= s1
        c2 *= s2
    return shared_time_mass * math.sqrt(max(1e-12, c1 * c2)), mode


def derive_member_updates(
    events: list[dict[str, object]],
    slices: list[dict[str, object]],
    current_scales: dict[int, float],
    *,
    seed_block_scales: dict[int, dict[str, float]] | None,
    min_entries: int,
) -> dict[int, dict[str, float]]:
    by_block: dict[int, dict[str, float]] = defaultdict(lambda: {"entries": 0.0, "weight_sum": 0.0, "weighted_residual_sum": 0.0})
    for event in events:
        mass, _ = apply_member_stack(event, slices, current_scales, seed_block_scales)
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


def evaluate_models(
    events: list[dict[str, object]],
    slices: list[dict[str, object]],
    seed_block_scales: dict[int, dict[str, float]],
    member_scales: dict[int, float],
    *,
    member_label: str,
    member_base: str,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    residuals: dict[str, list[float]] = {
        "shared": [],
        "shared_time": [],
        "shared_time_seed_block": [],
        member_label: [],
    }
    counters: dict[str, int] = defaultdict(int)
    for event in events:
        time_scale, _, mode = time_scale_for_run(slices, int(event["run"]))
        counters[f"time_scale_{mode}"] += 1
        shared_mass = float(event["shared_mass_mev"])
        shared_time_mass = shared_mass * time_scale
        residuals["shared"].append(shared_mass - M_PI0_MEV)
        residuals["shared_time"].append(shared_time_mass - M_PI0_MEV)

        block1 = int(event["seed1_block"])
        block2 = int(event["seed2_block"])
        has1 = block1 in seed_block_scales
        has2 = block2 in seed_block_scales
        if has1 and has2:
            counters["both_seed_block_scales"] += 1
        elif has1 or has2:
            counters["one_seed_block_scale"] += 1
        else:
            counters["no_seed_block_scale"] += 1
        full1 = float(seed_block_scales[block1]["block_scale_full"]) if has1 else 1.0
        full2 = float(seed_block_scales[block2]["block_scale_full"]) if has2 else 1.0
        residuals["shared_time_seed_block"].append(
            shared_time_mass * math.sqrt(max(1e-12, full1 * full2)) - M_PI0_MEV
        )

        member_seed_scales = seed_block_scales if member_base == "seed_block" else None
        member_mass, _ = apply_member_stack(event, slices, member_scales, member_seed_scales)
        residuals[member_label].append(member_mass - M_PI0_MEV)
    return residuals, counters


def assign_bin(emin: float, edges: list[float]) -> tuple[float, float] | None:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= emin < hi:
            return lo, hi
    if edges and math.isclose(emin, edges[-1]):
        return edges[-2], edges[-1]
    return None


def clip_update_scale(update_scale: float, max_update_frac: float) -> float:
    if max_update_frac <= 0.0:
        return update_scale
    lo = max(1e-12, 1.0 - max_update_frac)
    hi = 1.0 + max_update_frac
    return min(max(update_scale, lo), hi)


def energy_bin_edges_for_closure(emin_range: tuple[float, float]) -> list[float]:
    requested = [0.6, 0.8, 1.2, 1.6, 2.0, 2.5]
    lo, hi = emin_range
    edges = [edge for edge in requested if lo <= edge <= hi]
    if not edges or edges[0] > lo:
        edges = [lo] + edges
    if edges[-1] < hi:
        edges.append(hi)
    return edges


def write_energy_bin_closure(
    events: list[dict[str, object]],
    seed_residuals: list[float],
    member_residuals: list[float],
    out_path: Path,
    *,
    emin_range: tuple[float, float],
) -> None:
    edges = energy_bin_edges_for_closure(emin_range)
    by_model: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for event, seed_residual, member_residual in zip(events, seed_residuals, member_residuals):
        eb = assign_bin(float(event["min_e"]), edges)
        if eb is None:
            continue
        lo, hi = eb
        by_model[("shared_time_seed_block", lo, hi)].append(float(seed_residual))
        by_model[("member_weighted", lo, hi)].append(float(member_residual))

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["model", "elo", "ehi", "entries", "mean_mev", "mae_mev", "rmse_mev", "sigma_mev"])
        for lo, hi in zip(edges[:-1], edges[1:]):
            for model in ("shared_time_seed_block", "member_weighted"):
                vals = by_model.get((model, lo, hi), [])
                row = metrics(vals)
                writer.writerow([model, lo, hi, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]])


def write_residual_products(
    events: list[dict[str, object]],
    residuals: list[float],
    outdir: Path,
    *,
    emin_range: tuple[float, float],
    min_block_entries: int,
    min_bin_entries: int,
    label: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    edges = [e for e in DEFAULT_EDGES if emin_range[0] <= e <= emin_range[1]]
    if not edges:
        edges = [emin_range[0], emin_range[1]]
    if edges[0] > emin_range[0]:
        edges = [emin_range[0]] + edges
    if edges[-1] < emin_range[1]:
        edges = edges + [emin_range[1]]

    block_all: dict[int, list[float]] = defaultdict(list)
    block_bins: dict[tuple[int, float, float], list[float]] = defaultdict(list)
    occupancy: dict[int, int] = defaultdict(int)
    for event, residual in zip(events, residuals):
        block = int(event["low_seed_block"])
        block_all[block].append(float(residual))
        occupancy[block] += 1
        eb = assign_bin(float(event["min_e"]), edges)
        if eb is not None:
            block_bins[(block, eb[0], eb[1])].append(float(residual))

    with (outdir / "block_summary.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seed_block", "entries", "mean_residual_mev"])
        for block in sorted(block_all):
            vals = block_all[block]
            if len(vals) < min_block_entries:
                continue
            writer.writerow([block, len(vals), float(np.mean(vals))])

    with (outdir / "block_bin_residuals.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seed_block", "elo", "ehi", "entries", "mean_residual_mev"])
        for (block, lo, hi) in sorted(block_bins):
            vals = block_bins[(block, lo, hi)]
            if len(vals) < min_bin_entries:
                continue
            writer.writerow([block, lo, hi, len(vals), float(np.mean(vals))])

    with (outdir / "occupancy.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seed_block", "row", "col", "count"])
        for block, count in sorted(occupancy.items()):
            row = block // 30
            col = block % 30
            writer.writerow([block, row, col, count])

    worst = sorted(
        ((block, len(vals), float(np.mean(vals))) for block, vals in block_all.items() if len(vals) >= min_block_entries),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    with (outdir / "left_edge_outliers.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["seed_block", "row", "col", "entries", "mean_residual_mev"])
        for block, entries, mean_resid in worst[:20]:
            row = block // 30
            col = block % 30
            writer.writerow([block, row, col, entries, mean_resid])

    block_means = [float(np.mean(vals)) for block, vals in block_all.items() if len(vals) >= min_block_entries]
    summary = [
        f"{label} residual summary",
        f"selected_rows: {len(events)}",
        f"blocks_with_stats: {sum(1 for vals in block_all.values() if len(vals) >= min_block_entries)}",
        f"block_energy_bins_with_stats: {sum(1 for vals in block_bins.values() if len(vals) >= min_bin_entries)}",
    ]
    if block_means:
        abs_means = np.abs(np.asarray(block_means, dtype=float))
        summary.extend(
            [
                f"median_abs_mean_residual_mev: {float(np.median(abs_means)):.6f}",
                f"p95_abs_mean_residual_mev: {float(np.percentile(abs_means, 95)):.6f}",
                f"block_mean_spread_mev: {float(max(block_means) - min(block_means)):.6f}",
            ]
        )
    (outdir / "block_residual_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    input_paths = [normalize_path(p) for p in args.input_tsv]
    if args.input_summary:
        input_paths.extend(input_paths_from_summary(Path(args.input_summary)))
    input_paths = list(dict.fromkeys(input_paths))
    if not input_paths:
        raise SystemExit("no input TSVs provided")

    rows = read_rows(input_paths)
    member_index, missing_member_files = load_member_index(input_paths)
    timing_basis = [
        r
        for r in rows
        if cluster_mode_ok_row(r, args.nclusters_mode)
        and r["e1"] >= 0.6
        and r["e2"] >= 0.6
        and r["min_e"] >= args.min_e_floor
        and abs(r["x1"]) < 29.025
        and abs(r["x2"]) < 29.025
        and abs(r["y1"]) < 35.475
        and abs(r["y2"]) < 35.475
    ]
    timing_center = resolve_timing_center(timing_basis, parse_timing_center(args.timing_center))
    selected_rows = select_raw_rows(
        rows,
        mass_window=(args.mass_window[0], args.mass_window[1]),
        timing_window=args.timing_window,
        timing_center=timing_center,
        min_e_floor=args.min_e_floor,
        nclusters_mode=args.nclusters_mode,
        emin_range=(args.emin_range[0], args.emin_range[1]),
    )

    placeholder_events = [{"run": int(r["run"]), "segment": int(r["_segment"]), "event": int(r["event"])} for r in selected_rows]
    event_key_to_row = {(int(r["run"]), int(r["_segment"]), int(r["event"])): r for r in selected_rows}

    pooled_test: dict[str, list[float]] = defaultdict(list)
    fold_metric_rows: list[list[object]] = []
    fold_update_rows: list[list[object]] = []
    fold_summary_rows: list[list[object]] = []
    pooled_seed_events: list[dict[str, object]] = []
    pooled_seed_residuals: list[float] = []
    pooled_member_events: list[dict[str, object]] = []
    pooled_member_residuals: list[float] = []
    closure_examples: list[object] = []
    member_model_prefix = "shared_time_seed_member_iter" if args.member_base == "seed_block" else "shared_time_member_iter"

    for fold_index, (fold, train_keys, test_keys) in enumerate(fold_member_events(placeholder_events, args.split_mode)):
        train_rows = [event_key_to_row[(int(e["run"]), int(e["segment"]), int(e["event"]))] for e in train_keys]
        test_rows = [event_key_to_row[(int(e["run"]), int(e["segment"]), int(e["event"]))] for e in test_keys]

        coeffs, fit_summary = fit_fold_shared_curve(train_rows)
        train_member_block_map = (
            shuffled_member_block_map(member_index, seed=args.shuffle_seed + 2 * fold_index)
            if args.shuffle_member_blocks
            else None
        )
        test_member_block_map = (
            shuffled_member_block_map(member_index, seed=args.shuffle_seed + 2 * fold_index + 1)
            if args.shuffle_member_blocks
            else None
        )

        train_events, train_closure = rows_to_member_events(
            train_rows,
            coeffs,
            member_index,
            rtol=args.rtol,
            atol=args.atol,
            member_block_map=train_member_block_map,
        )
        test_events, test_closure = rows_to_member_events(
            test_rows,
            coeffs,
            member_index,
            rtol=args.rtol,
            atol=args.atol,
            member_block_map=test_member_block_map,
        )
        for example in list(train_closure["examples"]) + list(test_closure["examples"]):
            if len(closure_examples) < 20:
                closure_examples.append(example)

        if not train_events or not test_events:
            raise SystemExit(f"{fold}: no usable member-weighted events after sidecar closure checks")

        slices = build_slices(train_events, args.n_slices)
        seed_block_scales = derive_block_scales(train_events, slices, args.min_block_entries)
        current_member_scales: dict[int, float] = {}

        base_member_label = f"{member_model_prefix}_0"
        for sample_name, sample in (("train", train_events), ("test", test_events)):
            residuals, counters = evaluate_models(
                sample,
                slices,
                seed_block_scales,
                current_member_scales,
                member_label=base_member_label,
                member_base=args.member_base,
            )
            for model, vals in residuals.items():
                row = metrics(vals)
                fold_metric_rows.append([fold, 0, sample_name, model, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]])
                if sample_name == "test" and model in ("shared", "shared_time", "shared_time_seed_block"):
                    pooled_test[model].extend(vals)
            fold_summary_rows.append(
                [
                    fold,
                    0,
                    sample_name,
                    len(sample),
                    len(slices),
                    len(seed_block_scales),
                    len(current_member_scales),
                    counters.get("time_scale_range", 0),
                    counters.get("time_scale_nearest", 0),
                    counters.get("both_seed_block_scales", 0),
                    counters.get("one_seed_block_scale", 0),
                    counters.get("no_seed_block_scale", 0),
                    fit_summary["corr_rmse_mev"],
                    train_closure["usable_rows"],
                    test_closure["usable_rows"],
                ]
            )

        for iteration in range(1, args.iterations + 1):
            updates = derive_member_updates(
                train_events,
                slices,
                current_member_scales,
                seed_block_scales=seed_block_scales if args.member_base == "seed_block" else None,
                min_entries=args.min_block_entries,
            )
            for block, row in sorted(updates.items()):
                update_scale = float(row["update_scale_full"]) ** float(args.damping)
                update_scale = clip_update_scale(update_scale, float(args.max_update_frac))
                prev_scale = float(current_member_scales.get(block, 1.0))
                new_scale = prev_scale * update_scale
                current_member_scales[block] = new_scale
                fold_update_rows.append(
                    [
                        fold,
                        iteration,
                        block,
                        int(row["entries"]),
                        row["weight_sum"],
                        row["mean_residual_mev"],
                        row["update_scale_full"],
                        update_scale,
                        prev_scale,
                        new_scale,
                    ]
                )

            member_label = f"{member_model_prefix}_{iteration}"
            for sample_name, sample in (("train", train_events), ("test", test_events)):
                residuals, counters = evaluate_models(
                    sample,
                    slices,
                    seed_block_scales,
                    current_member_scales,
                    member_label=member_label,
                    member_base=args.member_base,
                )
                for model, vals in residuals.items():
                    if model not in (member_label,):
                        continue
                    row = metrics(vals)
                    fold_metric_rows.append([fold, iteration, sample_name, model, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]])
                    if sample_name == "test":
                        pooled_test[model].extend(vals)
                fold_summary_rows.append(
                    [
                        fold,
                        iteration,
                        sample_name,
                        len(sample),
                        len(slices),
                        len(seed_block_scales),
                        len(current_member_scales),
                        counters.get("time_scale_range", 0),
                        counters.get("time_scale_nearest", 0),
                        counters.get("both_seed_block_scales", 0),
                        counters.get("one_seed_block_scale", 0),
                        counters.get("no_seed_block_scale", 0),
                        fit_summary["corr_rmse_mev"],
                        train_closure["usable_rows"],
                        test_closure["usable_rows"],
                    ]
                )

        final_member_label = f"{member_model_prefix}_{args.iterations}"
        seed_residuals_dict, _ = evaluate_models(
            test_events,
            slices,
            seed_block_scales,
            current_member_scales,
            member_label=final_member_label,
            member_base=args.member_base,
        )
        pooled_seed_events.extend(test_events)
        pooled_seed_residuals.extend(seed_residuals_dict["shared_time_seed_block"])
        pooled_member_events.extend(test_events)
        pooled_member_residuals.extend(seed_residuals_dict[final_member_label])

    with (outdir / "fold_metrics.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["fold", "iteration", "sample", "model", "entries", "mean_mev", "mae_mev", "rmse_mev", "sigma_mev"])
        writer.writerows(fold_metric_rows)

    with (outdir / "fold_update_table.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "fold",
                "iteration",
                "block",
                "entries",
                "weight_sum",
                "mean_residual_mev",
                "update_scale_full",
                "applied_update_scale",
                "prev_scale",
                "new_scale",
            ]
        )
        writer.writerows(fold_update_rows)

    with (outdir / "fold_summary.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "fold",
                "iteration",
                "sample",
                "events",
                "time_slices",
                "seed_block_scales",
                "member_block_scales",
                "events_with_time_scale_by_range",
                "events_with_time_scale_by_nearest",
                "events_with_both_seed_block_scales",
                "events_with_one_seed_block_scale",
                "events_without_seed_block_scale",
                "train_shared_fit_rmse_mev",
                "train_usable_member_rows",
                "test_usable_member_rows",
            ]
        )
        writer.writerows(fold_summary_rows)

    with (outdir / "pooled_test_metrics.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["model", "entries", "mean_mev", "mae_mev", "rmse_mev", "sigma_mev"])
        model_order = ["shared", "shared_time", "shared_time_seed_block"] + [f"{member_model_prefix}_{i}" for i in range(1, args.iterations + 1)]
        for model in model_order:
            vals = pooled_test.get(model, [])
            if not vals:
                continue
            row = metrics(vals)
            writer.writerow([model, row["entries"], row["mean_mev"], row["mae_mev"], row["rmse_mev"], row["sigma_mev"]])

    write_residual_products(
        pooled_seed_events,
        pooled_seed_residuals,
        outdir / "seed_block_post_stack",
        emin_range=(args.emin_range[0], args.emin_range[1]),
        min_block_entries=args.min_block_entries,
        min_bin_entries=args.min_bin_entries,
        label="Held-out seed-block baseline",
    )
    write_residual_products(
        pooled_member_events,
        pooled_member_residuals,
        outdir / f"member_weighted_post_stack_iter_{args.iterations}",
        emin_range=(args.emin_range[0], args.emin_range[1]),
        min_block_entries=args.min_block_entries,
        min_bin_entries=args.min_bin_entries,
        label=f"Held-out member-weighted stack iter {args.iterations}",
    )
    write_energy_bin_closure(
        pooled_member_events,
        pooled_seed_residuals,
        pooled_member_residuals,
        outdir / "energy_bin_closure.tsv",
        emin_range=(args.emin_range[0], args.emin_range[1]),
    )

    summary = [
        "Held-out member-weighted block validation",
        f"input_tsvs: {len(input_paths)}",
        f"selected_rows_pre_member_checks: {len(selected_rows)}",
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
        f"damping: {args.damping:.6f}",
        f"iterations: {args.iterations}",
        f"member_base: {args.member_base}",
        f"max_update_frac: {args.max_update_frac:.6f}",
        f"shuffle_member_blocks: {int(args.shuffle_member_blocks)}",
        f"shuffle_seed: {args.shuffle_seed}",
        f"shuffle_map_mode: independent_train_test_maps_per_fold",
        f"missing_member_files: {len(missing_member_files)}",
        "",
        "Conventions:",
        "  shared curve is refit on each training fold only",
        "  time-slice scales are built from shared-only training-fold events",
        "  seed-block baseline is derived from lower-energy seed residuals on the training fold",
        "  member-weighted updates are derived from lower-energy cluster members on the training fold",
        "  application uses both clusters via deposit-weighted cluster factors",
        "",
        "Member-sidecar closure examples:",
    ]
    for example in closure_examples[:20]:
        summary.append(f"  {example}")
    (outdir / "member_weighted_summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
