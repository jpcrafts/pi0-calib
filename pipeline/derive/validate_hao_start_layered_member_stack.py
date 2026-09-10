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
from plot_compact_shared_curve_mass_fits import plot_fit
from plot_period_fallback_mgg_fits import fit_peak


ENERGY_BINS = [(0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 2.5)]
MODELS = ["hao_raw", "hao_energy", "hao_energy_run", "hao_energy_run_seed", "hao_energy_run_seed_member"]


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fold validation for a Hao-start energy/run/seed/member residual stack.")
    ap.add_argument("--conversion-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--timing-center", type=float, default=None)
    ap.add_argument("--timing-window", type=float, default=1.0)
    ap.add_argument("--emin-range", nargs=2, type=float, default=[0.6, 2.5])
    ap.add_argument("--fit-window", nargs=2, type=float, default=[105.0, 165.0])
    ap.add_argument("--hist-window", nargs=2, type=float, default=[80.0, 200.0])
    ap.add_argument("--min-fit-entries", type=int, default=200)
    ap.add_argument("--min-seed-events", type=int, default=80)
    ap.add_argument("--min-member-weight", type=float, default=20.0)
    ap.add_argument("--energy-damping", type=float, default=0.75)
    ap.add_argument("--run-damping", type=float, default=0.75)
    ap.add_argument("--seed-damping", type=float, default=0.50)
    ap.add_argument("--member-damping", type=float, default=0.20)
    ap.add_argument("--max-energy-mass-update-frac", type=float, default=0.03)
    ap.add_argument("--max-run-mass-update-frac", type=float, default=0.02)
    ap.add_argument("--max-seed-energy-update-frac", type=float, default=0.01)
    ap.add_argument("--max-member-energy-update-frac", type=float, default=0.003)
    ap.add_argument("--left-cols", nargs="+", type=int, default=[0, 1, 2])
    return ap.parse_args()


def energy_bin(min_e: float) -> tuple[float, float] | None:
    for lo, hi in ENERGY_BINS:
        if lo <= min_e < hi or (math.isclose(hi, 2.5) and lo <= min_e <= hi):
            return lo, hi
    return None


def bin_label(bin_pair: tuple[float, float]) -> str:
    return f"{bin_pair[0]:.1f}-{bin_pair[1]:.1f}"


def clamp(v: float, frac: float) -> float:
    if frac <= 0.0:
        return v
    return min(max(v, 1.0 - frac), 1.0 + frac)


def load_conversion_summary(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def member_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (int(row["run"]), int(row["segment"]), int(row["event"]), int(row["prod_cluster_slot"]))


def load_members(path: Path) -> dict[tuple[int, int, int, int], list[dict[str, float]]]:
    out: dict[tuple[int, int, int, int], list[dict[str, float]]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = member_key(row)
            out[key].append(
                {
                    "block": float(row["block"]),
                    "row": float(row["row"]),
                    "col": float(row["col"]),
                    "member_energy": float(row["member_energy"]),
                    "member_fraction": float(row["member_fraction"]),
                }
            )
    return out


def read_events(summary_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    qa: list[dict[str, object]] = []
    for info in summary_rows:
        compact = Path(info["output_tsv"])
        sidecar = Path(info["sidecar"])
        members = load_members(sidecar)
        total = selected = missing = 0
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
                run = int(row["run"])
                seg = int(info["segment"])
                event = int(row["event"])
                m1 = members.get((run, seg, event, 0), [])
                m2 = members.get((run, seg, event, 1), [])
                if not m1 or not m2:
                    missing += 1
                    continue
                low_slot = 0 if e1 <= e2 else 1
                low_seed = int(row["seed1_block"]) if low_slot == 0 else int(row["seed2_block"])
                low_col = int(row["seed1_col"]) if low_slot == 0 else int(row["seed2_col"])
                events.append(
                    {
                        "run": run,
                        "segment": seg,
                        "event": event,
                        "fold": seg % 2,
                        "mass_mev": float(row["pair_m"]) * 1000.0,
                        "e1": e1,
                        "e2": e2,
                        "min_e": min_e,
                        "seed1_block": int(row["seed1_block"]),
                        "seed2_block": int(row["seed2_block"]),
                        "seed1_col": int(row["seed1_col"]),
                        "seed2_col": int(row["seed2_col"]),
                        "low_seed": low_seed,
                        "low_col": low_col,
                        "members1": m1,
                        "members2": m2,
                    }
                )
                selected += 1
        qa.append({"run": info["run"], "segment": seg, "total_rows": total, "selected_events": selected, "missing_member_events": missing})
    return events, qa


def derive_energy_updates(train: list[dict[str, object]], args: argparse.Namespace) -> tuple[dict[str, float], list[dict[str, object]]]:
    updates: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for eb in ENERGY_BINS:
        label = bin_label(eb)
        vals = [float(ev["mass_mev"]) for ev in train if energy_bin(float(ev["min_e"])) == eb]
        fit = fit_peak(np.asarray(vals, dtype=float), args.fit_window[0], args.fit_window[1])
        ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_fit_entries and math.isfinite(float(fit["mu"]))
        full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
        applied = clamp(full**args.energy_damping, args.max_energy_mass_update_frac)
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
                "factor_applied": applied,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[label] = applied
    return updates, rows


def apply_energy(mass: float, ev: dict[str, object], updates: dict[str, float]) -> float:
    eb = energy_bin(float(ev["min_e"]))
    if eb is None:
        return mass
    return mass * updates.get(bin_label(eb), 1.0)


def derive_run_updates(train: list[dict[str, object]], base_masses: dict[int, float], args: argparse.Namespace) -> tuple[dict[int, float], list[dict[str, object]]]:
    updates: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    by_run: dict[int, list[float]] = defaultdict(list)
    for i, ev in enumerate(train):
        by_run[int(ev["run"])].append(base_masses[i])
    for run, vals in sorted(by_run.items()):
        fit = fit_peak(np.asarray(vals, dtype=float), args.fit_window[0], args.fit_window[1])
        ok = int(fit["fit_ok"]) == 1 and int(fit["entries"]) >= args.min_fit_entries and math.isfinite(float(fit["mu"]))
        full = M_PI0_MEV / float(fit["mu"]) if ok else 1.0
        applied = clamp(full**args.run_damping, args.max_run_mass_update_frac)
        rows.append(
            {
                "layer": "run",
                "key": run,
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
            updates[run] = applied
    return updates, rows


def derive_seed_updates(train: list[dict[str, object]], base_masses: dict[int, float], args: argparse.Namespace) -> tuple[dict[int, float], list[dict[str, object]]]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for i, ev in enumerate(train):
        if int(ev["low_col"]) in set(args.left_cols):
            continue
        by_seed[int(ev["low_seed"])].append(base_masses[i])
    updates: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for seed, vals in sorted(by_seed.items()):
        residuals = np.asarray(vals, dtype=float) - M_PI0_MEV
        median_residual = float(np.median(residuals)) if residuals.size else math.nan
        ok = len(vals) >= args.min_seed_events and math.isfinite(median_residual)
        full_energy = (M_PI0_MEV / (M_PI0_MEV + median_residual)) ** 2 if ok else 1.0
        applied_energy = clamp(full_energy**args.seed_damping, args.max_seed_energy_update_frac)
        rows.append(
            {
                "layer": "seed",
                "key": seed,
                "entries": len(vals),
                "median_residual_mev": median_residual,
                "factor_full": full_energy,
                "factor_applied": applied_energy,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[seed] = applied_energy
    return updates, rows


def apply_seed(mass: float, ev: dict[str, object], updates: dict[int, float]) -> float:
    return mass * math.sqrt(max(1.0e-12, updates.get(int(ev["low_seed"]), 1.0)))


def derive_member_updates(train: list[dict[str, object]], base_masses: dict[int, float], args: argparse.Namespace) -> tuple[dict[int, float], list[dict[str, object]]]:
    weighted_resid: dict[int, list[tuple[float, float]]] = defaultdict(list)
    left_cols = set(args.left_cols)
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
    updates: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for block, vals in sorted(weighted_resid.items()):
        wsum = sum(w for _r, w in vals)
        mean = sum(r * w for r, w in vals) / wsum if wsum > 0 else math.nan
        ok = wsum >= args.min_member_weight and math.isfinite(mean)
        full_energy = (M_PI0_MEV / (M_PI0_MEV + mean)) ** 2 if ok else 1.0
        applied_energy = clamp(full_energy**args.member_damping, args.max_member_energy_update_frac)
        rows.append(
            {
                "layer": "member",
                "key": block,
                "entries": len(vals),
                "weight_sum": wsum,
                "weighted_mean_residual_mev": mean,
                "factor_full": full_energy,
                "factor_applied": applied_energy,
                "support_pass": int(ok),
            }
        )
        if ok:
            updates[block] = applied_energy
    return updates, rows


def apply_member(mass: float, ev: dict[str, object], updates: dict[int, float]) -> float:
    cluster_factors = []
    for cluster_key in ("members1", "members2"):
        factor = 0.0
        weight = 0.0
        for member in ev[cluster_key]:  # type: ignore[index]
            w = max(0.0, float(member["member_fraction"]))
            factor += w * updates.get(int(member["block"]), 1.0)
            weight += w
        if weight > 0:
            factor /= weight
        else:
            factor = 1.0
        cluster_factors.append(factor)
    return mass * math.sqrt(max(1.0e-12, cluster_factors[0] * cluster_factors[1]))


def fit_eval_rows(fold: str, sample: str, events: list[dict[str, object]], masses_by_model: dict[str, list[float]], args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, list[int]]] = [("pooled", list(range(len(events))))]
    for run in sorted({int(ev["run"]) for ev in events}):
        groups.append((f"run_{run}", [i for i, ev in enumerate(events) if int(ev["run"]) == run]))
    for eb in ENERGY_BINS:
        label = f"emin_{bin_label(eb)}"
        groups.append((label, [i for i, ev in enumerate(events) if energy_bin(float(ev["min_e"])) == eb]))
    for eb in ENERGY_BINS:
        e_label = bin_label(eb)
        for run in sorted({int(ev["run"]) for ev in events}):
            groups.append(
                (
                    f"run_{run}_emin_{e_label}",
                    [i for i, ev in enumerate(events) if int(ev["run"]) == run and energy_bin(float(ev["min_e"])) == eb],
                )
            )
    for group, idxs in groups:
        if not idxs:
            continue
        for model in MODELS:
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


def plot_trends(rows: list[dict[str, object]], outdir: Path) -> None:
    pdf = outdir / "hao_start_layered_member_stack_trends.pdf"
    with PdfPages(pdf) as pp:
        for group in ["pooled", "emin_0.6-0.8", "emin_0.8-1.0", "emin_1.0-1.2", "emin_1.2-1.5", "emin_1.5-2.0", "emin_2.0-2.5"]:
            fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
            labels = []
            for model in MODELS:
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
            axs[0].set_title(f"Hao-start layered stack held-out fits: {group}")
            axs[1].set_ylabel("fit sigma (MeV)")
            axs[1].set_xlabel("held-out fold")
            for ax in axs:
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def plot_run_energy_trends(rows: list[dict[str, object]], outdir: Path) -> None:
    pdf = outdir / "hao_start_layered_member_stack_run_energy_trends.pdf"
    with PdfPages(pdf) as pp:
        for eb in ENERGY_BINS:
            e_label = bin_label(eb)
            fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
            for model in MODELS:
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
            axs[0].set_title(f"Hao-start layered stack: run trend, min_pair_E {e_label} GeV")
            axs[1].set_ylabel("fit sigma (MeV)")
            axs[1].set_xlabel("run")
            for ax in axs:
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def plot_mass_pages(rows_events: list[dict[str, object]], masses_by_model: dict[str, list[float]], outdir: Path, args: argparse.Namespace) -> None:
    pdf = outdir / "hao_start_layered_member_stack_mass_fits.pdf"
    with PdfPages(pdf) as pp:
        for group, idxs in [
            ("pooled", list(range(len(rows_events)))),
            ("emin_0.6-0.8", [i for i, ev in enumerate(rows_events) if energy_bin(float(ev["min_e"])) == (0.6, 0.8)]),
        ]:
            if not idxs:
                continue
            fig, axs = plt.subplots(1, len(MODELS), figsize=(4.0 * len(MODELS), 3.8), constrained_layout=True)
            for ax, model in zip(axs, MODELS):
                masses = np.asarray([masses_by_model[model][i] for i in idxs], dtype=float)
                plot_fit(ax, masses, f"{group} {model}", tuple(args.hist_window), tuple(args.fit_window))
            pp.savefig(fig)
            plt.close(fig)
    print(pdf)


def evaluate_fold(name: str, train: list[dict[str, object]], test: list[dict[str, object]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]]]:
    update_rows: list[dict[str, object]] = []
    energy_updates, rows = derive_energy_updates(train, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_energy = {i: apply_energy(float(ev["mass_mev"]), ev, energy_updates) for i, ev in enumerate(train)}
    run_updates, rows = derive_run_updates(train, train_energy, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_energy_run = {i: train_energy[i] * run_updates.get(int(ev["run"]), 1.0) for i, ev in enumerate(train)}
    seed_updates, rows = derive_seed_updates(train, train_energy_run, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    train_seed = {i: apply_seed(train_energy_run[i], ev, seed_updates) for i, ev in enumerate(train)}
    member_updates, rows = derive_member_updates(train, train_seed, args)
    update_rows.extend({"fold": name, **row} for row in rows)

    masses: dict[str, list[float]] = {model: [] for model in MODELS}
    for ev in test:
        raw = float(ev["mass_mev"])
        energy = apply_energy(raw, ev, energy_updates)
        energy_run = energy * run_updates.get(int(ev["run"]), 1.0)
        seed = apply_seed(energy_run, ev, seed_updates)
        member = apply_member(seed, ev, member_updates)
        masses["hao_raw"].append(raw)
        masses["hao_energy"].append(energy)
        masses["hao_energy_run"].append(energy_run)
        masses["hao_energy_run_seed"].append(seed)
        masses["hao_energy_run_seed_member"].append(member)
    fit_rows = fit_eval_rows(name, "test", test, masses, args)
    return fit_rows, update_rows, masses


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = load_conversion_summary(Path(args.conversion_summary))
    events, qa = read_events(summary, args)
    write_tsv(outdir / "sidecar_quality.tsv", qa)

    all_fit_rows: list[dict[str, object]] = []
    all_update_rows: list[dict[str, object]] = []
    pooled_test_events: list[dict[str, object]] = []
    pooled_masses: dict[str, list[float]] = {model: [] for model in MODELS}
    for test_fold in [0, 1]:
        train = [ev for ev in events if int(ev["fold"]) != test_fold]
        test = [ev for ev in events if int(ev["fold"]) == test_fold]
        fit_rows, update_rows, masses = evaluate_fold(f"segment_parity_test_{test_fold}", train, test, args)
        all_fit_rows.extend(fit_rows)
        all_update_rows.extend(update_rows)
        pooled_test_events.extend(test)
        for model in MODELS:
            pooled_masses[model].extend(masses[model])

    all_fit_rows.extend(fit_eval_rows("pooled_folds", "test", pooled_test_events, pooled_masses, args))
    write_tsv(outdir / "heldout_fit_metrics.tsv", all_fit_rows)
    write_tsv(outdir / "fold_update_tables.tsv", all_update_rows)
    plot_trends(all_fit_rows, outdir)
    plot_run_energy_trends(all_fit_rows, outdir)
    plot_mass_pages(pooled_test_events, pooled_masses, outdir, args)
    print(outdir / "heldout_fit_metrics.tsv")
    print(outdir / "fold_update_tables.tsv")


if __name__ == "__main__":
    main()
