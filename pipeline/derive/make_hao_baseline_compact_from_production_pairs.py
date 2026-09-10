#!/usr/bin/env python3
"""Build compact calibration TSVs from Hao-baseline production replay pairs.

The production exact-2 pair TSVs do not carry seed blocks.  The matching
production member sidecars do, keyed by run/segment/t_entry/source_cluster_index.
This script joins the two and writes compact TSVs with the same core columns used
by the raw-WF calibration scanners.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import re
from pathlib import Path


PAIR_RE = re.compile(r"run(\d+)_seg(\d+)_production_exact2_pairs\.tsv$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", nargs="+", type=Path)
    source.add_argument("--pairs-dir", action="append", type=Path)
    ap.add_argument("--runs", type=Path, help="Optional run manifest used to filter --pairs-dir inputs.")
    ap.add_argument("--sidecar-dir", action="append", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--max-missing-fraction", type=float, default=0.001,
        help="Fail when aggregate pair rows missing an exact sidecar join exceed this fraction.",
    )
    return ap.parse_args()


def load_runs(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    runs: set[int] = set()
    with path.open() as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#") or value.lower().startswith("run"):
                continue
            runs.add(int(value.split()[0]))
    return runs


def discover_pairs(args: argparse.Namespace) -> list[Path]:
    paths = list(args.pairs or [])
    for base in args.pairs_dir or []:
        paths.extend(base.glob("run*_seg*_production_exact2_pairs.tsv"))
    allowed = load_runs(args.runs)
    out: list[Path] = []
    for path in sorted(set(paths)):
        match = PAIR_RE.search(path.name)
        if match and (allowed is None or int(match.group(1)) in allowed):
            out.append(path)
    return out


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def i(row: dict[str, str], key: str, default: int = -1) -> int:
    val = row.get(key, "")
    return default if val == "" else int(float(val))


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    val = row.get(key, "")
    return default if val == "" else float(val)


def sidecar_path(sidecar_dirs: list[Path], run: int, seg: int) -> Path | None:
    name = f"nps_production_{run}_{seg}_wf_calib.production_members.tsv"
    for base in sidecar_dirs:
        path = base / name
        if path.exists():
            return path
    return None


def load_sidecar(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    # One row per cluster is enough; member_index 0 carries cluster metadata.
    out: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            # NPS.prod.* pair indices address the flattened production vector.
            # The matching sidecar field is prod_cluster_slot; source_cluster_index
            # restarts independently inside each accidental/production time window.
            key = (i(row, "t_entry"), i(row, "prod_cluster_slot"))
            if key not in out:
                out[key] = row
    return out


FIELDS = [
    "t_entry",
    "wf_entry",
    "run",
    "event",
    "nclusters",
    "pair_i",
    "pair_j",
    "pair_m",
    "e1",
    "e2",
    "min_e",
    "x1",
    "y1",
    "x2",
    "y2",
    "t1",
    "t2",
    "dt12",
    "size1",
    "size2",
    "seed1_block",
    "seed2_block",
    "seed1_row",
    "seed1_col",
    "seed2_row",
    "seed2_col",
]


def convert_pair_file(pair_path: Path, sidecar_dirs: list[Path], outdir: Path) -> tuple[Path, dict[str, object]]:
    match = PAIR_RE.search(pair_path.name)
    if not match:
        raise SystemExit(f"cannot parse run/segment from {pair_path}")
    run = int(match.group(1))
    seg = int(match.group(2))
    side_path = sidecar_path(sidecar_dirs, run, seg)
    if side_path is None:
        return outdir / f"nps_production_{run}_{seg}_wf.tsv", {
            "run": run,
            "segment": seg,
            "input_pairs": pair_path,
            "sidecar": "",
            "output_tsv": "",
            "input_rows": 0,
            "output_rows": 0,
            "missing_clusters": 0,
            "status": "missing_sidecar",
        }

    side = load_sidecar(side_path)
    rows_out: list[dict[str, object]] = []
    missing = 0
    pair_rows = read_tsv(pair_path)
    for row in pair_rows:
        t_entry = i(row, "t_entry")
        c1 = side.get((t_entry, i(row, "pair_i")))
        c2 = side.get((t_entry, i(row, "pair_j")))
        if c1 is None or c2 is None:
            missing += 1
            continue
        e1 = f(row, "e1")
        e2 = f(row, "e2")
        rows_out.append(
            {
                "t_entry": t_entry,
                "wf_entry": i(c1, "wf_entry", i(c2, "wf_entry")),
                "run": run,
                "event": i(row, "event"),
                "nclusters": 2,
                "pair_i": i(row, "pair_i"),
                "pair_j": i(row, "pair_j"),
                "pair_m": f(row, "pair_m"),
                "e1": e1,
                "e2": e2,
                "min_e": min(e1, e2),
                "x1": f(row, "x1"),
                "y1": f(row, "y1"),
                "x2": f(row, "x2"),
                "y2": f(row, "y2"),
                "t1": f(row, "t1"),
                "t2": f(row, "t2"),
                "dt12": f(row, "dt12"),
                "size1": i(row, "size1"),
                "size2": i(row, "size2"),
                "seed1_block": i(c1, "seed_block"),
                "seed2_block": i(c2, "seed_block"),
                "seed1_row": i(c1, "seed_row"),
                "seed1_col": i(c1, "seed_col"),
                "seed2_row": i(c2, "seed_row"),
                "seed2_col": i(c2, "seed_col"),
            }
        )

    out = outdir / f"nps_production_{run}_{seg}_wf.tsv"
    write_tsv(out, rows_out, FIELDS)
    return out, {
        "run": run,
        "segment": seg,
        "input_pairs": pair_path,
        "sidecar": side_path,
        "output_tsv": out,
        "input_rows": len(pair_rows),
        "output_rows": len(rows_out),
        "missing_clusters": missing,
        "status": "ok",
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_lines = []
    pair_paths = discover_pairs(args)
    if not pair_paths:
        raise SystemExit("no production pair TSVs found")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(convert_pair_file, pair_path, args.sidecar_dir, args.output_dir)
            for pair_path in pair_paths
        ]
        for future in concurrent.futures.as_completed(futures):
            out, row = future.result()
            rows.append(row)
            if row["status"] == "ok" and int(row["output_rows"]) > 0:
                summary_lines.append(f"input_tsv: {out}")
    rows.sort(key=lambda row: (int(row["run"]), int(row["segment"])))
    summary_lines.sort()
    write_tsv(args.output_dir / "conversion_summary.tsv", rows, list(rows[0].keys()) if rows else [])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(summary_lines) + ("\n" if summary_lines else ""), encoding="ascii")
    print(args.output_dir / "conversion_summary.tsv")
    print(args.summary)
    input_rows = sum(int(row["input_rows"]) for row in rows)
    missing = sum(int(row["missing_clusters"]) for row in rows)
    missing_fraction = missing / input_rows if input_rows else 0.0
    print(
        f"pair rows={input_rows} exact joins={input_rows - missing} "
        f"missing={missing} missing_fraction={missing_fraction:.9g}"
    )
    if missing_fraction > args.max_missing_fraction:
        print(
            f"join loss exceeds --max-missing-fraction={args.max_missing_fraction:.9g}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
