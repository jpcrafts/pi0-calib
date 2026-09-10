#!/usr/bin/env python3
"""Apply a frozen package to all clusters in an NPS t_prod ROOT tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nps_pi0_calibration import FrozenCalibration  # noqa: E402


def quote_root(value: Path) -> str:
    return str(value.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--segment", type=int, default=-1)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument(
        "--missing-seed",
        choices=("error", "identity", "centroid"),
        default="error",
        help="Policy when exact sidecar seed is unavailable",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tree-only", action="store_true",
        help="Write only t_prod and calibration metadata (smoke tests only).",
    )
    parser.add_argument(
        "--allow-unvalidated", action="store_true",
        help="Allow a diagnostic package (validation tests only).",
    )
    args = parser.parse_args()

    for path in (args.input_root, args.package):
        if not path.exists():
            parser.error(f"not found: {path}")
    if args.sidecar is not None and not args.sidecar.is_file():
        parser.error(f"not found: {args.sidecar}")
    if args.missing_seed == "error" and args.sidecar is None:
        parser.error("--missing-seed error requires --sidecar")
    if args.input_root.resolve() == args.output_root.resolve():
        parser.error("input and output ROOT paths must differ")
    if args.output_root.exists() and not args.overwrite:
        parser.error(f"output exists; pass --overwrite to replace it: {args.output_root}")
    if args.output_root.exists():
        args.output_root.unlink()
    calibration = FrozenCalibration.load(args.package)
    approval = calibration.metadata.get("approval_status", "unreviewed")
    if approval != "validated" and not args.allow_unvalidated:
        parser.error(
            f"package approval_status is {approval!r}; use only after review, or pass "
            "--allow-unvalidated for a diagnostic test"
        )
    if args.run not in calibration.run_period:
        parser.error(f"run {args.run} is not covered by package")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)

    macro = Path(__file__).resolve().with_name("write_nps_corrected_cluster_tree.C")
    require_exact = args.missing_seed == "error"
    identity_missing = args.missing_seed == "identity"
    sidecar = quote_root(args.sidecar) if args.sidecar else ""
    call = (
        f'{quote_root(macro)}("{quote_root(args.input_root)}",'
        f'"{quote_root(args.package)}","{quote_root(args.output_root)}",'
        f'{args.run},{args.segment},{args.max_events},"{sidecar}",'
        f'{str(require_exact).lower()},{str(identity_missing).lower()},'
        f'{str(not args.tree_only).lower()},'
        f'{str(args.allow_unvalidated).lower()})'
    )
    result = subprocess.run(["root", "-l", "-b", "-q", call], check=False)
    if result.returncode != 0 or not args.output_root.is_file():
        raise SystemExit(result.returncode or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
