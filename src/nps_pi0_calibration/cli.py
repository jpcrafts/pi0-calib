"""Command-line application for tabular cluster energies."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .calibration import FrozenCalibration


OUTPUT_FIELDS = [
    "calibration_period",
    "curve_scale",
    "run_scale",
    "seed_scale",
    "lowe_scale",
    "total_scale",
    "corrected_energy_gev",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen NPS Pi0 calibration to TSV/CSV cluster rows."
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--input", required=True, help="Input TSV/CSV path, or - for stdin")
    parser.add_argument("--output", required=True, help="Output path, or - for stdout")
    parser.add_argument("--delimiter", choices=("tab", "comma"))
    parser.add_argument("--run-field", default="run")
    parser.add_argument("--energy-field", default="energy_gev")
    parser.add_argument("--seed-field", default="seed_block")
    parser.add_argument("--missing-seed", choices=("identity", "error"), default="identity")
    parser.add_argument("--no-verify-manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unvalidated", action="store_true",
        help="Allow a diagnostic package (validation tests only).",
    )
    return parser.parse_args(argv)


def _delimiter(args: argparse.Namespace) -> str:
    if args.delimiter:
        return "\t" if args.delimiter == "tab" else ","
    return "," if args.input != "-" and Path(args.input).suffix.lower() == ".csv" else "\t"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    calibration = FrozenCalibration.load(args.package, verify_manifest=not args.no_verify_manifest)
    approval = calibration.metadata.get("approval_status", "unreviewed")
    if approval != "validated" and not args.allow_unvalidated:
        raise SystemExit(
            f"package approval_status is {approval!r}; pass --allow-unvalidated only for diagnostics"
        )
    delimiter = _delimiter(args)
    if args.output != "-":
        output_path = Path(args.output)
        if args.input != "-" and output_path.resolve() == Path(args.input).resolve():
            raise SystemExit("refusing to overwrite input file")
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite to replace it: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    input_handle = sys.stdin if args.input == "-" else Path(args.input).open(newline="", encoding="utf-8")
    output_handle = sys.stdout if args.output == "-" else Path(args.output).open("w", newline="", encoding="utf-8")
    try:
        reader = csv.DictReader(input_handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("input lacks header")
        if overlap := set(reader.fieldnames) & set(OUTPUT_FIELDS):
            raise ValueError(f"input already contains calibration output fields: {sorted(overlap)}")
        required = {args.run_field, args.energy_field, args.seed_field}
        if missing := required - set(reader.fieldnames):
            raise ValueError(f"input lacks fields: {sorted(missing)}")
        writer = csv.DictWriter(
            output_handle,
            fieldnames=[*reader.fieldnames, *OUTPUT_FIELDS],
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        for row_number, row in enumerate(reader, start=2):
            try:
                seed_text = row[args.seed_field].strip()
                correction = calibration.correct(
                    int(float(row[args.run_field])),
                    float(row[args.energy_field]),
                    int(float(seed_text)) if seed_text else None,
                    missing_seed=args.missing_seed,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid input row {row_number}: {error}") from error
            row.update(
                {
                    "calibration_period": correction.period,
                    "curve_scale": f"{correction.curve_scale:.16g}",
                    "run_scale": f"{correction.run_scale:.16g}",
                    "seed_scale": f"{correction.seed_scale:.16g}",
                    "lowe_scale": f"{correction.lowe_scale:.16g}",
                    "total_scale": f"{correction.total_scale:.16g}",
                    "corrected_energy_gev": f"{correction.corrected_energy_gev:.16g}",
                }
            )
            writer.writerow(row)
    finally:
        if input_handle is not sys.stdin:
            input_handle.close()
        if output_handle is not sys.stdout:
            output_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
