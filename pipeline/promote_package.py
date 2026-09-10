#!/usr/bin/env python3
"""Copy a reviewed diagnostic candidate into an immutable validated package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from nps_pi0_calibration import FrozenCalibration  # noqa: E402


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("key", "value"), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def set_metadata(rows: list[dict[str, str]], key: str, value: str) -> None:
    for row in rows:
        if row["key"] == key:
            row["value"] = value
            return
    rows.append({"key": key, "value": value})


def write_manifest(package: Path) -> None:
    output = package / "package_manifest.tsv"
    files = sorted(path for path in package.iterdir() if path.is_file() and path != output)
    with output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("file", "bytes", "sha256"),
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for path in files:
            writer.writerow({
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if candidate == output:
        parser.error("candidate and output must differ")
    if not args.approved_by.strip() or not args.note.strip():
        parser.error("approval identity and note must be non-empty")
    calibration = FrozenCalibration.load(candidate)
    if calibration.metadata.get("approval_status", "unreviewed") == "validated":
        parser.error("candidate is already validated")
    if output.exists():
        parser.error(f"refusing to overwrite release package: {output}")
    shutil.copytree(candidate, output)
    metadata_path = output / "package_metadata.tsv"
    metadata = read_metadata(metadata_path)
    set_metadata(metadata, "approval_status", "validated")
    set_metadata(metadata, "approved_by", args.approved_by.strip())
    set_metadata(metadata, "approval_note", args.note.strip())
    set_metadata(metadata, "approval_utc", datetime.now(timezone.utc).isoformat())
    set_metadata(metadata, "candidate_source", str(candidate))
    write_metadata(metadata_path, metadata)
    write_manifest(output)
    approved = FrozenCalibration.load(output)
    if approved.metadata.get("approval_status") != "validated":
        raise RuntimeError("promoted package failed approval validation")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
