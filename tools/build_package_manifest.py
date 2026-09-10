#!/usr/bin/env python3
"""Write deterministic size/hash metadata for a frozen package."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    if not package.is_dir():
        parser.error(f"not a directory: {package}")
    output = package / "package_manifest.tsv"
    files = sorted(
        path for path in package.iterdir() if path.is_file() and path != output
    )
    with output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("file", "bytes", "sha256"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
