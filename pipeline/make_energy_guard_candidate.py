#!/usr/bin/env python3
"""Copy frozen numerical tables into an unreviewed upper-energy-guard candidate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from promote_package import read_metadata, set_metadata, write_manifest, write_metadata
from nps_pi0_calibration import FrozenCalibration
from nps_pi0_calibration.calibration import ENERGY_POLICY


def make_candidate(source: Path, output: Path) -> None:
    FrozenCalibration.load(source)
    # Refuse existing destinations; source and published packages stay immutable.
    output.mkdir(parents=True, exist_ok=False)
    for name in ("period_photon_curve.tsv", "run_period_lut.tsv", "run_scalar_lut.tsv",
                 "seed_block_scale.tsv", "lowe_photon_scale.tsv", "lowe_smooth_profile.tsv",
                 "lowe_shape_tilt.tsv", "kinematic_config.yaml"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    metadata = [row for row in read_metadata(source / "package_metadata.tsv")
                if not (row["key"].startswith("approval_") or row["key"].startswith("approved_")
                        or row["key"] == "validation_report")]
    for key, value in {
        "approval_status": "diagnostic_unreviewed",
        "source_package": str(source.resolve()),
        "energy_application_policy": ENERGY_POLICY,
        "energy_full_until_gev": "2.0",
        "energy_unity_at_gev": "2.5",
    }.items():
        set_metadata(metadata, key, value)
    write_metadata(output / "package_metadata.tsv", metadata)
    (output / "README.txt").write_text(
        "UNREVIEWED upper-energy-guard candidate, not approved for production.\n"
        "Numerical tables unchanged; application behavior changes.\n"
        "Original Hao cluster E <= 2 GeV: original total correction.\n"
        "2 < E < 2.5 GeV: each component raised to w=1-3t^2+2t^3, t=(E-2)/0.5.\n"
        "E >= 2.5 GeV: original energy copied exactly; all component scales unity.\n"
        "No event/Pi0 gate. Mixed-energy events use each cluster's own energy.\n"
        "Requires new mass-fit, energy-bin, and missing-mass validation before promotion.\n",
        encoding="ascii",
    )
    write_manifest(output)
    FrozenCalibration.load(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--output-package", required=True, type=Path)
    args = parser.parse_args()
    make_candidate(args.source_package, args.output_package)
    print(args.output_package)


if __name__ == "__main__":
    main()
