from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from nps_pi0_calibration import CalibrationError, FrozenCalibration


def write_tsv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


class FrozenCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        package = Path(self.temp.name)
        write_tsv(
            package / "package_metadata.tsv",
            ["key", "value"],
            [
                ["schema_version", 3],
                ["kinematic", "test"],
                ["target", "LH2"],
                ["baseline", "hao_pi0_db"],
                ["lowe_damping", 0.75],
                ["lowe_scope", "global"],
                ["lowe_profile", "smoothstep"],
                ["lowe_full_until", 0.8],
                ["lowe_unity_at", 0.9],
            ],
        )
        write_tsv(package / "run_period_lut.tsv", ["run", "period"], [[100, "p1"]])
        write_tsv(
            package / "period_photon_curve.tsv",
            ["period", "energy_gev", "log_energy_scale"],
            [["p1", 0.6, 0.0], ["p1", 1.0, math.log(1.02)]],
        )
        write_tsv(
            package / "run_scalar_lut.tsv",
            ["run", "fit_ok", "applied_energy_scale"],
            [[100, 1, 1.01]],
        )
        write_tsv(
            package / "seed_block_scale.tsv",
            ["seed_block", "support_pass", "applied_energy_scale"],
            [[42, 1, 0.99]],
        )
        write_tsv(
            package / "lowe_photon_scale.tsv",
            ["fold", "emin", "emax", "damping", "applied_photon_energy_scale"],
            [["full_sample", 0.6, 0.9, 0.75, 1.04]],
        )
        self.calibration = FrozenCalibration.load(package)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_factorization_and_log_interpolation(self) -> None:
        result = self.calibration.correct(100, 0.8, 42)
        expected_curve = math.sqrt(1.02)
        self.assertAlmostEqual(result.curve_scale, expected_curve)
        self.assertAlmostEqual(result.lowe_scale, 1.04)
        self.assertAlmostEqual(result.total_scale, expected_curve * 1.01 * 0.99 * 1.04)
        self.assertAlmostEqual(result.corrected_energy_gev, 0.8 * result.total_scale)

    def test_smooth_transition(self) -> None:
        result = self.calibration.correct(100, 0.85, 42)
        self.assertAlmostEqual(result.lowe_scale, math.sqrt(1.04))

    def test_upper_guard_is_exact_identity_without_curve_evaluation(self) -> None:
        from unittest.mock import patch
        with patch.object(self.calibration, "_curve_scale", side_effect=AssertionError("extrapolation")):
            for energy in (2.5, math.nextafter(2.5, math.inf), 3.0, 4.0, 6.0, 100.0):
                result = self.calibration.correct(100, energy, 42)
                self.assertEqual(result.corrected_energy_gev, energy)
                for scale in (result.curve_scale, result.run_scale, result.seed_scale,
                              result.lowe_scale, result.total_scale):
                    self.assertEqual(scale, 1.0)

    def test_upper_taper_covers_run_and_seed_components(self) -> None:
        original = (1.02, 1.01, 0.99, 1.0)
        for energy in (1.5, 2.0, 2.1, 2.25, 2.4, 2.49):
            result = self.calibration.correct(100, energy, 42)
            t = max(0.0, (energy - 2.0) / 0.5)
            weight = 1.0 - 3.0 * t**2 + 2.0 * t**3
            applied = (result.curve_scale, result.run_scale, result.seed_scale, result.lowe_scale)
            for actual, raw in zip(applied, original):
                self.assertAlmostEqual(actual, raw**weight, places=14)
            self.assertAlmostEqual(result.total_scale, math.prod(original)**weight, places=14)
            self.assertEqual(result.total_scale, math.prod(applied))

    def test_upper_taper_boundary_continuity(self) -> None:
        for boundary in (2.0, 2.5):
            left = self.calibration.correct(100, boundary - 1e-8, 42).total_scale
            right = self.calibration.correct(100, boundary + 1e-8, 42).total_scale
            self.assertAlmostEqual(left, right, places=12)

    def test_invalid_energy_still_fails(self) -> None:
        for energy in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(CalibrationError):
                self.calibration.correct(100, energy, 42)

    def test_missing_seed_identity_or_error(self) -> None:
        self.assertEqual(self.calibration.correct(100, 1.0, None).seed_scale, 1.0)
        with self.assertRaises(CalibrationError):
            self.calibration.correct(100, 1.0, None, missing_seed="error")

    def test_uncovered_run_fails(self) -> None:
        with self.assertRaises(CalibrationError):
            self.calibration.correct(101, 1.0, 42)

    def test_duplicate_supported_seed_fails(self) -> None:
        package = Path(self.temp.name)
        with (package / "seed_block_scale.tsv").open("a", encoding="ascii") as handle:
            handle.write("42\t1\t1.01\n")
        with self.assertRaises(CalibrationError):
            FrozenCalibration.load(package)


if __name__ == "__main__":
    unittest.main()
