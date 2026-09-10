from __future__ import annotations

import unittest
from pathlib import Path

from nps_pi0_calibration import FrozenCalibration


PACKAGE = Path(__file__).resolve().parents[1] / "packages" / "x60_4_lh2_v5_smooth_lowe"


class ReleasePackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = FrozenCalibration.load(PACKAGE)

    def test_identity_matches_root_reference_vector(self) -> None:
        # Generated independently by the production ROOT writer on run 4253.
        result = self.calibration.correct(4253, 0.560775101184845, 608)
        self.assertAlmostEqual(result.curve_scale, 1.044112837538297, places=14)
        self.assertAlmostEqual(result.run_scale, 0.9998358800010637, places=14)
        self.assertAlmostEqual(result.seed_scale, 1.002300463855615, places=14)
        self.assertAlmostEqual(result.lowe_scale, 1.0, places=14)
        self.assertAlmostEqual(result.corrected_energy_gev, 0.5867631170516354, places=14)

    def test_smooth_lowe_is_continuous_at_boundaries(self) -> None:
        epsilon = 1.0e-10
        for boundary in (0.8, 0.9):
            left = self.calibration.correct(4253, boundary - epsilon, 608).lowe_scale
            right = self.calibration.correct(4253, boundary + epsilon, 608).lowe_scale
            self.assertAlmostEqual(left, right, places=8)


if __name__ == "__main__":
    unittest.main()
