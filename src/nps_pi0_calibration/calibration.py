"""Load and apply one frozen cluster-energy calibration package."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path


class CalibrationError(RuntimeError):
    """Package or application input violates the calibration contract."""


@dataclass(frozen=True)
class Correction:
    run: int
    input_energy_gev: float
    seed_block: int | None
    period: str
    curve_scale: float
    run_scale: float
    seed_scale: float
    lowe_scale: float
    total_scale: float
    corrected_energy_gev: float


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CalibrationError(f"missing package file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise CalibrationError(f"empty package file: {path}")
    return rows


def _metadata(path: Path) -> dict[str, str]:
    rows = _read_tsv(path)
    try:
        return {row["key"]: row["value"] for row in rows}
    except KeyError as error:
        raise CalibrationError(f"bad metadata schema: {path}") from error


class FrozenCalibration:
    """Immutable package implementing the production cluster correction."""

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir).resolve()
        self.metadata = _metadata(self.package_dir / "package_metadata.tsv")
        self.kinematic = self.metadata.get("kinematic", "")
        self.target = self.metadata.get("target", "")
        self.baseline = self.metadata.get("baseline", "")
        self.lowe_scope = self.metadata.get("lowe_scope", "global")
        self.lowe_profile = self.metadata.get("lowe_profile", "hard_window")
        self.lowe_damping = float(self.metadata.get("lowe_damping", "0.75"))
        self.lowe_full_until = float(self.metadata.get("lowe_full_until", "0.8"))
        self.lowe_unity_at = float(self.metadata.get("lowe_unity_at", "0.8"))

        self.run_period = self._load_run_periods()
        self.curves = self._load_curves()
        self.run_scales = self._load_run_scales()
        self.seed_scales = self._load_seed_scales()
        self.lowe_min, self.lowe_max, self.lowe_scales = self._load_lowe_scales()
        self._load_optional_tilt()
        self._validate()

    @classmethod
    def load(cls, package_dir: str | Path, verify_manifest: bool = True) -> "FrozenCalibration":
        calibration = cls(package_dir)
        if verify_manifest:
            calibration.verify_manifest()
        return calibration

    def _load_run_periods(self) -> dict[int, str]:
        result: dict[int, str] = {}
        for row in _read_tsv(self.package_dir / "run_period_lut.tsv"):
            run, period = int(row["run"]), row["period"]
            if run <= 0 or not period or (run in result and result[run] != period):
                raise CalibrationError(f"invalid run-period assignment for run {run}")
            result[run] = period
        return result

    @staticmethod
    def _insert_unique(result: dict[int, float], key: int, value: float, source: str) -> None:
        if key in result:
            raise CalibrationError(f"duplicate {source} row for {key}")
        result[key] = value

    def _load_curves(self) -> dict[str, list[tuple[float, float]]]:
        result: dict[str, list[tuple[float, float]]] = {}
        for row in _read_tsv(self.package_dir / "period_photon_curve.tsv"):
            result.setdefault(row["period"], []).append(
                (float(row["energy_gev"]), float(row["log_energy_scale"]))
            )
        for points in result.values():
            points.sort()
            if len(points) < 2 or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
                raise CalibrationError("energy-curve knots must be strictly increasing")
        return result

    def _load_run_scales(self) -> dict[int, float]:
        result: dict[int, float] = {}
        for row in _read_tsv(self.package_dir / "run_scalar_lut.tsv"):
            if int(row.get("fit_ok", "1")) == 1:
                self._insert_unique(
                    result, int(row["run"]), float(row["applied_energy_scale"]), "run scale"
                )
        return result

    def _load_seed_scales(self) -> dict[int, float]:
        result: dict[int, float] = {}
        for row in _read_tsv(self.package_dir / "seed_block_scale.tsv"):
            if int(row.get("support_pass", "1")) == 1:
                self._insert_unique(
                    result,
                    int(row["seed_block"]),
                    float(row["applied_energy_scale"]),
                    "seed scale",
                )
        return result

    def _load_lowe_scales(self) -> tuple[float, float, dict[str, float]]:
        selected = []
        for row in _read_tsv(self.package_dir / "lowe_photon_scale.tsv"):
            if row.get("fold") != "full_sample":
                continue
            if abs(float(row["damping"]) - self.lowe_damping) < 1.0e-12:
                selected.append(row)
        if not selected:
            raise CalibrationError(f"no full-sample low-E row for damping {self.lowe_damping}")
        bounds = {(float(row["emin"]), float(row["emax"])) for row in selected}
        if len(bounds) != 1:
            raise CalibrationError("low-E rows have inconsistent energy bounds")
        scales: dict[str, float] = {}
        for row in selected:
            key = row.get("period", "") if self.lowe_scope == "period" else "global"
            if not key or key in scales:
                raise CalibrationError(f"duplicate or empty low-E scale key: {key!r}")
            scales[key] = float(row["applied_photon_energy_scale"])
        emin, emax = bounds.pop()
        return emin, emax, scales

    def _load_optional_tilt(self) -> None:
        self.tilt_coefficient = 0.0
        self.tilt_min = 0.6
        self.tilt_pivot = 0.7
        self.tilt_linear_end = 0.8
        self.tilt_unity_at = 0.9
        path = self.package_dir / "lowe_shape_tilt.tsv"
        if not path.is_file():
            return
        rows = _read_tsv(path)
        if len(rows) != 1 or rows[0].get("fold") != "full_sample":
            raise CalibrationError("low-E shape tilt requires one full_sample row")
        row = rows[0]
        self.tilt_coefficient = float(row["applied_log_scale_coefficient"])
        self.tilt_min = float(row.get("energy_min_gev", "0.6"))
        self.tilt_pivot = float(row.get("pivot_energy_gev", "0.7"))
        self.tilt_linear_end = float(row.get("linear_end_gev", "0.8"))
        self.tilt_unity_at = float(row.get("unity_at_gev", "0.9"))

    def _validate(self) -> None:
        try:
            schema_version = int(self.metadata["schema_version"])
        except (KeyError, ValueError) as error:
            raise CalibrationError("missing or invalid package schema_version") from error
        if schema_version != 3:
            raise CalibrationError(f"unsupported package schema_version: {schema_version}")
        if not self.kinematic or not self.target or not self.run_period:
            raise CalibrationError("package identity or run coverage missing")
        if self.baseline != "hao_pi0_db":
            raise CalibrationError(
                f"unsupported baseline {self.baseline!r}; input must use Hao Pi0 DB energy"
            )
        if set(self.run_period) != set(self.run_scales):
            missing = sorted(set(self.run_period) - set(self.run_scales))
            extra = sorted(set(self.run_scales) - set(self.run_period))
            raise CalibrationError(f"run-scale coverage mismatch: missing={missing[:8]} extra={extra[:8]}")
        periods = set(self.run_period.values())
        if missing := periods - set(self.curves):
            raise CalibrationError(f"periods lack energy curves: {sorted(missing)}")
        if self.lowe_scope == "global":
            expected = {"global"}
        elif self.lowe_scope == "period":
            expected = periods
        else:
            raise CalibrationError(f"unsupported low-E scope: {self.lowe_scope}")
        if set(self.lowe_scales) != expected:
            raise CalibrationError("low-E scale coverage does not match configured scope")
        if self.lowe_profile not in {"hard_window", "smoothstep"}:
            raise CalibrationError(f"unsupported low-E profile: {self.lowe_profile}")
        if self.lowe_profile == "smoothstep" and not (
            self.lowe_min <= self.lowe_full_until < self.lowe_unity_at <= self.lowe_max
        ):
            raise CalibrationError("smooth low-E profile has invalid energy ordering")
        values = list(self.run_scales.values()) + list(self.seed_scales.values()) + list(self.lowe_scales.values())
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise CalibrationError("non-finite or non-positive package scale")
        if not (
            self.tilt_min < self.tilt_pivot < self.tilt_linear_end < self.tilt_unity_at
        ):
            raise CalibrationError("invalid low-E shape-tilt energy ordering")

    def verify_manifest(self) -> None:
        path = self.package_dir / "package_manifest.tsv"
        if not path.is_file():
            return
        for row in _read_tsv(path):
            target = self.package_dir / row["file"]
            if target.name == path.name:
                continue
            if not target.is_file():
                raise CalibrationError(f"manifest file missing: {target}")
            if "bytes" in row and row["bytes"] and target.stat().st_size != int(row["bytes"]):
                raise CalibrationError(f"manifest size mismatch: {target}")
            if "sha256" in row and row["sha256"]:
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if digest != row["sha256"]:
                    raise CalibrationError(f"manifest hash mismatch: {target}")

    def _curve_scale(self, period: str, energy_gev: float) -> float:
        points = self.curves[period]
        if energy_gev <= points[0][0]:
            return math.exp(points[0][1])
        if energy_gev >= points[-1][0]:
            return math.exp(points[-1][1])
        for left, right in zip(points, points[1:]):
            if energy_gev <= right[0]:
                fraction = (energy_gev - left[0]) / (right[0] - left[0])
                return math.exp((1.0 - fraction) * left[1] + fraction * right[1])
        raise AssertionError("curve interval lookup failed")

    def _tilt_basis(self, energy_gev: float) -> float:
        if energy_gev < self.tilt_min or energy_gev >= self.tilt_unity_at:
            return 0.0
        if energy_gev <= self.tilt_linear_end:
            return (self.tilt_pivot - energy_gev) / (self.tilt_pivot - self.tilt_min)
        width = self.tilt_unity_at - self.tilt_linear_end
        t = (energy_gev - self.tilt_linear_end) / width
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        start_value = (self.tilt_pivot - self.tilt_linear_end) / (
            self.tilt_pivot - self.tilt_min
        )
        start_derivative = -1.0 / (self.tilt_pivot - self.tilt_min)
        return h00 * start_value + h10 * width * start_derivative

    def _lowe_scale(self, period: str, energy_gev: float) -> float:
        profile_scale = 1.0
        if self.lowe_min <= energy_gev < self.lowe_max:
            amplitude = self.lowe_scales[period if self.lowe_scope == "period" else "global"]
            if self.lowe_profile == "hard_window":
                profile_scale = amplitude
            elif energy_gev <= self.lowe_full_until:
                profile_scale = amplitude
            elif energy_gev < self.lowe_unity_at:
                t = (energy_gev - self.lowe_full_until) / (
                    self.lowe_unity_at - self.lowe_full_until
                )
                weight = 1.0 - (3.0 * t**2 - 2.0 * t**3)
                profile_scale = amplitude**weight
        return profile_scale * math.exp(self.tilt_coefficient * self._tilt_basis(energy_gev))

    def correct(
        self,
        run: int,
        energy_gev: float,
        seed_block: int | None,
        *,
        missing_seed: str = "identity",
    ) -> Correction:
        """Correct one cluster already calibrated with Hao's Pi0 DB coefficient."""
        if run not in self.run_period:
            raise CalibrationError(f"run {run} is not covered by package")
        if not math.isfinite(energy_gev) or energy_gev < 0.0:
            raise CalibrationError(f"invalid cluster energy: {energy_gev}")
        if missing_seed not in {"identity", "error"}:
            raise CalibrationError(f"invalid missing-seed policy: {missing_seed}")
        period = self.run_period[run]
        curve = self._curve_scale(period, energy_gev)
        run_scale = self.run_scales[run]
        if seed_block is None or seed_block not in self.seed_scales:
            if missing_seed == "error":
                raise CalibrationError(f"seed block {seed_block} lacks supported correction")
            seed = 1.0
        else:
            seed = self.seed_scales[seed_block]
        lowe = self._lowe_scale(period, energy_gev)
        total = curve * run_scale * seed * lowe
        if not math.isfinite(total) or total <= 0.0:
            raise CalibrationError("computed non-finite or non-positive total scale")
        return Correction(
            run=run,
            input_energy_gev=energy_gev,
            seed_block=seed_block,
            period=period,
            curve_scale=curve,
            run_scale=run_scale,
            seed_scale=seed,
            lowe_scale=lowe,
            total_scale=total,
            corrected_energy_gev=energy_gev * total,
        )
