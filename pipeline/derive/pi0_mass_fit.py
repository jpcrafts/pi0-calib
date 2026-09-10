#!/usr/bin/env python3
"""Canonical pi0 invariant-mass fit and explicit quality contract.

This module deliberately has no arithmetic-moment fallback.  A failed fit is a
failed fit; callers must add statistics (normally chronological support runs)
and retry the same model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit


M_PI0_MEV = 134.9766


@dataclass(frozen=True)
class FitConfig:
    window_lo: float = 118.0
    window_hi: float = 150.0
    bin_width: float = 1.0
    min_entries: int = 500
    mu_min: float = 125.0
    mu_max: float = 142.0
    sigma_fit_min: float = 1.0
    sigma_fit_max: float = 12.0
    sigma_accept_min: float = 2.0
    sigma_accept_max: float = 10.0
    max_mu_err: float = 0.50
    max_chi2_ndf: float = math.inf
    warn_chi2_ndf: float = 3.0
    min_amp_significance: float = 3.0
    max_window_shift: float = 0.50


DEFAULT_CONFIG = FitConfig()
STABILITY_WINDOWS = ((116.0, 150.0), (118.0, 152.0))


def gaussian_linear_endpoints(
    x: np.ndarray,
    amp: float,
    mu: float,
    sigma: float,
    background_lo: float,
    background_hi: float,
    window_lo: float,
    window_hi: float,
) -> np.ndarray:
    fraction = (x - window_lo) / (window_hi - window_lo)
    background = background_lo + fraction * (background_hi - background_lo)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + background


def evaluate_model(x: np.ndarray, result: dict[str, object]) -> np.ndarray:
    return gaussian_linear_endpoints(
        np.asarray(x, dtype=float),
        float(result["amp"]),
        float(result["mu_mev"]),
        float(result["sigma_mev"]),
        float(result["background_lo"]),
        float(result["background_hi"]),
        float(result["window_lo"]),
        float(result["window_hi"]),
    )


def _empty_result(config: FitConfig, entries: int, reason: str) -> dict[str, object]:
    return {
        "entries": entries,
        "window_entries": entries,
        "window_lo": config.window_lo,
        "window_hi": config.window_hi,
        "bin_width": config.bin_width,
        "optimizer_ok": 0,
        "covariance_ok": 0,
        "quality_ok": 0,
        "quality_reasons": reason,
        "quality_warnings": "none",
        "mu_mev": math.nan,
        "mu_residual_mev": math.nan,
        "mu_err_mev": math.nan,
        "sigma_mev": math.nan,
        "sigma_err_mev": math.nan,
        "amp": math.nan,
        "amp_err": math.nan,
        "amp_significance": math.nan,
        "background_lo": math.nan,
        "background_hi": math.nan,
        "chi2": math.nan,
        "ndf": 0,
        "chi2_ndf": math.nan,
        "max_window_shift_mev": math.nan,
        "variant_116_150_mu_mev": math.nan,
        "variant_118_152_mu_mev": math.nan,
    }


def _fit_once(values: np.ndarray, config: FitConfig) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    work = finite[(finite >= config.window_lo) & (finite <= config.window_hi)]
    if work.size < 50:
        return _empty_result(config, int(work.size), "too_few_for_optimizer")

    bins = max(16, int(round((config.window_hi - config.window_lo) / config.bin_width)))
    hist, edges = np.histogram(work, bins=bins, range=(config.window_lo, config.window_hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    edge_count = max(3, bins // 8)
    left0 = max(0.0, float(np.median(hist[:edge_count])))
    right0 = max(0.0, float(np.median(hist[-edge_count:])))
    peak = int(np.argmax(hist))
    background_at_peak = left0 + (right0 - left0) * (peak / max(bins - 1, 1))
    amp0 = max(1.0, float(hist[peak]) - background_at_peak)
    mu0 = float(np.clip(centers[peak], config.mu_min, config.mu_max))

    def model(x: np.ndarray, amp: float, mu: float, sigma: float, bg_lo: float, bg_hi: float) -> np.ndarray:
        return gaussian_linear_endpoints(
            x, amp, mu, sigma, bg_lo, bg_hi, config.window_lo, config.window_hi
        )

    try:
        popt, pcov = curve_fit(
            model,
            centers,
            hist.astype(float),
            p0=[amp0, mu0, 4.5, left0, right0],
            sigma=np.sqrt(np.maximum(hist.astype(float), 1.0)),
            absolute_sigma=True,
            bounds=(
                [0.0, config.mu_min, config.sigma_fit_min, 0.0, 0.0],
                [np.inf, config.mu_max, config.sigma_fit_max, np.inf, np.inf],
            ),
            maxfev=30000,
        )
    except Exception as exc:
        return _empty_result(config, int(work.size), f"optimizer_failed:{type(exc).__name__}")

    amp, mu, sigma, bg_lo, bg_hi = [float(x) for x in popt]
    covariance_ok = bool(np.all(np.isfinite(pcov)) and np.all(np.diag(pcov) >= 0.0))
    errors = np.sqrt(np.diag(pcov)) if covariance_ok else np.full(5, math.nan)
    expected = model(centers, *popt)
    chi2 = float(np.sum((hist - expected) ** 2 / np.maximum(expected, 1.0)))
    ndf = max(0, bins - len(popt))
    chi2_ndf = chi2 / ndf if ndf > 0 else math.nan
    amp_err = float(errors[0])
    amp_significance = amp / amp_err if math.isfinite(amp_err) and amp_err > 0 else math.nan

    return {
        "entries": int(finite.size),
        "window_entries": int(work.size),
        "window_lo": config.window_lo,
        "window_hi": config.window_hi,
        "bin_width": (config.window_hi - config.window_lo) / bins,
        "optimizer_ok": 1,
        "covariance_ok": int(covariance_ok),
        "quality_ok": 0,
        "quality_reasons": "not_evaluated",
        "mu_mev": mu,
        "mu_residual_mev": mu - M_PI0_MEV,
        "mu_err_mev": float(errors[1]),
        "sigma_mev": sigma,
        "sigma_err_mev": float(errors[2]),
        "amp": amp,
        "amp_err": amp_err,
        "amp_significance": amp_significance,
        "background_lo": bg_lo,
        "background_hi": bg_hi,
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2_ndf,
        "max_window_shift_mev": math.nan,
        "variant_116_150_mu_mev": math.nan,
        "variant_118_152_mu_mev": math.nan,
    }


def _quality_reasons(result: dict[str, object], config: FitConfig) -> list[str]:
    reasons: list[str] = []
    if int(result["window_entries"]) < config.min_entries:
        reasons.append("entries")
    if not int(result["optimizer_ok"]):
        reasons.append("optimizer")
        return reasons
    if not int(result["covariance_ok"]):
        reasons.append("covariance")
    mu = float(result["mu_mev"])
    sigma = float(result["sigma_mev"])
    mu_err = float(result["mu_err_mev"])
    chi2_ndf = float(result["chi2_ndf"])
    significance = float(result["amp_significance"])
    shift = float(result["max_window_shift_mev"])
    if not (config.mu_min < mu < config.mu_max):
        reasons.append("mu_bound")
    if not (config.sigma_accept_min <= sigma <= config.sigma_accept_max):
        reasons.append("sigma")
    if not math.isfinite(mu_err) or mu_err > config.max_mu_err:
        reasons.append("mu_error")
    if not math.isfinite(chi2_ndf) or chi2_ndf > config.max_chi2_ndf:
        reasons.append("chi2")
    if not math.isfinite(significance) or significance < config.min_amp_significance:
        reasons.append("significance")
    if not math.isfinite(shift) or shift > config.max_window_shift:
        reasons.append("window_stability")
    return reasons


def _quality_warnings(result: dict[str, object], config: FitConfig) -> list[str]:
    warnings: list[str] = []
    chi2_ndf = float(result["chi2_ndf"])
    if math.isfinite(chi2_ndf) and chi2_ndf > config.warn_chi2_ndf:
        warnings.append("chi2")
    return warnings


def fit_pi0_mass(
    values: Iterable[float] | np.ndarray,
    config: FitConfig = DEFAULT_CONFIG,
    stability_windows: tuple[tuple[float, float], ...] = STABILITY_WINDOWS,
) -> dict[str, object]:
    """Fit one spectrum, test nearby windows, and apply quality gates."""
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    primary = _fit_once(array, config)
    variant_mus: list[float] = []
    variant_keys = ["variant_116_150_mu_mev", "variant_118_152_mu_mev"]
    for key, (lo, hi) in zip(variant_keys, stability_windows):
        variant = _fit_once(array, replace(config, window_lo=lo, window_hi=hi, min_entries=50))
        mu = float(variant["mu_mev"])
        primary[key] = mu
        if int(variant["optimizer_ok"]) and math.isfinite(mu):
            variant_mus.append(mu)
    primary_mu = float(primary["mu_mev"])
    primary["max_window_shift_mev"] = (
        max(abs(mu - primary_mu) for mu in variant_mus)
        if int(primary["optimizer_ok"]) and len(variant_mus) == len(stability_windows)
        else math.nan
    )
    reasons = _quality_reasons(primary, config)
    warnings = _quality_warnings(primary, config)
    primary["quality_ok"] = int(not reasons)
    primary["quality_reasons"] = "pass" if not reasons else ",".join(reasons)
    primary["quality_warnings"] = "none" if not warnings else ",".join(warnings)
    return primary
