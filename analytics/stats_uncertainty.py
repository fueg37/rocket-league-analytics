"""Uncertainty estimation utilities with deterministic reproducibility defaults."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

DEFAULT_UNCERTAINTY_SEED = 20250214


def deterministic_seed(*parts: object, base_seed: int = DEFAULT_UNCERTAINTY_SEED) -> int:
    """Derive a stable integer seed from semantic inputs."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) ^ int(base_seed)) % (2**32 - 1)


def reliability_from_sample_size(sample_size: int) -> str:
    if sample_size >= 30:
        return "high"
    if sample_size >= 10:
        return "medium"
    return "low"


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    iterations: int = 1500,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean interval for continuous metrics."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.default_rng(DEFAULT_UNCERTAINTY_SEED if seed is None else seed)
    sample_ix = rng.integers(0, arr.size, size=(iterations, arr.size))
    sample_means = arr[sample_ix].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(sample_means, alpha))
    hi = float(np.quantile(sample_means, 1.0 - alpha))
    return float(arr.mean()), lo, hi


def bayesian_binomial_interval(
    successes: int,
    trials: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    draws: int = 5000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Beta posterior credible interval for rates/probabilities."""
    if trials <= 0:
        return 0.0, 0.0, 0.0
    alpha_post = prior_alpha + max(0, int(successes))
    beta_post = prior_beta + max(0, int(trials - successes))
    rng = np.random.default_rng(DEFAULT_UNCERTAINTY_SEED if seed is None else seed)
    draws_arr = rng.beta(alpha_post, beta_post, size=max(500, int(draws)))
    alpha = (1.0 - confidence) / 2.0
    mean = float(alpha_post / (alpha_post + beta_post))
    lo = float(np.quantile(draws_arr, alpha))
    hi = float(np.quantile(draws_arr, 1.0 - alpha))
    return mean, lo, hi
