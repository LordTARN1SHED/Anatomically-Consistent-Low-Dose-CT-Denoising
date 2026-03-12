from __future__ import annotations

import numpy as np


def add_signal_dependent_gaussian_noise(
    x_norm: np.ndarray,
    sigma_min: float = 0.01,
    sigma_max: float = 0.06,
    alpha: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Signal-dependent Gaussian noise on normalized [-1, 1] image."""
    if rng is None:
        rng = np.random.default_rng()

    x01 = np.clip((x_norm + 1.0) / 2.0, 0.0, 1.0)
    sigma0 = rng.uniform(sigma_min, sigma_max)
    sigma = sigma0 + alpha * np.sqrt(np.maximum(x01, 0.0))
    noisy = x01 + rng.normal(0.0, sigma, size=x01.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 2.0 - 1.0).astype(np.float32)


def add_poisson_noise_image_domain(
    x_norm: np.ndarray,
    peak: float = 30.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Poisson proxy in image domain when sinogram is unavailable."""
    if rng is None:
        rng = np.random.default_rng()

    x01 = np.clip((x_norm + 1.0) / 2.0, 1e-6, 1.0)
    noisy = rng.poisson(x01 * peak) / peak
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 2.0 - 1.0).astype(np.float32)
