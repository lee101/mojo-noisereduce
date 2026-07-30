"""NumPy-facing wrappers around the compiled Mojo kernels."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def _spectrogram(values, name: str = "magnitude") -> np.ndarray:
    values = f64(values)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional spectrogram")
    return values


def _weights(values, name: str) -> np.ndarray:
    values = f64(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return values


def triangular_weights(radius: int) -> np.ndarray:
    if radius < 1:
        return np.ones(1, dtype=np.float64)
    values = np.concatenate(
        [
            np.linspace(0, 1, radius + 1, endpoint=False),
            np.linspace(1, 0, radius + 2),
        ]
    )[1:-1]
    return f64(values / values.sum())


def stationary_statistics(
    magnitude, n_std_thresh_stationary: float = 1.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude = _spectrogram(magnitude)
    mean = np.empty(magnitude.shape[0], dtype=np.float64)
    deviation = np.empty(magnitude.shape[0], dtype=np.float64)
    threshold = np.empty(magnitude.shape[0], dtype=np.float64)
    lib().mnr_stationary_threshold(
        addr(magnitude),
        magnitude.shape[0],
        magnitude.shape[1],
        n_std_thresh_stationary,
        addr(mean),
        addr(deviation),
        addr(threshold),
    )
    return mean, deviation, threshold


def stationary_threshold(
    magnitude, n_std_thresh_stationary: float = 1.5
) -> np.ndarray:
    return stationary_statistics(
        magnitude, n_std_thresh_stationary
    )[2]


def stationary_mask(
    magnitude,
    thresholds,
    prop_decrease: float = 1.0,
    freq_weights=None,
    time_weights=None,
) -> np.ndarray:
    magnitude = _spectrogram(magnitude)
    thresholds = f64(thresholds)
    freq_weights = (
        triangular_weights(0)
        if freq_weights is None
        else _weights(freq_weights, "freq_weights")
    )
    time_weights = (
        triangular_weights(0)
        if time_weights is None
        else _weights(time_weights, "time_weights")
    )
    if magnitude.ndim != 2 or thresholds.shape != (magnitude.shape[0],):
        raise ValueError("thresholds must have one value per frequency row")
    mask = np.empty_like(magnitude)
    work = np.empty_like(magnitude)
    lib().mnr_stationary_mask(
        addr(magnitude),
        addr(thresholds),
        *magnitude.shape,
        prop_decrease,
        addr(freq_weights),
        freq_weights.size,
        addr(time_weights),
        time_weights.size,
        addr(mask),
        addr(work),
    )
    return mask


def nonstationary_mask(
    magnitude,
    samplerate: int,
    hop_length: int,
    time_constant_s: float = 2.0,
    thresh_n_mult_nonstationary: float = 2.0,
    sigmoid_slope_nonstationary: float = 10.0,
    prop_decrease: float = 1.0,
    freq_weights=None,
    time_weights=None,
) -> np.ndarray:
    magnitude = _spectrogram(magnitude)
    if samplerate <= 0 or hop_length <= 0 or time_constant_s <= 0:
        raise ValueError(
            "samplerate, hop_length, and time_constant_s must be positive"
        )
    t_frames = time_constant_s * samplerate / float(hop_length)
    coefficient = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
    freq_weights = (
        triangular_weights(0)
        if freq_weights is None
        else _weights(freq_weights, "freq_weights")
    )
    time_weights = (
        triangular_weights(0)
        if time_weights is None
        else _weights(time_weights, "time_weights")
    )
    mask = np.empty_like(magnitude)
    work = np.empty_like(magnitude)
    lib().mnr_nonstationary_mask(
        addr(magnitude),
        *magnitude.shape,
        coefficient,
        thresh_n_mult_nonstationary,
        sigmoid_slope_nonstationary,
        prop_decrease,
        addr(freq_weights),
        freq_weights.size,
        addr(time_weights),
        time_weights.size,
        addr(mask),
        addr(work),
    )
    return mask
