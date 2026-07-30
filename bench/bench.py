"""Benchmarks against noisereduce 3.0.3 on identical inputs."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
from scipy.signal import fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_noisereduce as mnr  # noqa: E402
from mojo_noisereduce._kernels import (  # noqa: E402
    nonstationary_mask,
    stationary_mask,
    stationary_statistics,
    triangular_weights,
)
import noisereduce as upstream  # noqa: E402
from noisereduce.spectralgate.nonstationary import (  # noqa: E402
    get_time_smoothed_representation,
)
from noisereduce.spectralgate.utils import _amp_to_db, sigmoid  # noqa: E402


def time_best(function, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(0)
    magnitude = np.abs(rng.normal(size=(513, 4_000))) + 1e-8
    freq = triangular_weights(11)
    time_weights = triangular_weights(4)
    smoothing = np.outer(freq, time_weights)
    db = _amp_to_db(magnitude)
    threshold = db.mean(axis=1) + 1.5 * db.std(axis=1)

    def numpy_statistics():
        values = _amp_to_db(magnitude)
        mean = values.mean(axis=1)
        deviation = values.std(axis=1)
        return (
            mean,
            deviation,
            mean + 1.5 * deviation,
        )

    def numpy_stationary():
        raw = (_amp_to_db(magnitude) > threshold[:, None]) * 0.8 + 0.2
        return fftconvolve(raw, smoothing, mode="same")

    def numpy_nonstationary():
        smooth = get_time_smoothed_representation(
            magnitude, 22_050, 256, 2.0
        )
        above = (magnitude - smooth) / smooth
        mask = sigmoid(above, -2.0, 10.0)
        return fftconvolve(mask, smoothing, mode="same") * 0.8 + 0.2

    kernel_cases = [
        (
            "stationary statistics (513 x 4000)",
            lambda: stationary_statistics(magnitude, 1.5),
            numpy_statistics,
        ),
        (
            "stationary mask (513 x 4000)",
            lambda: stationary_mask(
                magnitude, threshold, 0.8, freq, time_weights
            ),
            numpy_stationary,
        ),
        (
            "nonstationary mask (513 x 4000)",
            lambda: nonstationary_mask(
                magnitude,
                22_050,
                256,
                2.0,
                2.0,
                10.0,
                0.8,
                freq,
                time_weights,
            ),
            numpy_nonstationary,
        ),
    ]

    sr = 22_050
    samples = sr * 20
    seconds = np.arange(samples) / sr
    noise = rng.normal(scale=0.08, size=samples)
    signal = (
        0.35 * np.sin(2 * np.pi * 440 * seconds)
        + 0.12 * np.sin(2 * np.pi * 880 * seconds)
        + noise
    )
    common = dict(
        sr=sr,
        n_fft=1024,
        hop_length=256,
        padding=8_192,
        chunk_size=None,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
    )
    end_to_end_cases = [
        (
            "reduce_noise nonstationary (20 s)",
            lambda: mnr.reduce_noise(signal, stationary=False, **common),
            lambda: upstream.reduce_noise(signal, stationary=False, **common),
        ),
        (
            "reduce_noise stationary (20 s)",
            lambda: mnr.reduce_noise(
                signal,
                stationary=True,
                y_noise=noise[: sr * 2],
                **common,
            ),
            lambda: upstream.reduce_noise(
                signal,
                stationary=True,
                y_noise=noise[: sr * 2],
                **common,
            ),
        ),
    ]

    print(f"Machine: {cpu_name()} ({platform.machine()}, {platform.system()})")
    print()
    print("| case | mojo-noisereduce | upstream | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_function, upstream_function in (
        kernel_cases + end_to_end_cases
    ):
        mojo_result = mojo_function()
        upstream_result = upstream_function()
        if isinstance(mojo_result, tuple):
            for actual, expected in zip(mojo_result, upstream_result):
                assert np.allclose(actual, expected, atol=5e-13)
        else:
            assert np.allclose(
                mojo_result, upstream_result, atol=5e-12, rtol=2e-11
            )
        mojo_time = time_best(mojo_function)
        upstream_time = time_best(upstream_function)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | "
            f"{upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
