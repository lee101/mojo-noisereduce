from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.signal import fftconvolve, stft

import mojo_noisereduce as mnr
from mojo_noisereduce._kernels import (
    nonstationary_mask,
    stationary_mask,
    stationary_statistics,
    triangular_weights,
)
from mojo_noisereduce._lib import f64
from mojo_noisereduce.spectralgate.base import _smoothing_filter
from mojo_noisereduce.spectralgate.nonstationary import (
    get_time_smoothed_representation,
)

upstream = pytest.importorskip("noisereduce")
from noisereduce.spectralgate.base import (  # noqa: E402
    _smoothing_filter as upstream_smoothing_filter,
)
from noisereduce.spectralgate.nonstationary import (  # noqa: E402
    get_time_smoothed_representation as upstream_time_smooth,
)
from noisereduce.spectralgate.stationary import (  # noqa: E402
    SpectralGateStationary as UpstreamStationary,
)
from noisereduce.spectralgate.utils import _amp_to_db, sigmoid  # noqa: E402

RNG = np.random.default_rng(1234)


def audio(dtype=np.float64, channels=1, samples=12_000, sr=8_000):
    time = np.arange(samples) / sr
    clean = 0.35 * np.sin(2 * np.pi * 440 * time)
    clean += 0.12 * np.sin(2 * np.pi * 880 * time)
    noise = RNG.normal(scale=0.07, size=samples)
    signal = (clean + noise).astype(dtype)
    if channels == 2:
        signal = np.vstack([signal, (0.7 * signal).astype(dtype)])
    return signal, noise.astype(dtype)


COMMON = dict(
    sr=8_000,
    n_fft=256,
    win_length=256,
    hop_length=64,
    padding=512,
    chunk_size=None,
    freq_mask_smooth_hz=250,
    time_mask_smooth_ms=32,
)


def test_reduce_noise_signature_matches_upstream():
    assert inspect.signature(mnr.reduce_noise) == inspect.signature(
        upstream.reduce_noise
    )


@pytest.mark.parametrize("freq_radius,time_radius", [(1, 1), (3, 5), (11, 4)])
def test_smoothing_filter_matches_upstream(freq_radius, time_radius):
    assert np.allclose(
        _smoothing_filter(freq_radius, time_radius),
        upstream_smoothing_filter(freq_radius, time_radius),
        atol=2e-17,
    )


def test_stationary_statistics_match_upstream_numpy():
    magnitude = np.abs(RNG.normal(size=(129, 311))) + 1e-8
    db = _amp_to_db(magnitude)
    mean, deviation, threshold = stationary_statistics(magnitude, 1.7)
    assert np.allclose(mean, db.mean(axis=1), atol=2e-14)
    assert np.allclose(deviation, db.std(axis=1), atol=2e-14)
    assert np.allclose(
        threshold, db.mean(axis=1) + 1.7 * db.std(axis=1), atol=3e-14
    )


def test_simd_tail_and_parallel_threshold_paths_match_upstream():
    magnitude = np.abs(RNG.normal(size=(129, 509))) + 1e-8
    db = _amp_to_db(magnitude)
    mean, deviation, threshold = stationary_statistics(magnitude, 1.3)
    assert np.allclose(mean, db.mean(axis=1), atol=2e-14)
    assert np.allclose(deviation, db.std(axis=1), atol=2e-14)
    assert np.allclose(
        threshold, db.mean(axis=1) + 1.3 * db.std(axis=1), atol=3e-14
    )

    freq = triangular_weights(3)
    time = triangular_weights(2)
    raw = (_amp_to_db(magnitude) > threshold[:, None]) * 0.77 + 0.23
    expected = fftconvolve(raw, np.outer(freq, time), mode="same")
    actual = stationary_mask(magnitude, threshold, 0.77, freq, time)
    assert np.allclose(actual, expected, atol=5e-15)


def test_float64_c_contiguous_input_stays_zero_copy():
    values = np.empty((17, 19), dtype=np.float64, order="C")
    assert f64(values) is values


def test_stationary_mask_preserves_strict_db_threshold_boundary():
    magnitude = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    db = _amp_to_db(magnitude)
    for threshold in db[0]:
        actual = stationary_mask(
            magnitude,
            np.array([threshold]),
            freq_weights=np.ones(1),
            time_weights=np.ones(1),
        )
        assert np.array_equal(actual, db > threshold)


@pytest.mark.parametrize("smooth", [False, True])
def test_stationary_mask_matches_upstream_formula(smooth):
    magnitude = np.abs(RNG.normal(size=(129, 257))) + 1e-8
    _, _, threshold = stationary_statistics(magnitude, 1.5)
    freq = triangular_weights(5) if smooth else triangular_weights(0)
    time = triangular_weights(3) if smooth else triangular_weights(0)
    raw = (_amp_to_db(magnitude) > threshold[:, None]) * 0.72 + 0.28
    expected = fftconvolve(raw, np.outer(freq, time), mode="same")
    actual = stationary_mask(magnitude, threshold, 0.72, freq, time)
    assert np.allclose(actual, expected, atol=4e-15)


@pytest.mark.parametrize("smooth", [False, True])
def test_nonstationary_mask_matches_upstream_formula(smooth):
    magnitude = np.abs(RNG.normal(size=(129, 257))) + 1e-6
    freq = triangular_weights(5) if smooth else triangular_weights(0)
    time = triangular_weights(3) if smooth else triangular_weights(0)
    smoothed = upstream_time_smooth(magnitude, 8_000, 64, 1.3)
    above = (magnitude - smoothed) / smoothed
    raw = sigmoid(above, -1.8, 8.0)
    expected = fftconvolve(raw, np.outer(freq, time), mode="same")
    expected = expected * 0.81 + 0.19
    actual = nonstationary_mask(
        magnitude, 8_000, 64, 1.3, 1.8, 8.0, 0.81, freq, time
    )
    assert np.allclose(actual, expected, atol=2e-14)


def test_time_smoothed_representation_matches_upstream():
    magnitude = np.abs(RNG.normal(size=(65, 81))) + 1e-5
    actual = get_time_smoothed_representation(magnitude, 16_000, 128, 0.7)
    expected = upstream_time_smooth(magnitude, 16_000, 128, 0.7)
    assert np.array_equal(actual, expected)


def test_stationary_end_to_end_matches_upstream():
    signal, noise = audio()
    options = {
        **COMMON,
        "stationary": True,
        "y_noise": noise[:3_000],
        "n_std_thresh_stationary": 1.7,
        "prop_decrease": 0.83,
    }
    actual = mnr.reduce_noise(signal, **options)
    expected = upstream.reduce_noise(signal, **options)
    assert np.allclose(actual, expected, atol=3e-14, rtol=2e-13)


def test_nonstationary_end_to_end_matches_upstream():
    signal, _ = audio()
    options = {
        **COMMON,
        "stationary": False,
        "time_constant_s": 1.1,
        "thresh_n_mult_nonstationary": 1.6,
        "sigmoid_slope_nonstationary": 7,
        "prop_decrease": 0.9,
    }
    actual = mnr.reduce_noise(signal, **options)
    expected = upstream.reduce_noise(signal, **options)
    assert np.allclose(actual, expected, atol=3e-14, rtol=2e-13)


@pytest.mark.parametrize("stationary", [False, True])
def test_multichannel_matches_upstream(stationary):
    signal, noise = audio(channels=2, samples=8_000)
    options = {
        **COMMON,
        "stationary": stationary,
        "y_noise": noise[:2_000] if stationary else None,
    }
    actual = mnr.reduce_noise(signal, **options)
    expected = upstream.reduce_noise(signal, **options)
    assert actual.shape == signal.shape
    assert np.allclose(actual, expected, atol=3e-14, rtol=2e-13)


def test_float32_dtype_and_values_match_upstream():
    signal, _ = audio(dtype=np.float32, samples=8_000)
    actual = mnr.reduce_noise(signal, stationary=False, **COMMON)
    expected = upstream.reduce_noise(signal, stationary=False, **COMMON)
    assert actual.dtype == expected.dtype == np.float32
    assert np.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_chunked_parallel_output_matches_upstream():
    signal, _ = audio(samples=12_000)
    options = {
        **COMMON,
        "chunk_size": 4_000,
        "stationary": False,
        "n_jobs": 2,
    }
    actual = mnr.reduce_noise(signal, **options)
    expected = upstream.reduce_noise(signal, **{**options, "n_jobs": 1})
    assert np.allclose(actual, expected, atol=3e-14, rtol=2e-13)


def test_prop_decrease_zero_matches_upstream_reconstruction():
    signal, _ = audio(samples=8_000)
    for stationary in (False, True):
        actual = mnr.reduce_noise(
            signal, stationary=stationary, prop_decrease=0.0, **COMMON
        )
        expected = upstream.reduce_noise(
            signal, stationary=stationary, prop_decrease=0.0, **COMMON
        )
        assert np.allclose(actual, expected, atol=3e-14, rtol=2e-13)


def test_class_statistics_match_upstream():
    signal, noise = audio(samples=8_000)
    options = dict(
        y=signal,
        sr=8_000,
        y_noise=noise,
        n_std_thresh_stationary=1.5,
        chunk_size=None,
        clip_noise_stationary=True,
        padding=512,
        n_fft=256,
        win_length=256,
        hop_length=64,
        time_constant_s=2.0,
        freq_mask_smooth_hz=250,
        time_mask_smooth_ms=32,
        tmp_folder=None,
        prop_decrease=1.0,
        use_tqdm=False,
        n_jobs=1,
    )
    ours = mnr.SpectralGateStationary(**options)
    theirs = UpstreamStationary(**options)
    assert np.allclose(ours.mean_freq_noise, theirs.mean_freq_noise, atol=2e-14)
    assert np.allclose(ours.std_freq_noise, theirs.std_freq_noise, atol=2e-14)
    assert np.allclose(ours.noise_thresh, theirs.noise_thresh, atol=3e-14)


def test_invalid_shapes_and_smoothing_parameters_match_behavior():
    with pytest.raises(ValueError, match="Waveform"):
        mnr.reduce_noise(np.zeros((2, 3, 4)), sr=8_000)
    with pytest.raises(ValueError, match="freq_mask_smooth_hz"):
        mnr.reduce_noise(
            np.zeros(2_000),
            sr=8_000,
            n_fft=256,
            freq_mask_smooth_hz=1,
        )


def test_fortran_order_stft_magnitude_is_handled():
    signal, _ = audio(samples=2_000)
    _, _, spectrum = stft(
        signal, nfft=256, nperseg=256, noverlap=192, padded=False
    )
    magnitude = np.abs(spectrum)
    assert magnitude.flags.f_contiguous
    assert spectrum.shape[0] == 129
    mean, deviation, _ = stationary_statistics(magnitude)
    db = _amp_to_db(magnitude)
    assert np.allclose(mean, db.mean(axis=1), atol=2e-14)
    assert np.allclose(deviation, db.std(axis=1), atol=2e-14)


def test_torch_backend_is_explicitly_out_of_scope():
    with pytest.raises(NotImplementedError, match="CPU backend"):
        mnr.reduce_noise(np.zeros(1_000), sr=8_000, use_torch=True)


def test_device_name_is_ignored_by_covered_cpu_backend():
    signal, _ = audio(samples=2_000)
    options = {**COMMON, "stationary": False}
    expected = mnr.reduce_noise(signal, device="cpu", **options)
    actual = mnr.reduce_noise(signal, device="cuda", **options)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (3,)])
def test_empty_or_non_matrix_spectrogram_is_rejected_before_ffi(shape):
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        stationary_statistics(np.empty(shape))


@pytest.mark.parametrize("weights", [np.empty(0), np.ones((1, 1))])
def test_malformed_weights_are_rejected_before_ffi(weights):
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        stationary_mask(np.ones((2, 3)), np.ones(2), freq_weights=weights)


def test_invalid_nonstationary_time_parameters_are_rejected_before_ffi():
    with pytest.raises(ValueError, match="must be positive"):
        nonstationary_mask(np.ones((2, 3)), 8_000, 0)


@pytest.mark.parametrize("dtype", [np.int16, np.float16, np.complex64])
def test_unsupported_waveform_dtype_is_not_silently_narrowed(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        mnr.reduce_noise(np.ones(1_000, dtype=dtype), sr=8_000)


def test_class_get_traces_honors_nonzero_start_frame():
    signal, _ = audio(samples=4_000)
    gate = mnr.SpectralGateNonStationary(
        y=signal,
        sr=8_000,
        chunk_size=None,
        padding=512,
        n_fft=256,
        win_length=256,
        hop_length=64,
        time_constant_s=2.0,
        freq_mask_smooth_hz=250,
        time_mask_smooth_ms=32,
        thresh_n_mult_nonstationary=2,
        sigmoid_slope_nonstationary=10,
        tmp_folder=None,
        prop_decrease=1.0,
        use_tqdm=False,
        n_jobs=1,
    )
    assert gate.get_traces(500, 1_500).shape == (1_000,)
