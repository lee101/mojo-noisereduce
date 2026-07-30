from __future__ import annotations

from .spectralgate.nonstationary import SpectralGateNonStationary
from .spectralgate.stationary import SpectralGateStationary


def reduce_noise(
    y,
    sr,
    stationary=False,
    y_noise=None,
    prop_decrease=1.0,
    time_constant_s=2.0,
    freq_mask_smooth_hz=500,
    time_mask_smooth_ms=50,
    thresh_n_mult_nonstationary=2,
    sigmoid_slope_nonstationary=10,
    n_std_thresh_stationary=1.5,
    tmp_folder=None,
    chunk_size=600000,
    padding=30000,
    n_fft=1024,
    win_length=None,
    hop_length=None,
    clip_noise_stationary=True,
    use_tqdm=False,
    n_jobs=1,
    use_torch=False,
    device="cuda",
):
    if use_torch:
        raise NotImplementedError("the covered subset is the NumPy/SciPy CPU backend")
    common = dict(
        y=y,
        sr=sr,
        chunk_size=chunk_size,
        padding=padding,
        prop_decrease=prop_decrease,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        time_constant_s=time_constant_s,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
        tmp_folder=tmp_folder,
        use_tqdm=use_tqdm,
        n_jobs=n_jobs,
    )
    if stationary:
        gate = SpectralGateStationary(
            y_noise=y_noise,
            n_std_thresh_stationary=n_std_thresh_stationary,
            clip_noise_stationary=clip_noise_stationary,
            **common,
        )
    else:
        gate = SpectralGateNonStationary(
            thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
            sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
            **common,
        )
    return gate.get_traces()
