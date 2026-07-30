from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, istft, stft

from .._kernels import nonstationary_mask
from .base import SpectralGate


def get_time_smoothed_representation(
    spectral, samplerate, hop_length, time_constant_s=0.001
):
    t_frames = time_constant_s * samplerate / float(hop_length)
    coefficient = (
        np.sqrt(1 + 4 * t_frames**2) - 1
    ) / (2 * t_frames**2)
    return filtfilt(
        [coefficient],
        [1, coefficient - 1],
        spectral,
        axis=-1,
        padtype=None,
    )


class SpectralGateNonStationary(SpectralGate):
    def __init__(
        self,
        y,
        sr,
        chunk_size,
        padding,
        n_fft,
        win_length,
        hop_length,
        time_constant_s,
        freq_mask_smooth_hz,
        time_mask_smooth_ms,
        thresh_n_mult_nonstationary,
        sigmoid_slope_nonstationary,
        tmp_folder,
        prop_decrease,
        use_tqdm,
        n_jobs,
    ):
        self._thresh_n_mult_nonstationary = thresh_n_mult_nonstationary
        self._sigmoid_slope_nonstationary = sigmoid_slope_nonstationary
        super().__init__(
            y,
            sr,
            prop_decrease,
            chunk_size,
            padding,
            n_fft,
            win_length,
            hop_length,
            time_constant_s,
            freq_mask_smooth_hz,
            time_mask_smooth_ms,
            tmp_folder,
            use_tqdm,
            n_jobs,
        )

    def spectral_gating_nonstationary(self, chunk):
        denoised_channels = np.zeros(chunk.shape, chunk.dtype)
        for channel_index, channel in enumerate(chunk):
            _, _, spectrum = stft(
                channel,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
                padded=False,
            )
            magnitude = np.empty(
                spectrum.shape, dtype=np.float64, order="C"
            )
            np.abs(spectrum, out=magnitude)
            mask = nonstationary_mask(
                magnitude,
                self.sr,
                self._hop_length,
                self._time_constant_s,
                self._thresh_n_mult_nonstationary,
                self._sigmoid_slope_nonstationary,
                self._prop_decrease,
                self._freq_weights,
                self._time_weights,
            )
            _, signal = istft(
                spectrum * mask,
                nfft=self._n_fft,
                noverlap=self._win_length - self._hop_length,
                nperseg=self._win_length,
            )
            denoised_channels[channel_index, : len(signal)] = signal
        return denoised_channels

    def _do_filter(self, chunk):
        return self.spectral_gating_nonstationary(chunk)
