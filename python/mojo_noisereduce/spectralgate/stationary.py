from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from .._kernels import stationary_mask, stationary_statistics
from .base import SpectralGate


class SpectralGateStationary(SpectralGate):
    def __init__(
        self,
        y,
        sr,
        y_noise,
        n_std_thresh_stationary,
        chunk_size,
        clip_noise_stationary,
        padding,
        n_fft,
        win_length,
        hop_length,
        time_constant_s,
        freq_mask_smooth_hz,
        time_mask_smooth_ms,
        tmp_folder,
        prop_decrease,
        use_tqdm,
        n_jobs,
    ):
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
        self.n_std_thresh_stationary = n_std_thresh_stationary
        if y_noise is None:
            noise = self.y
        else:
            noise = np.array(y_noise)
            if noise.ndim == 1:
                noise = noise[np.newaxis, :]
            elif noise.ndim > 2:
                raise ValueError("Waveform must be in shape (# frames, # channels)")
            if noise.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError("y_noise must have dtype float32 or float64")
        self.y_noise = np.mean(noise, axis=0)
        if clip_noise_stationary:
            self.y_noise = self.y_noise[:chunk_size]
        _, _, noise_stft = stft(
            self.y_noise,
            nfft=self._n_fft,
            noverlap=self._win_length - self._hop_length,
            nperseg=self._win_length,
            padded=False,
        )
        noise_magnitude = np.empty(
            noise_stft.shape, dtype=np.float64, order="C"
        )
        np.abs(noise_stft, out=noise_magnitude)
        (
            self.mean_freq_noise,
            self.std_freq_noise,
            self.noise_thresh,
        ) = stationary_statistics(
            noise_magnitude, self.n_std_thresh_stationary
        )

    def spectral_gating_stationary(self, chunk):
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
            mask = stationary_mask(
                magnitude,
                self.noise_thresh,
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
        return self.spectral_gating_stationary(chunk)
