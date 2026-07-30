from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .._kernels import triangular_weights


def _smoothing_filter(n_grad_freq, n_grad_time):
    return np.outer(
        triangular_weights(n_grad_freq), triangular_weights(n_grad_time)
    )


class SpectralGate:
    def __init__(
        self,
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
    ):
        self.sr = sr
        y = np.array(y)
        self.flat = y.ndim == 1
        if self.flat:
            self.y = y[np.newaxis, :]
        elif y.ndim == 2:
            self.y = y
        else:
            raise ValueError("Waveform must be in shape (# frames, # channels)")
        if y.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("y must have dtype float32 or float64")
        self._dtype = y.dtype
        self.n_channels, self.n_frames = self.y.shape
        self._chunk_size = chunk_size
        self.padding = padding
        self.n_jobs = n_jobs
        self.use_tqdm = use_tqdm
        self._tmp_folder = tmp_folder
        self._n_fft = n_fft
        self._win_length = n_fft if win_length is None else win_length
        self._hop_length = (
            self._win_length // 4 if hop_length is None else hop_length
        )
        self._time_constant_s = time_constant_s
        self._prop_decrease = prop_decrease
        if freq_mask_smooth_hz is None and time_mask_smooth_ms is None:
            self.smooth_mask = False
            self._freq_weights = triangular_weights(0)
            self._time_weights = triangular_weights(0)
        else:
            self._generate_mask_smoothing_filter(
                freq_mask_smooth_hz, time_mask_smooth_ms
            )

    def _generate_mask_smoothing_filter(
        self, freq_mask_smooth_hz, time_mask_smooth_ms
    ):
        if freq_mask_smooth_hz is None:
            n_grad_freq = 1
        else:
            n_grad_freq = int(
                freq_mask_smooth_hz / (self.sr / (self._n_fft / 2))
            )
            if n_grad_freq < 1:
                minimum = int(self.sr / (self._n_fft / 2))
                raise ValueError(
                    f"freq_mask_smooth_hz needs to be at least {minimum}Hz"
                )
        if time_mask_smooth_ms is None:
            n_grad_time = 1
        else:
            n_grad_time = int(
                time_mask_smooth_ms
                / ((self._hop_length / self.sr) * 1000)
            )
            if n_grad_time < 1:
                minimum = int((self._hop_length / self.sr) * 1000)
                raise ValueError(
                    f"time_mask_smooth_ms needs to be at least {minimum}ms"
                )
        self.smooth_mask = not (n_grad_time == 1 and n_grad_freq == 1)
        if self.smooth_mask:
            self._freq_weights = triangular_weights(n_grad_freq)
            self._time_weights = triangular_weights(n_grad_time)
            self._smoothing_filter = np.outer(
                self._freq_weights, self._time_weights
            )
        else:
            self._freq_weights = triangular_weights(0)
            self._time_weights = triangular_weights(0)

    def _read_chunk(self, i1, i2):
        i1b = max(i1, 0)
        i2b = min(i2, self.n_frames)
        chunk = np.zeros(
            (self.n_channels, i2 - i1), dtype=self.y.dtype
        )
        chunk[:, i1b - i1 : i2b - i1] = self.y[:, i1b:i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame):
        i1 = start_frame - self.padding
        i2 = end_frame + self.padding
        padded = self._read_chunk(i1, i2)
        filtered = self._do_filter(padded)
        return filtered[:, start_frame - i1 : end_frame - i1]

    def _get_filtered_chunk(self, index):
        start = index * self._chunk_size
        end = (index + 1) * self._chunk_size
        return self.filter_chunk(start, end)

    def _do_filter(self, chunk):
        raise NotImplementedError

    def get_traces(self, start_frame=None, end_frame=None):
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.n_frames if end_frame is None else end_frame
        if (
            self._chunk_size is not None
            and end_frame - start_frame > self._chunk_size
        ):
            first = start_frame // self._chunk_size
            last = (end_frame - 1) // self._chunk_size
            indices = list(range(first, last + 1))
            if self.n_jobs == 1:
                filtered = [self._get_filtered_chunk(index) for index in indices]
            else:
                filtered = Parallel(
                    n_jobs=self.n_jobs, temp_folder=self._tmp_folder
                )(
                    delayed(self._get_filtered_chunk)(index)
                    for index in indices
                )
            pieces = []
            for index, chunk in zip(indices, filtered):
                local_start = (
                    start_frame - index * self._chunk_size
                    if index == first
                    else 0
                )
                local_end = (
                    end_frame - index * self._chunk_size
                    if index == last
                    else self._chunk_size
                )
                pieces.append(chunk[:, local_start:local_end])
            result = np.concatenate(pieces, axis=1).astype(
                self._dtype, copy=False
            )
        else:
            result = self.filter_chunk(start_frame, end_frame).astype(
                self._dtype, copy=False
            )
        return result.reshape(-1) if self.flat else result
