# mojo-noisereduce

`mojo-noisereduce` is a Mojo port of the compute-heavy spectral-gating stages
in Python's [`noisereduce`](https://pypi.org/project/noisereduce/) package. It
provides the same `reduce_noise` signature for the covered CPU backend and
matches noisereduce 3.0.3 numerically, while keeping the FFT work in SciPy.

The package is useful as an importable Python library: NumPy arrays cross a
small `ctypes` boundary into compiled Mojo kernels, and the result is returned
as a NumPy array with the upstream shape and dtype.

## Coverage

Covered:

- stationary spectral gating, including per-frequency noise statistics,
  thresholding, mask smoothing, and `prop_decrease`;
- nonstationary spectral gating, including forward-backward temporal noise-floor
  smoothing, sigmoid masking, mask smoothing, and `prop_decrease`;
- mono and channel-first multichannel arrays;
- float32 and float64 input behavior (other waveform dtypes are rejected rather
  than silently narrowed);
- padded chunk processing and `n_jobs` parallel chunk execution;
- the upstream `reduce_noise` signature and the
  `SpectralGateStationary`/`SpectralGateNonStationary` class names.

Not covered:

- the optional Torch/CUDA backend (`use_torch=True`);
- plotting and synthetic-noise helper modules;
- a progress bar for `use_tqdm`.

`device` remains in the signature for compatibility, but the NumPy backend
always executes on the CPU. FFT framing and reconstruction deliberately use
`scipy.signal.stft` and `scipy.signal.istft`; replacing a mature FFT
implementation is not the valuable part of this port.

## Install

The repository pins the tested Mojo nightly and declares Python, NumPy, SciPy,
joblib, pytest, and upstream noisereduce through Pixi:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-noisereduce.so`.

## Usage

This example generates one second of noisy audio and reduces its noise without
reading or writing any files:

```python
import numpy as np
import mojo_noisereduce as nr

sr = 16_000
t = np.arange(sr) / sr
rng = np.random.default_rng(0)
noisy = np.sin(2 * np.pi * 440 * t) + rng.normal(0, 0.12, t.size)

cleaned = nr.reduce_noise(
    y=noisy,
    sr=sr,
    stationary=False,
    n_fft=512,
    hop_length=128,
    padding=2048,
    chunk_size=None,
)
print(cleaned.shape, cleaned.dtype)
```

Run it inside the environment with `pixi run python example.py`. Existing code
using upstream's covered CPU API only needs to change the import.

## How it works

SciPy creates a frequency-major complex STFT for each channel. The Python layer
writes its magnitude directly into one C-contiguous float64 array, allocates
output and scratch arrays, and passes their addresses to the shared library as
64-bit integers. Before the synchronous call, the wrapper validates non-empty
dimensions and weight lengths, converts every input to a C-contiguous float64
array, and keeps all input and output arrays strongly referenced. Arrays already
in the required dtype and layout cross this boundary without a copy. Mojo
reconstructs mutable `UnsafePointer` values
inside non-parametric C ABI exports; no Mojo-owned allocation crosses the FFI
boundary.

The stationary kernel computes top-dB-clipped means, standard deviations, and
threshold masks. The nonstationary kernel applies the same forward-backward
one-pole filter and sigmoid as upstream. Both paths smooth the mask as two
zero-padded one-dimensional convolutions, equivalent to upstream's separable
triangular 2-D filter. Contiguous columns are processed at the native float64
SIMD width, with scalar tails for remainders and boundary samples. Independent
frequency rows use up to 16 workers only when the problem exceeds a
kernel-specific 65,536-operation threshold. Python then applies the mask to the
complex STFT and calls SciPy's inverse STFT.

No GPU path is included. The separable smoothing passes are about 0.25
flop/byte, while the other stages contain row reductions or serial
forward-backward recurrences; none offers enough full-kernel arithmetic
intensity and independent work to repay host/device transfers. Passing a GPU
device name therefore continues to use the CPU backend.

## Correctness

The test suite compares intermediate statistics and masks as well as complete
audio traces against the installed noisereduce 3.0.3 package. It covers
stationary and nonstationary modes, smoothing on and off, mono and multichannel
input, float32 behavior, and serial and parallel chunking.

```bash
pixi run build
pixi run test
```

## Benchmarks

Measured with `pixi run bench`, which holds the repository's machine-wide
benchmark lock. Times are the best of three runs on an Intel Xeon E5-2697 v4 at
2.30 GHz (x86_64 Linux). The kernel rows compare against the NumPy/SciPy
formulas used by upstream; the final rows call each package's public
`reduce_noise` on the same 20-second signal. Ratios above 1 mean Mojo is faster.

| case | mojo-noisereduce | upstream | upstream / Mojo |
|---|---:|---:|---:|
| stationary statistics (513 x 4000) | 27.85 ms | 115.44 ms | 4.14x |
| stationary mask (513 x 4000) | 70.70 ms | 355.06 ms | 5.02x |
| nonstationary mask (513 x 4000) | 34.18 ms | 284.60 ms | 8.33x |
| reduce_noise nonstationary (20 s) | 239.74 ms | 314.67 ms | 1.31x |
| reduce_noise stationary (20 s) | 227.22 ms | 385.14 ms | 1.70x |

Results vary with FFT libraries, CPU topology, signal length, and smoothing
widths. Run `pixi run bench` on the target machine rather than treating these
numbers as universal.

## Development

The required checks are:

```bash
pixi run build
pixi run test
pixi run bench
```

Mojo kernels live in one compilation unit at `src/noisereduce.mojo`; the Python
package is under `python/mojo_noisereduce`.
