"""ctypes bindings for the Mojo spectral-gating kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_NOISEREDUCE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-noisereduce.so"
)
SOURCE = os.path.join(ROOT, "src", "noisereduce.mojo")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mnr_stationary_threshold": ([I, I, I, F, I, I, I], None),
    "mnr_stationary_mask": ([I, I, I, I, F, I, I, I, I, I, I], None),
    "mnr_nonstationary_mask": (
        [I, I, I, F, F, F, F, I, I, I, I, I, I],
        None,
    ),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and (
            os.environ.get("MOJO_NOISEREDUCE_LIB")
            or os.path.getmtime(LIB) >= os.path.getmtime(SOURCE)
        )
    ):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    if not os.path.exists(script):
        raise BuildError(f"missing shared library and build script: {LIB}")
    result = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype
        _library = library
    return _library


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(values: np.ndarray) -> int:
    address = int(values.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null NumPy buffer to Mojo")
    return address
