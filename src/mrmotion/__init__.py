"""Rigid motion estimation and tracking for MRI: navigator reconstruction, registration and an extended Kalman filter, all pinned to one thread."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

from ._threads import single_threaded, thread_counts

__all__ = [
    "__version__",
    "single_threaded",
    "thread_counts",
]
