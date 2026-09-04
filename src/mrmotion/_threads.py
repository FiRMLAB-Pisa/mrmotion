"""Hold the whole pipeline to one thread.

Prospective correction runs while the scan runs. It shares a machine with the
reconstruction it is steering, and what it owes that scan is a pose before the
next acquisition rather than a pose as fast as possible: a step that usually
takes 20 ms and occasionally takes 200 because eight threads contended for a
machine that had none to spare is worse than one that always takes 40.

Four libraries take threads here and each is told separately, because none of
them knows about the others. Setting them once at import would be rude to
whatever else shares the process, so they are scoped instead: the count is put
back on the way out, including when the block raises.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

__all__ = ["single_threaded", "thread_counts"]


@contextmanager
def _simpleitk(count: int) -> Iterator[None]:
    """ITK's registration threads, which are process-wide."""
    try:
        import SimpleITK
    except ImportError:  # pragma: no cover - SimpleITK is a hard dependency
        yield
        return
    process = SimpleITK.ProcessObject
    restore = process.GetGlobalDefaultNumberOfThreads()
    process.SetGlobalDefaultNumberOfThreads(count)
    try:
        yield
    finally:
        process.SetGlobalDefaultNumberOfThreads(restore)


@contextmanager
def _torch(count: int) -> Iterator[None]:
    """Torch's intra-op threads, if torch is here at all."""
    try:
        import torch
    except ImportError:
        yield
        return
    restore = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(restore)


@contextmanager
def _blas(count: int) -> Iterator[None]:
    """Whatever OpenMP or BLAS runtime numpy and scipy were built against."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl is a dependency
        yield
        return
    with threadpool_limits(limits=count):
        yield


@contextmanager
def _openmp_environment(count: int) -> Iterator[None]:
    """Set the variables a library started inside the block will read.

    ``threadpool_limits`` reaches runtimes already loaded; a plan built inside
    the block by something loaded lazily reads the environment instead.
    """
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    restore = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(count)
    try:
        yield
    finally:
        for name, value in restore.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def single_threaded(count: int = 1) -> Iterator[None]:
    """Run the block with every library here held to ``count`` threads.

    FINUFFT is not in the list because it takes its thread count per plan
    rather than from a global: :func:`mrmotion.reconstruct_navigator` passes
    ``nthreads`` to the plan it builds. Everything that has a global is set
    here.

    Parameters
    ----------
    count
        Threads to allow. One, unless something has measured otherwise.

    Examples
    --------
    >>> from mrmotion import single_threaded, thread_counts
    >>> with single_threaded():
    ...     counts = thread_counts()
    >>> counts["torch"] in (1, None)
    True
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    with (
        _openmp_environment(count),
        _blas(count),
        _torch(count),
        _simpleitk(count),
    ):
        yield


def thread_counts() -> dict[str, Any]:
    """Report what each library is set to, for asserting it rather than hoping.

    Returns
    -------
    dict
        One entry per library, ``None`` where it is not installed.
    """
    counts: dict[str, Any] = {"torch": None, "simpleitk": None, "blas": None}
    with suppress(ImportError):
        import torch

        counts["torch"] = torch.get_num_threads()
    with suppress(ImportError):
        import SimpleITK

        counts["simpleitk"] = SimpleITK.ProcessObject.GetGlobalDefaultNumberOfThreads()
    with suppress(ImportError):
        from threadpoolctl import threadpool_info

        limits = [pool["num_threads"] for pool in threadpool_info()]
        counts["blas"] = max(limits) if limits else None
    return counts
