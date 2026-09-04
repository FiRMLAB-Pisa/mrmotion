"""Reconstruct a navigator's planes from what was sampled along them.

A navigator buys its speed by measuring very little: a handful of planes, each
undersampled, each read in a few milliseconds. There is no reconstruction
problem worth solving here -- the planes are registered against each other, not
diagnosed -- so this grids the samples and transforms them, and stops.

FINUFFT takes its thread count per call rather than from a global, which is why
it is not in :func:`mrmotion.single_threaded` and is passed ``nthreads`` here
instead. It defaults to one for the same reason everything else does: the scan
is waiting.
"""

from __future__ import annotations

import numpy as np

__all__ = ["reconstruct_navigator"]


def _finufft():
    """Import FINUFFT, or say what is missing."""
    try:
        import finufft
    except ImportError as error:  # pragma: no cover - finufft is a dependency
        raise ImportError(
            "navigator reconstruction needs FINUFFT: pip install finufft"
        ) from error
    return finufft


def reconstruct_navigator(
    kspace: np.ndarray,
    trajectory: np.ndarray,
    shape: tuple[int, int],
    *,
    density: np.ndarray | None = None,
    threads: int = 1,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Grid a navigator's planes and transform them.

    Parameters
    ----------
    kspace
        Samples, shaped ``(planes, samples)`` or ``(planes, coils, samples)``.
        Several coils are combined by root sum of squares, which needs no
        sensitivities and is all a registration target needs.
    trajectory
        Where each sample was taken, shaped ``(planes, samples, 2)`` in cycles
        per field of view, so within ``[-0.5, 0.5)``.
    shape
        Matrix size of each plane.
    density
        Weight per sample, shaped like one plane's samples or broadcastable to
        the trajectory. Without it every sample counts once, which under a
        radial or spiral navigator leaves the centre of k-space overweighted.
    threads
        Threads FINUFFT may use. One, because the scan is waiting.
    tolerance
        FINUFFT's accuracy. Loose by reconstruction standards, because the
        planes are registered rather than read.

    Returns
    -------
    numpy.ndarray
        Real plane magnitudes, shaped ``(planes, *shape)``, ready for
        :meth:`mrmotion.NavigatorMotionTracker.track`.

    Raises
    ------
    ValueError
        If the trajectory does not match the samples, or leaves the range a
        cycles-per-field-of-view trajectory has to be in.

    Examples
    --------
    >>> import numpy as np
    >>> from mrmotion import reconstruct_navigator
    >>> angles = np.linspace(0, np.pi, 16, endpoint=False)
    >>> radius = np.linspace(-0.5, 0.5, 64, endpoint=False)
    >>> spokes = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
    ...     axis=-1,
    ... ).reshape(1, -1, 2)
    >>> samples = np.ones((1, spokes.shape[1]), dtype=np.complex128)
    >>> reconstruct_navigator(samples, spokes, (32, 32)).shape
    (1, 32, 32)
    """
    kspace = np.asarray(kspace)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if kspace.ndim == 2:
        kspace = kspace[:, None, :]
    if kspace.ndim != 3:
        raise ValueError(
            f"expected (planes, samples) or (planes, coils, samples), got "
            f"{tuple(kspace.shape)}"
        )
    planes, coils, samples = kspace.shape
    if trajectory.shape != (planes, samples, 2):
        raise ValueError(
            f"trajectory {tuple(trajectory.shape)} does not match samples "
            f"{(planes, samples, 2)}"
        )
    if np.abs(trajectory).max() > 0.5 + 1e-9:
        raise ValueError(
            "trajectory must be in cycles per field of view, within [-0.5, 0.5]; "
            f"its largest coordinate is {np.abs(trajectory).max():.3f}"
        )

    weights = (
        None
        if density is None
        else np.broadcast_to(np.asarray(density, dtype=np.float64), (planes, samples))
    )
    finufft = _finufft()
    images = np.empty((planes, *shape), dtype=np.float64)
    for index in range(planes):
        # FINUFFT wants radians, and the trajectory is in cycles.
        first = np.ascontiguousarray(trajectory[index, :, 0] * 2 * np.pi)
        second = np.ascontiguousarray(trajectory[index, :, 1] * 2 * np.pi)
        total = np.zeros(shape, dtype=np.float64)
        for coil in range(coils):
            values = np.ascontiguousarray(kspace[index, coil].astype(np.complex128))
            if weights is not None:
                values = values * weights[index]
            gridded = finufft.nufft2d1(
                first,
                second,
                values,
                shape,
                isign=1,
                eps=tolerance,
                nthreads=threads,
            )
            total += np.abs(gridded) ** 2
        images[index] = np.sqrt(total)
    return images
