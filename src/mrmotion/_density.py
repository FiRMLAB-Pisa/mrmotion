"""Density compensation for a navigator trajectory, estimated once.

A non-Cartesian navigator visits the centre of k-space far more often than the
edge, so the adjoint of the sampling operator is not the image: the low
frequencies are counted many times over and the result is blurred and
shaded. Weighting each sample by the reciprocal of how densely its
neighbourhood was visited undoes that, and the weights that do it are a
property of the trajectory alone -- not of the data, and not of the head. A
real-time application estimates them once, when it learns the trajectory, and
spends nothing on them again.

An analytic ramp is the usual shortcut for radial spokes and it is wrong where
it matters. It keeps growing with radius while the spokes are separating faster
than the gridding kernel is wide, so the outer, most sparsely visited k-space
is the part it over-weights. The Pipe-Menon fixed point measures the density
the reconstruction actually sees instead of assuming it, and stops growing
where the assumption fails.

This wraps mri-nufft's implementation of that fixed point, which reaches it
through the same FINUFFT that reconstructs the navigator.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["estimate_density"]


def _pipe():
    """Import mri-nufft's density estimation, or say what is missing."""
    try:
        from mrinufft.density import pipe
    except ImportError as error:  # pragma: no cover - mri-nufft is a dependency
        raise ImportError(
            "density estimation needs mri-nufft: pip install mri-nufft"
        ) from error
    return pipe


def estimate_density(
    trajectory: np.ndarray,
    shape: tuple[int, int],
    *,
    iterations: int = 160,
    oversampling: float = 2.0,
    backend: str = "finufft",
    threads: int = 1,
) -> np.ndarray:
    """Pipe-Menon density compensation weights for a navigator trajectory.

    Parameters
    ----------
    trajectory
        Where the navigator samples, in cycles per field of view over
        ``[-1/2, 1/2]``, as ``(samples, 2)`` for one plane or
        ``(planes, samples, 2)`` for several. Planes are weighted one at a
        time: each is reconstructed on its own, so each has its own density.
    shape
        The grid the navigator is reconstructed onto, which is what the
        density is relative to.
    iterations
        Fixed-point iterations. The iteration converges slowly and the weights
        are still moving by several percent when the pose registered from the
        plane they produce has stopped: on a 96-squared navigator of 96 golden
        angle spokes, that pose settles to a hundredth of a degree somewhere
        past a hundred iterations, and the default sits above it. Raising it
        costs only start-up time, and on a larger grid or a 3D navigator it
        costs proportionally more of it.
    oversampling
        Grid oversampling the fixed point runs on.
    backend
        The mri-nufft backend to run it through. ``"finufft"`` runs on the CPU
        and needs nothing installed that the navigator does not already need;
        ``"gpunufft"`` and ``"cufinufft"`` are faster if present.
    threads
        Threads the backend may use. One, because a real-time application is
        holding a core for the sequence even while it is only starting up.

    Returns
    -------
    numpy.ndarray
        Weights shaped like ``trajectory`` without its last axis, to be passed
        to :func:`mrmotion.reconstruct_navigator` as ``density``.

    Raises
    ------
    ValueError
        If the trajectory is not a stack of 2D coordinates, or leaves
        ``[-1/2, 1/2]``.

    Examples
    --------
    Estimated once, when the application learns the trajectory:

    >>> import numpy as np
    >>> import mrmotion as mm
    >>> angles = np.arange(16) * np.pi * (3 - np.sqrt(5))
    >>> radius = np.linspace(-0.5, 0.5, 32, endpoint=False)
    >>> spokes = np.stack(
    ...     [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
    ...     axis=-1,
    ... ).reshape(1, -1, 2)
    >>> weights = mm.estimate_density(spokes, (32, 32))
    >>> weights.shape, bool((weights >= 0).all())
    ((1, 512), True)
    """
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim == 2:
        stacked, squeeze = trajectory[None], True
    elif trajectory.ndim == 3:
        stacked, squeeze = trajectory, False
    else:
        raise ValueError(
            "trajectory must be (samples, 2) or (planes, samples, 2), got "
            f"{trajectory.shape}"
        )
    if stacked.shape[-1] != 2:
        raise ValueError(f"a plane's coordinates are 2D, got {stacked.shape[-1]}D")
    if np.abs(stacked).max() > 0.5:
        raise ValueError(
            "trajectory is in cycles per field of view and must lie within "
            f"[-1/2, 1/2], got {np.abs(stacked).max():.3f}"
        )

    pipe = _pipe()
    # mri-nufft warns that it is rescaling to [-pi, pi) on the assumption the
    # samples were in [-1/2, 1/2). They were: that is checked above.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*rescaled to.*")
        weights = [
            np.abs(
                np.asarray(
                    pipe(
                        np.ascontiguousarray(plane),
                        tuple(int(size) for size in shape),
                        backend=backend,
                        max_iter=int(iterations),
                        osf=float(oversampling),
                        nthreads=int(threads),
                    )
                )
            ).astype(np.float64)
            for plane in stacked
        ]
    estimate = np.stack(weights)
    return estimate[0] if squeeze else estimate
