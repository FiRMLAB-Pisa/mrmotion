"""Density compensation weights, and what they are worth to a pose."""

from __future__ import annotations

import finufft
import numpy as np
import pytest

from mrmotion import RigidRegistration, estimate_density, reconstruct_navigator

GRID = 64
SPOKES = 64


def spokes(count: int = SPOKES, samples: int = GRID) -> np.ndarray:
    angles = np.arange(count) * np.pi * (3 - np.sqrt(5))
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    return np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
        axis=-1,
    ).reshape(-1, 2)


def phantom(size: int = GRID) -> np.ndarray:
    rows, columns = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    image = ((rows / 0.7) ** 2 + (columns / 0.55) ** 2 < 1).astype(float)
    image[size // 3 : size // 2, size // 3 : 2 * size // 3] += 0.6
    return image


def acquire(image: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    return finufft.nufft2d2(
        np.ascontiguousarray(2 * np.pi * trajectory[:, 0]),
        np.ascontiguousarray(2 * np.pi * trajectory[:, 1]),
        np.ascontiguousarray(image.astype(np.complex128)),
        isign=-1,
        eps=1e-6,
        nthreads=1,
    )


def test_the_weights_follow_the_trajectory() -> None:
    """One weight per sample, non-negative, and the same shape in as out."""
    trajectory = spokes()
    flat = estimate_density(trajectory, (GRID, GRID), iterations=5)
    stacked = estimate_density(trajectory[None], (GRID, GRID), iterations=5)
    assert flat.shape == (trajectory.shape[0],)
    assert stacked.shape == (1, trajectory.shape[0])
    assert (flat >= 0).all()
    assert np.allclose(flat, stacked[0])


def test_the_weight_rises_with_radius() -> None:
    """The centre is visited by every spoke, so it counts for least."""
    weights = estimate_density(spokes(), (GRID, GRID), iterations=20)
    radius = np.abs(np.linspace(-0.5, 0.5, GRID, endpoint=False))
    profile = weights.reshape(SPOKES, GRID).mean(axis=0)
    inner, outer = profile[radius < 0.1].mean(), profile[radius > 0.3].mean()
    assert outer > 4 * inner


def test_it_beats_the_ramp_on_the_image() -> None:
    """The ramp over-weights the outer k-space the spokes have left sparse."""
    trajectory = spokes()
    truth = phantom()
    truth = truth / truth.max()
    data = acquire(truth, trajectory)[None, None]

    ramp = np.tile(np.abs(np.linspace(-0.5, 0.5, GRID, endpoint=False)), SPOKES)
    pipe = estimate_density(trajectory, (GRID, GRID), iterations=40)

    def residual(density):
        plane = reconstruct_navigator(
            data, trajectory[None], (GRID, GRID), density=density[None]
        )[0]
        return np.sqrt(((plane / plane.max() - truth) ** 2).mean())

    assert residual(pipe) < residual(ramp)


def test_it_sharpens_a_pose() -> None:
    """Which is the point: a better plane is a better registration."""
    trajectory = spokes()
    truth = phantom()
    shifted = np.roll(truth, 5, axis=1)
    ramp = np.tile(np.abs(np.linspace(-0.5, 0.5, GRID, endpoint=False)), SPOKES)
    pipe = estimate_density(trajectory, (GRID, GRID), iterations=40)
    registration = RigidRegistration()

    def measured(density):
        planes = [
            reconstruct_navigator(
                acquire(image, trajectory)[None, None],
                trajectory[None],
                (GRID, GRID),
                density=density[None],
            )[0]
            for image in (truth, shifted)
        ]
        estimate = registration(*planes, spacing=(1.0, 1.0)).parameters
        return abs(np.asarray(estimate)[2] - 5.0)

    assert measured(pipe) <= measured(ramp)


def test_a_trajectory_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="cycles per field of view"):
        estimate_density(spokes() * 3, (GRID, GRID), iterations=2)


def test_a_three_dimensional_trajectory_is_refused() -> None:
    trajectory = np.zeros((SPOKES * GRID, 3))
    with pytest.raises(ValueError, match="coordinates are 2D"):
        estimate_density(trajectory, (GRID, GRID), iterations=2)
