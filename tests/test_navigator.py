"""Reconstructing a navigator's planes, and handing them to the tracker."""

from __future__ import annotations

import numpy as np
import pytest

from mrmotion import RigidRegistration, reconstruct_navigator


def radial(samples: int = 64, spokes: int = 48):
    """A radial plane trajectory, in cycles per field of view."""
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples, endpoint=False)
    return np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
        axis=-1,
    ).reshape(1, -1, 2)


def sample(image: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """What a scanner would measure of ``image`` along ``trajectory``."""
    rows, columns = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    centre = (np.array(image.shape) - 1) / 2
    phase = np.exp(
        -2j
        * np.pi
        * (
            trajectory[0, :, None, None, 0] * (rows - centre[0])
            + trajectory[0, :, None, None, 1] * (columns - centre[1])
        )
    )
    return (phase * image).sum(axis=(1, 2))[None]


def phantom(size: int = 32) -> np.ndarray:
    image = np.zeros((size, size))
    image[10:22, 12:20] = 1.0
    return image


def test_a_plane_comes_back() -> None:
    """Sampled off a phantom and gridded back, the plane is the phantom."""
    truth = phantom()
    trajectory = radial()
    density = np.abs(np.linspace(-0.5, 0.5, 64, endpoint=False))
    weights = np.tile(density, trajectory.shape[1] // 64)[None]
    planes = reconstruct_navigator(
        sample(truth, trajectory), trajectory, truth.shape, density=weights
    )
    recovered = planes[0] / planes[0].max()
    assert np.corrcoef(recovered.ravel(), truth.ravel())[0, 1] > 0.9


def test_the_planes_can_be_registered() -> None:
    """The whole path: sample, grid, register, and get the shift back."""
    truth = phantom()
    trajectory = radial()
    density = np.abs(np.linspace(-0.5, 0.5, 64, endpoint=False))
    weights = np.tile(density, trajectory.shape[1] // 64)[None]
    fixed = reconstruct_navigator(
        sample(truth, trajectory), trajectory, truth.shape, density=weights
    )[0]
    moved = reconstruct_navigator(
        sample(np.roll(truth, 2, axis=0), trajectory),
        trajectory,
        truth.shape,
        density=weights,
    )[0]
    estimate = RigidRegistration()(fixed, moved, spacing=(1.0, 1.0))
    assert np.abs(np.asarray(estimate.parameters)[1:]).max() > 1.0


def test_several_coils_are_combined() -> None:
    trajectory = radial()
    samples = np.ones((1, 4, trajectory.shape[1]), dtype=np.complex128)
    assert reconstruct_navigator(samples, trajectory, (16, 16)).shape == (1, 16, 16)


def test_a_mismatched_trajectory_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match samples"):
        reconstruct_navigator(np.ones((1, 10), dtype=complex), radial(), (16, 16))


def test_a_trajectory_out_of_range_is_refused() -> None:
    trajectory = radial() * 10
    samples = np.ones((1, trajectory.shape[1]), dtype=complex)
    with pytest.raises(ValueError, match="cycles per field of view"):
        reconstruct_navigator(samples, trajectory, (16, 16))


def test_the_thread_count_reaches_finufft() -> None:
    """FINUFFT has no global, so the count has to travel with the call."""
    seen = {}
    import finufft

    original = finufft.nufft2d1

    def watching(*args, **kwargs):
        seen["nthreads"] = kwargs.get("nthreads")
        return original(*args, **kwargs)

    finufft.nufft2d1 = watching
    try:
        trajectory = radial()
        reconstruct_navigator(
            np.ones((1, trajectory.shape[1]), dtype=complex), trajectory, (16, 16)
        )
    finally:
        finufft.nufft2d1 = original
    assert seen["nthreads"] == 1
