"""What a navigator costs: gridding, registration, and the pose that comes out.

Run it as ``python scripts/benchmark_latency.py``. Every number is measured on
one thread, because that is how the package runs.
"""

from __future__ import annotations

import time

import finufft
import numpy as np
from scipy.ndimage import rotate, shift

import mrmotion as mm

REPEATS = 10
NAVIGATORS = 12
AXES = [((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 0, 1))]


def spokes(grid, count):
    """Golden-angle radial spokes, in cycles per field of view."""
    angles = np.arange(count) * np.pi * (3 - np.sqrt(5))
    radius = np.linspace(-0.5, 0.5, grid, endpoint=False)
    return np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)],
        axis=-1,
    ).reshape(-1, 2)


def moved(image):
    """Put the anatomy somewhere else.

    Registering an image against itself converges immediately and would time
    the wrong thing.
    """
    return shift(rotate(image, 6.0, reshape=False, order=3), (4.0, -3.0), order=3)


def phantom(size):
    """Something with an edge to register."""
    rows, columns = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    image = ((rows / 0.75) ** 2 + (columns / 0.6) ** 2 < 1).astype(float)
    image[size // 3 : size // 2, size // 3 : 2 * size // 3] += 0.6
    return image


def acquire(image, trajectory, coils):
    """Sample the image along the trajectory, once per coil."""
    sensitivities = np.ones((1, *image.shape)) if coils == 1 else _maps(image, coils)
    return np.stack(
        [
            finufft.nufft2d2(
                np.ascontiguousarray(2 * np.pi * trajectory[:, 0]),
                np.ascontiguousarray(2 * np.pi * trajectory[:, 1]),
                np.ascontiguousarray((sensitivity * image).astype(np.complex128)),
                isign=-1,
                eps=1e-6,
                nthreads=1,
            )
            for sensitivity in sensitivities
        ]
    )


def _maps(image, coils):
    rows, columns = np.mgrid[0 : image.shape[0], 0 : image.shape[1]] / image.shape[0]
    return np.stack(
        [
            np.exp(
                -(
                    (columns - 0.5 - 0.6 * np.cos(angle)) ** 2
                    + (rows - 0.5 - 0.6 * np.sin(angle)) ** 2
                )
                / 0.35
            )
            for angle in np.linspace(0, 2 * np.pi, coils, endpoint=False)
        ]
    )


def timed(call, repeats=REPEATS):
    """Milliseconds per call, after one to warm the plans."""
    call()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    return 1e3 * (time.perf_counter() - start) / repeats


def main():
    """Print the table."""
    with mm.single_threaded():
        print("start-up, once per trajectory")
        for grid, count in [(64, 64), (96, 96), (128, 128)]:
            trajectory = spokes(grid, count)
            elapsed = timed(
                lambda t=trajectory, g=grid: mm.estimate_density(t, (g, g)), repeats=3
            )
            print(
                f"  density, {grid}^2 from {count} spokes "
                f"({trajectory.shape[0]} samples): {elapsed:7.0f} ms"
            )

        print("\nper plane, every navigator")
        for grid, count in [(64, 64), (96, 96), (128, 128)]:
            trajectory = spokes(grid, count)
            density = mm.estimate_density(trajectory, (grid, grid))
            image = phantom(grid)
            for coils in (1, 8):
                data = acquire(image, trajectory, coils)
                grid_ms = timed(
                    lambda d=data, t=trajectory, g=grid, w=density: (
                        mm.reconstruct_navigator(
                            d[None], t[None], (g, g), density=w[None]
                        )
                    )
                )
                plane = mm.reconstruct_navigator(
                    data[None], trajectory[None], (grid, grid), density=density[None]
                )[0]
                registration = mm.RigidRegistration()
                register_ms = timed(
                    lambda a=plane, b=plane, r=registration: r(
                        a, b, spacing=(1.0, 1.0)
                    ),
                    repeats=3,
                )
                samples = trajectory.shape[0] * coils
                print(
                    f"  {grid}^2, {coils} coil{'s' if coils > 1 else ' '}: "
                    f"grid {grid_ms:6.2f} ms ({samples / grid_ms / 1e3:5.1f} Msample/s)"
                    f"   register {register_ms:6.1f} ms"
                )

        print("\nper navigator, three planes, gridded and tracked")
        print(
            "  (a sinusoidal trace, not one pose repeated: a filter fed the "
            "same\n   measurement twice predicts motion that is not there, and "
            "the next\n   registration starts from a worse guess than it would "
            "in a scan)"
        )
        for grid, count in [(64, 64), (96, 96)]:
            trajectory = spokes(grid, count)
            density = mm.estimate_density(trajectory, (grid, grid))
            image = phantom(grid)
            stack = np.stack([trajectory] * 3)
            weights = np.stack([density] * 3)
            for coils in (1, 8):
                trace = [
                    np.stack(
                        [
                            acquire(
                                shift(image, (2.0 * np.sin(step / 3), 0.0), order=3),
                                trajectory,
                                coils,
                            )
                        ]
                        * 3
                    )
                    for step in range(NAVIGATORS)
                ]
                tracker = mm.NavigatorMotionTracker()
                start = time.perf_counter()
                for data in trace:
                    tracker.track(
                        list(
                            mm.reconstruct_navigator(
                                data, stack, (grid, grid), density=weights
                            )
                        ),
                        AXES,
                        dt=0.5,
                        spacing=2.0,
                    )
                elapsed = 1e3 * (time.perf_counter() - start) / len(trace)
                print(
                    f"  {grid}^2, {coils} coil{'s' if coils > 1 else ' '}: "
                    f"{elapsed:6.1f} ms per navigator"
                )


if __name__ == "__main__":
    main()
