# mrmotion

Rigid motion estimation and tracking for MRI: a navigator's planes
reconstructed, registered and carried across the scan by an extended Kalman
filter, with every library held to one thread.

[![Tests](https://github.com/FiRMLAB-Pisa/mrmotion/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/mrmotion/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/mrmotion/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/mrmotion)
[![PyPI](https://img.shields.io/pypi/v/mrmotion.svg)](https://pypi.org/project/mrmotion/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A navigator resolves six degrees of freedom out of images that are each only
two-dimensional. One plane sees the motion in its own plane — rotation about its
normal, translation along its two axes — and planes that between them span the
volume see all six. So the planes are registered against the first navigator
acquired, solved together for the pose that explains all of them, and carried
across the scan by a filter.

- **One thread, and it is checked rather than assumed** — prospective correction
  shares a machine with the reconstruction it steers, and owes it a pose before
  the next acquisition. A step that usually takes 20 ms and occasionally 200,
  because eight threads contended for a machine with none to spare, is worse
  than one that always takes 40. Four libraries take threads and each is told
  separately; `thread_counts()` reads them back so the tests assert them from
  inside the block
- **FINUFFT is told per call**, because it has no global to set — the count
  travels with the plan the navigator reconstruction builds, and there is a test
  that it arrives
- **The pose is solved as a rotation vector** and only then turned into Euler
  angles, which is exact. The approximation is upstream, and it is the one any
  2D registration rests on: a plane measures only the motion it can see
- **Nothing requires three planes, or requires them orthogonal** — what it
  requires is that the normals span the rotations and the in-plane axes span the
  translations

## Quick Start

```bash
pip install mrmotion
```

```python
import mrmotion as mm

# k-space along each plane -> plane images, FINUFFT on one thread
planes = mm.reconstruct_navigator(kspace, trajectory, shape=(64, 64))

# planes -> a filtered pose, registered against the first navigator seen
tracker = mm.NavigatorMotionTracker()
pose = tracker.track(planes, axes, dt=0.1, spacing=10.0)

# or hold something else of your own to one thread
with mm.single_threaded():
    ...
```

## Related Works

- **SimpleITK** — <https://simpleitk.org/>. The rigid registration each plane is
  measured with.
- **FINUFFT** — <https://finufft.readthedocs.io/>. What grids the navigator.
- **`lab-midas/ismrm-moco-workshop`** —
  <https://github.com/lab-midas/ismrm-moco-workshop>. Hands-on motion estimation
  and correction, and the basis for the example.
- Maclaren J, Herbst M, Speck O, Zaitsev M. *Prospective motion correction in
  brain imaging: a review.* Magn Reson Med 2013;69:621-636.

## Development

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh
pytest -q
```
