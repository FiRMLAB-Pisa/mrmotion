"""The README figure: a navigator plane, and a pose tracked across a scan."""
import io, urllib.request
import matplotlib.pyplot as plt
import numpy as np
import finufft
from scipy.ndimage import rotate, shift
import mrmotion as mm

URL = ("https://raw.githubusercontent.com/lab-midas/"
       "ismrm-moco-workshop/master/data/brain_slice.npz")
with urllib.request.urlopen(URL) as r:
    brain = np.abs(np.load(io.BytesIO(r.read()))["arr_0"][20:236, :, 0])
brain /= brain.max()
N, GRID, SPOKES = brain.shape[0], 96, 96

angles = np.arange(SPOKES) * np.pi * (3 - np.sqrt(5))
radius = np.linspace(-0.5, 0.5, GRID, endpoint=False)
traj = np.stack([np.outer(np.cos(angles), radius),
                 np.outer(np.sin(angles), radius)], -1).reshape(-1, 2)
dens = mm.estimate_density(traj, (GRID, GRID))   # once, at start-up

def acquire(image):
    k = 2 * np.pi * traj * GRID / N
    return finufft.nufft2d2(np.ascontiguousarray(k[:, 0]), np.ascontiguousarray(k[:, 1]),
                            np.ascontiguousarray(image.astype(np.complex128)),
                            isign=-1, eps=1e-6, nthreads=1)

plane = mm.reconstruct_navigator(acquire(brain)[None, None], traj[None],
                                 (GRID, GRID), density=dens[None])[0]

AXES = [((1,0,0),(0,1,0)), ((0,1,0),(0,0,1)), ((1,0,0),(0,0,1))]
SPACING, DT, NAV = 2.0, 0.5, 20
anat = brain[::2, ::2]

def navigator(translation):
    out = []
    for u, v in AXES:
        u, v = np.array(u, float), np.array(v, float)
        out.append(shift(anat, (np.dot(translation, u) / SPACING,
                                np.dot(translation, v) / SPACING), order=3))
    return out

truth = np.array([np.array([2.0, -1.5, 3.0]) * np.sin(s / 5) for s in range(1, NAV + 1)])

def run(process_noise):
    rng = np.random.default_rng(0)
    t = mm.NavigatorMotionTracker(process_noise=process_noise)
    t.track(navigator(np.zeros(3)), AXES, spacing=SPACING)
    return np.asarray([t.track(navigator(p + rng.normal(0, 0.4, 3)), AXES,
                               dt=DT, spacing=SPACING).translation for p in truth])

measured, filtered = run(1e2), run(1e-1)
seconds = DT * np.arange(1, NAV + 1)

fig = plt.figure(figsize=(11, 2.7))
grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.4, 1.4, 1.4], wspace=0.3)
for column, (image, title) in enumerate(
        [(brain, f"imaging slice, {N}$^2$"), (plane, f"navigator, {GRID}$^2$")]):
    ax = fig.add_subplot(grid[0, column])
    ax.imshow(image, cmap="gray", vmax=np.percentile(image, 99.5)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
for axis, name in enumerate("xyz"):
    ax = fig.add_subplot(grid[0, 2 + axis])
    ax.plot(seconds, truth[:, axis], "k-", lw=1.2, label="head")
    ax.plot(seconds, measured[:, axis], ".", color="0.65", ms=4, label="measured")
    ax.plot(seconds, filtered[:, axis], "-", color="crimson", lw=1.2, label="filtered")
    ax.set_title(f"translation {name}", fontsize=9)
    ax.set_xlabel("s", fontsize=8); ax.tick_params(labelsize=7)
    if axis == 0:
        ax.set_ylabel("mm", fontsize=8); ax.legend(fontsize=6.5, frameon=False)
fig.savefig("examples/figures/tracking.png", dpi=160, bbox_inches="tight")
print("measured rms %.3f mm, filtered rms %.3f mm"
      % (np.sqrt(((measured - truth) ** 2).mean()),
         np.sqrt(((filtered - truth) ** 2).mean())))
