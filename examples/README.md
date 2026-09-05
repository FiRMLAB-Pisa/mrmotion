# Examples

The `.py` is the source: it runs as a script, lints with the rest of the
package, and reads as a diff. The `.ipynb` beside it is generated from it,
executed, and committed with its outputs, so it opens in Colab and runs top to
bottom. Both fetch a brain slice from the
[ISMRM motion-correction workshop](https://github.com/lab-midas/ismrm-moco-workshop)
(MIT) at run time — nothing is vendored here.

| example | shows | checked against |
|---|---|---|
| [`01-navigator`](01-navigator.ipynb) | `estimate_density`, `reconstruct_navigator` | the same registration on the full-resolution image |
| [`02-tracking`](02-tracking.ipynb) | `NavigatorMotionTracker`, `single_threaded` | a known pose, and the clock |

[`figures/make_tracking_figure.py`](figures/make_tracking_figure.py) draws the
README's figure, and is not one of the examples.

## Rebuilding

```bash
pip install -e .[dev] jupytext nbclient ipykernel
bash scripts/build_examples.sh
```

Every notebook is regenerated from its script and executed against the
interpreter the package is installed into. `--check` verifies the notebooks are
current without running them.
