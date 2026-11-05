# Usage

## Input format

All arrays use a fixed vocabulary of `K` labels in the same column order.

| Name | Shape | Contents |
| --- | --- | --- |
| `scores` | `(n, K)` | Finite real scores in `[0, 1]`; larger means more likely. |
| `targets` | `(n, K)` | Multi-hot labels, exactly `0` or `1`; required for calibration and evaluation. |
| `label_names` | `(K,)` | Unique nonempty strings, in model output order. |
| Cost matrix | `(K, K)` | Entry `[y, k]` is the cost of matching true label `y` to prediction `k`. |

Scores need not sum to one. If the predictor emits logits, apply its fixed
score transformation, such as a sigmoid, before creating these arrays.
Use the same transformation for calibration and inference. The package rejects
NaNs, infinities, out-of-range scores, and soft targets.

Save Unicode label names, not a NumPy object array:

```python
np.savez_compressed(
    "calibration.npz",
    scores=calibration_scores,
    targets=calibration_targets,
    label_names=np.asarray(label_names, dtype=str),
)
np.savez_compressed(
    "new_scores.npz",
    scores=new_scores,
    label_names=np.asarray(label_names, dtype=str),
)
np.save("costs.npy", cost_matrix)
```

The CLI reads NPZ files with `allow_pickle=False`. It always checks label order
against the saved calibration file. Its output prediction archive contains
`selected`, a boolean `(n, K)` matrix, and `label_names`.

For existing prediction archives using `probabilities` and `groundtruth`, copy
those arrays to `scores` and `targets`, respectively, and supply the predictor's
ordered vocabulary as `label_names`. Do not infer the vocabulary by sorting
label names: its order must match the original model output columns.

## Python API

### `calibrate(scores, targets, cost_matrix, *, alpha=0.25, beta=0.0, label_names=None)`

Returns a `CalibratedCSC` object. Calibration requires at least one sample and
one label. `alpha` must be in `(0, 1)` and `beta` in `[0, 1)`. If names are
omitted, labels are named `"0"`, `"1"`, and so on in column order.

The result exposes:

| Attribute | Meaning |
| --- | --- |
| `lambda_hat` | Selected inclusiveness parameter. |
| `score_threshold` | Score cutoff `1 - lambda_hat`; ties are included. |
| `calibration_risk` | Mean calibration loss at the chosen threshold. |
| `corrected_risk` | `(n * calibration_risk + 1) / (n + 1)`. |
| `full_set_fallback` | Whether no grid point met the corrected criterion. |
| `n_calibration` | Number of calibration examples. |
| `alpha`, `beta`, `label_names`, `cost_matrix` | Saved configuration. |

Use `model.predict(scores, label_names=names)` for a boolean prediction matrix,
or `model.predict_labels(...)` for names. An empty inference batch is allowed
if its shape is `(0, K)`. `model.save(path)` and `CalibratedCSC.load(path)`
persist and restore the fitted configuration as JSON. The parent directory
must exist.

### `model.evaluate(scores, targets, *, label_names=None)`

Computes metrics on a nonempty evaluation set without recalibrating:

- `severity_mean`: mean sample loss using the fitted `beta` and cost matrix.
- `set_size_mean`: mean number of selected labels.
- `precision_macro`, `recall_macro`, `f1_macro`: calculate each metric per
  sample, then average. A zero denominator gives zero for that sample.

F1 is averaged per sample; it is not the harmonic mean of the reported mean
precision and recall. Empty ground-truth sets have severity zero. Consequently,
if empty targets occur, mean severity need not equal one minus macro recall
even when the off-diagonal costs are all one.

### `severity_loss(targets, selected, cost_matrix, *, beta=0.0)`

Returns one loss per sample. `selected` is a binary matrix rather than model
scores. This function also supports an empty batch. It is useful for evaluating
a fixed comparator without fitting another calibrator.

## Hierarchies and costs

`atc_cost_matrix(codes, level=4)` uses ATC prefixes of lengths 1, 3, 4, and 5.
The common root has depth zero and the modeled labels have depth four. It
checks code syntax, not whether a code is present in a particular ATC edition.
Resolve NDC/RxNorm mappings and multiple ATC assignments before constructing
the model vocabulary. The helper does not download or redistribute an ATC
database.

For another equal-depth tree, supply paths excluding the shared root:

```python
from csc import hierarchy_cost_matrix

cost = hierarchy_cost_matrix([
    ("group_A", "label_1"),
    ("group_A", "label_2"),
    ("group_B", "label_3"),
])
# [[0, 0.5, 1], [0.5, 0, 1], [1, 1, 0]]
```

Custom costs may be asymmetric and need not obey the triangle inequality.
They must be fixed before calibration, bounded in `[0, 1]`, and have an exactly
zero diagonal. The diagonal condition makes full-set severity zero. Costs are
not divided by their observed maximum: a cost matrix whose maximum is `0.5`
retains that scale. The loss of a nonempty target with an empty prediction is
still one. Construct branch-weighted costs externally and pass them through
the same API.

## Choosing a calibration partition

Use training data to fit the predictor, and a separate validation partition
for checkpoint or hyperparameter selection. Use the calibration partition only
for CSC's threshold. Apply the saved threshold to final evaluation data or new
inputs. Keep all observations from the same independent unit in one partition.
For longitudinal patient data, choose the target-visit rule before splitting
and keep patients disjoint across partitions.

Choose costs, `alpha`, and `beta` independently of calibration and test outcomes.
Inspecting several test results and then choosing a setting makes those results
exploratory. A smaller `alpha` generally requires larger prediction sets;
`alpha < 1 / (n + 1)` forces the full-set fallback. `beta` changes the loss
being controlled, rather than a confidence level across patients.
