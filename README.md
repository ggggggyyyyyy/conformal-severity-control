# Conformal Severity Control

CSC turns multi-label model scores into prediction sets calibrated to a chosen
mismatch tolerance. It works with a fixed predictor and a fixed label-to-label
cost matrix. The package implements average severity and exact within-sample
CVaR, with the finite-sample conformal risk correction.

An ATC helper supports medication labels; custom bounded costs support other
hierarchies and multi-label tasks. The only runtime dependency is NumPy.

## Install

Requires Python 3.9 or newer. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/quickstart.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
The example uses synthetic data and needs no downloads or credentials.

## Use with your model

Prepare a calibration partition that was not used to train or select the
predictor. Keep one row per independent unit, such as one target visit per
patient. Fix the label vocabulary, cost matrix, `alpha`, and `beta` before
calibration. Keep final evaluation data separate.

```python
import numpy as np
from csc import CalibratedCSC, atc_cost_matrix, calibrate

# Both files use the same ordered label_names; scores have shape (n, K).
# calibration.npz also contains a binary targets array with shape (n, K).
with np.load("calibration.npz", allow_pickle=False) as data:
    names = data["label_names"].tolist()
    model = calibrate(
        data["scores"],
        data["targets"],
        atc_cost_matrix(names, level=4),
        alpha=0.25,
        beta=0.75,  # Use 0.0 for average severity.
        label_names=names,
    )
model.save("csc.json")

# A later inference process needs scores, not target labels.
model = CalibratedCSC.load("csc.json")
with np.load("new_scores.npz", allow_pickle=False) as data:
    selected = model.predict(data["scores"], label_names=data["label_names"])
    predictions = model.predict_labels(data["scores"], label_names=data["label_names"])
```

`selected` is a boolean matrix in the model's label order. `predictions` is a
list of label-name lists. Passing `label_names` checks that columns have not
been reordered. The saved JSON contains the threshold, costs, label names, and
aggregate calibration statistics; it contains no sample scores, targets, or IDs.

For non-ATC labels, pass your own `K × K` cost matrix in place of
`atc_cost_matrix(...)`. Entries must lie in `[0, 1]` and the diagonal must be
zero. Costs are used as supplied, without automatic rescaling.

## Command-line workflow

These three stages can run separately. The calibration command never opens
evaluation data, and prediction does not require targets.

```bash
csc calibrate --data calibration.npz --atc-level 4 \
  --alpha 0.25 --beta 0.75 --output csc.json

csc predict --model csc.json --data new_scores.npz --output prediction_sets.npz

csc evaluate --model csc.json --data test.npz --output metrics.json
```

For custom costs, replace `--atc-level 4` with `--cost-matrix costs.npy`.
Each NPZ contains `scores` and `label_names`; calibration and evaluation also
require `targets`. See the [input format](docs/usage.md#input-format).
`python -m csc` is equivalent to `csc`.

To try these commands with generated inputs:

```bash
python examples/quickstart.py --output-dir demo_data
csc calibrate --data demo_data/calibration.npz --atc-level 4 \
  --alpha 0.25 --beta 0.75 --output demo_data/csc.json
csc predict --model demo_data/csc.json --data demo_data/new_scores.npz \
  --output demo_data/prediction_sets.npz
csc evaluate --model demo_data/csc.json --data demo_data/test.npz \
  --output demo_data/metrics.json
```

## What is controlled?

For each true label, CSC finds the smallest cost to any predicted label.
`beta=0` averages those costs within each sample. Larger `beta` emphasizes the
worst mismatches through exact CVaR; the paper's upper-tail setting is `0.75`.
`alpha` bounds the expected sample-level loss under the method's assumptions.

The guarantee averages over calibration samples and a new exchangeable sample.
It is not a per-patient guarantee or a guarantee that every observed test mean
will be below `alpha`. Larger prediction sets can reduce this one-sided loss
while adding false positives. ATC distance measures taxonomy mismatch; it does
not establish clinical safety or drug interchangeability.

See [usage and API details](docs/usage.md) and the
[method and paper settings](docs/method.md).

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip install build
python -m build
```

Tests cover the fractional CVaR definition, finite-sample correction, threshold
ties, empty sets, full-set fallback, input validation, serialization, and the
separate command-line stages.

## Citation and license

This package accompanies *Severity-Controlled Prediction Sets for Medication
Recommendation* (2026), by Yu Gu, Zijun Yu, Chi-Kuang Yeh, Xinyu Wang, and
Ziyang Song. See [CITATION.cff](CITATION.cff).

The source code is available under the [MIT License](LICENSE). Patient data,
trained predictors, and third-party taxonomy databases are not distributed
with this package.
