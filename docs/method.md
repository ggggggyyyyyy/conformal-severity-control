# Method and paper settings

For scores `f_k(x)` in `[0, 1]`, the nested prediction set is

```text
C_lambda(x) = {k : f_k(x) >= 1 - lambda}.
```

For true label `y`, the closest-match severity is the minimum of `cost[y, k]`
over selected labels `k`, or one if the prediction set is empty. A sample with
no true labels has loss zero. With a nonempty target, let `z` contain the
closest-match severities of its `m` true labels.

For `beta=0`, the loss is `mean(z)`. For `0 <= beta < 1`, it is

```text
S_beta = min over t in [0, 1] of t + mean(max(z - t, 0)) / (1 - beta).
```

The implementation sorts severities in descending order and averages the
largest `(1 - beta) * m` values with fractional weight at the boundary. It does
not round the tail size up and then take an unweighted mean. For example,
`z = [1, 0.5, 0]` and `beta=0.5` gives `(1 + 0.5 * 0.5) / 1.5 = 5/6`.

CSC evaluates the fixed grid `np.linspace(0, 1, 401)` and selects its smallest
feasible value:

```text
(sum of n calibration losses + 1) / (n + 1) <= alpha.
```

If the grid has no feasible value, CSC uses `lambda=1`. This can happen only
when `alpha < 1 / (n + 1)`, given the required bounded costs and zero diagonal.
The fallback includes every label and has zero severity, although its corrected
calibration criterion is still greater than `alpha`. The saved flag distinguishes
this case from a feasible grid choice.

Under a fixed predictor, fixed loss, and exchangeable calibration/new samples,
the conformal risk result bounds expected severity by `alpha`. The expectation
includes both the calibration partition and the new sample. It does not give
a high-probability bound on a realized test-set mean, a guarantee under temporal
or site shift, or a bound on false positives.

## Correspondence to the paper

| Component | Package setting |
| --- | --- |
| CSC-Avg | `beta=0.0` |
| CSC-Upper | `beta=0.75` |
| ATC-4 equal-edge severity | `atc_cost_matrix(codes, level=4)` |
| Calibration grid | 401 equally spaced values, including 0 and 1 |
| Finite-sample correction | Add one to the loss sum; divide by `n + 1` |
| Inference | Include scores `>= 1 - lambda_hat` |
| FNR loss for nonempty targets | Custom costs `1 - np.eye(K)`, `beta=0` |
| Alternative bounded costs | Supply a fixed matrix; recalibrate for that matrix |

The package implements the post-hoc procedure from *Severity-Controlled
Prediction Sets for Medication Recommendation*. It starts from model scores
and labels. It does not include the patient data, processed patient prediction
arrays, raw EHR preprocessing, or trained predictors needed to reproduce the
paper's numerical tables from scratch. The synthetic example demonstrates the
API and is not a replication of the medication experiments.

When applying the paper settings to authorized data, preserve the predictor's
label order and use its original patient-level calibration and test partitions.
Do not use test labels to select thresholds, loss definitions, or operating
points. Record predictor, vocabulary, taxonomy version, split construction,
and cost scaling alongside results. Any upstream uncertainty about drug-to-ATC
mapping must be resolved in that preprocessing pipeline.

## References

- Yu Gu, Zijun Yu, Chi-Kuang Yeh, Xinyu Wang, and Ziyang Song.
  *Severity-Controlled Prediction Sets for Medication Recommendation*. 2026.
- Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal
  Schuster. *Conformal Risk Control*. ICLR, 2024.
- R. Tyrrell Rockafellar and Stanislav Uryasev. *Optimization of Conditional
  Value-at-Risk*. Journal of Risk, 2000.
