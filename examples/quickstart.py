"""Calibrate a fixed synthetic predictor and evaluate on independent examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from csc import CalibratedCSC, atc_cost_matrix, calibrate


def sample(rng, n, k):
    scores = rng.beta(2.0, 5.0, size=(n, k))
    targets = (rng.random((n, k)) < scores).astype(np.int8)
    return scores, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="optionally save demo inputs and calibration")
    args = parser.parse_args()
    names = np.asarray(["A10BA", "A10BB", "C09AA", "C09CA", "B01AC", "N02BE"])
    calibration_seed, test_seed = np.random.SeedSequence(2026).spawn(2)
    calibration_scores, calibration_targets = sample(np.random.default_rng(calibration_seed), 500, len(names))
    test_scores, test_targets = sample(np.random.default_rng(test_seed), 200, len(names))
    model = calibrate(calibration_scores, calibration_targets, atc_cost_matrix(names),
                      alpha=0.25, beta=0.75, label_names=names)
    print("Synthetic example; these are not results from patient data.")
    print(f"lambda={model.lambda_hat:.4f}; score cutoff={model.score_threshold:.4f}")
    print(json.dumps(model.evaluate(test_scores, test_targets), indent=2))
    print("First prediction:", model.predict_labels(test_scores[:1])[0])
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir / "calibration.npz", scores=calibration_scores,
                            targets=calibration_targets, label_names=names)
        np.savez_compressed(args.output_dir / "test.npz", scores=test_scores,
                            targets=test_targets, label_names=names)
        np.savez_compressed(args.output_dir / "new_scores.npz", scores=test_scores, label_names=names)
        path = args.output_dir / "csc.json"
        model.save(path)
        restored = CalibratedCSC.load(path)
        np.testing.assert_array_equal(model.predict(test_scores), restored.predict(test_scores))
        print(f"Saved demo files to {args.output_dir}")


if __name__ == "__main__":
    main()
