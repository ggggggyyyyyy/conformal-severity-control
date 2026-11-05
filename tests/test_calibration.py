import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from csc import CalibratedCSC, atc_cost_matrix, calibrate, hierarchy_cost_matrix, severity_loss


def definition_loss(targets, selected, cost, beta):
    """Direct CVaR minimization, independent of the sorted-tail implementation."""
    result = []
    for truth, prediction in zip(targets, selected):
        true_indices = np.flatnonzero(truth)
        predicted_indices = np.flatnonzero(prediction)
        if not len(true_indices):
            result.append(0.0)
            continue
        z = np.ones(len(true_indices)) if not len(predicted_indices) else np.array([
            min(cost[y, k] for k in predicted_indices) for y in true_indices
        ])
        candidates = np.unique(np.r_[0, z, 1])
        result.append(min(t + np.maximum(z - t, 0).mean() / (1 - beta) for t in candidates))
    return np.array(result)


class CostTests(unittest.TestCase):
    def test_atc_depth_and_label_order(self):
        codes = ["C09CA", "A10BA", "C09AA", "C09AB"]
        expected = [[0, 1, .5, .5], [1, 0, 1, 1], [.5, 1, 0, .25], [.5, 1, .25, 0]]
        np.testing.assert_array_equal(atc_cost_matrix(codes), expected)

    def test_generic_tree(self):
        cost = hierarchy_cost_matrix([("A", "x"), ("A", "y"), ("B", "z")])
        np.testing.assert_array_equal(cost, [[0, .5, 1], [.5, 0, 1], [1, 1, 0]])

    def test_invalid_hierarchies(self):
        for codes in ([], ["A10BA", "A10BA"], ["A10BA", "C09AA01"], ["a10ba"]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                atc_cost_matrix(codes)
        for paths in ([], [()], [("A",), ("B", "x")], [("A",), ("A",)]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                hierarchy_cost_matrix(paths)


class LossTests(unittest.TestCase):
    def test_fractional_tail_not_rounded_mean(self):
        cost = np.ones((4, 4)) - np.eye(4)
        cost[:3, 3] = [1, .5, 0]
        truth = [[1, 1, 1, 0]]
        selected = [[0, 0, 0, 1]]
        self.assertAlmostEqual(severity_loss(truth, selected, cost, beta=.5)[0], 5 / 6)
        self.assertAlmostEqual(severity_loss(truth, selected, cost, beta=0)[0], .5)
        self.assertEqual(severity_loss(truth, selected, cost, beta=.99)[0], 1)

    def test_empty_and_full_sets(self):
        cost = 1 - np.eye(3)
        truth = [[1, 0, 1], [0, 0, 0]]
        for beta in (0, .75):
            np.testing.assert_array_equal(severity_loss(truth, np.zeros((2, 3)), cost, beta=beta), [1, 0])
            np.testing.assert_array_equal(severity_loss(truth, np.ones((2, 3)), cost, beta=beta), [0, 0])

    def test_cost_scale_is_preserved(self):
        np.testing.assert_array_equal(severity_loss([[1, 0]], [[0, 1]], [[0, .2], [.7, 0]]), [.2])

    def test_loss_matches_variational_definition(self):
        rng = np.random.default_rng(37)
        cost = rng.random((8, 8))
        np.fill_diagonal(cost, 0)
        truth = rng.integers(0, 2, (21, 8))
        prediction = rng.integers(0, 2, (21, 8))
        truth[0] = 0
        prediction[1] = 0
        for beta in (0, .1, .35, .75, .999):
            with self.subTest(beta=beta):
                np.testing.assert_allclose(severity_loss(truth, prediction, cost, beta=beta),
                                           definition_loss(truth, prediction, cost, beta), atol=1e-13)


class CalibrationTests(unittest.TestCase):
    def test_threshold_includes_ties(self):
        model = calibrate([[.75], [.75], [.75]], [[1], [1], [1]], [[0]], alpha=.25)
        self.assertEqual(model.lambda_hat, .25)
        np.testing.assert_array_equal(model.predict([[.75], [.749], [1], [0]]), [[1], [0], [1], [0]])

    def test_finite_sample_correction_changes_selection(self):
        scores = [[.1], [.4], [.7], [.9]]
        model = calibrate(scores, np.ones((4, 1)), [[0]], alpha=.3)
        self.assertAlmostEqual(model.lambda_hat, .9)
        self.assertEqual(model.calibration_risk, 0)
        self.assertEqual(model.corrected_risk, .2)
        # At lambda=.6, the uncorrected mean .25 passes, but (1+1)/5 does not.
        self.assertEqual(severity_loss(np.ones((4, 1)), np.array(scores) >= .4, [[0]]).mean(), .25)

    def test_full_set_fallback_and_boundary(self):
        for alpha, fallback in ((.1, True), (1 / 3, False)):
            model = calibrate([[0, 0], [0, 0]], [[1, 0], [0, 1]], 1 - np.eye(2), alpha=alpha)
            self.assertEqual(model.full_set_fallback, fallback)
            self.assertEqual(model.lambda_hat, 1)
            np.testing.assert_array_equal(model.predict([[0, 0]]), [[1, 1]])
            self.assertEqual(model.calibration_risk, 0)
            self.assertEqual(model.corrected_risk, 1 / 3)

    def test_smallest_grid_point_can_be_zero(self):
        model = calibrate([[1], [1], [1]], [[1], [1], [1]], [[0]], alpha=.25)
        self.assertEqual(model.lambda_hat, 0)

    def test_grid_search_matches_direct_definition(self):
        rng = np.random.default_rng(62)
        scores = rng.integers(0, 9, (17, 6)) / 8
        truth = rng.integers(0, 2, (17, 6))
        cost = rng.random((6, 6))
        np.fill_diagonal(cost, 0)
        truth[0] = 0
        grid = np.linspace(0, 1, 401)
        for beta in (0, .35, .75, .99):
            means = np.array([definition_loss(truth, scores >= 1 - lam, cost, beta).mean() for lam in grid])
            self.assertTrue(np.all(np.diff(means) <= 1e-14))
            for alpha in (.01, .15, .3, .7):
                with self.subTest(beta=beta, alpha=alpha):
                    feasible = np.flatnonzero((17 * means + 1) / 18 <= alpha)
                    expected_index = int(feasible[0]) if len(feasible) else 400
                    model = calibrate(scores, truth, cost, alpha=alpha, beta=beta)
                    self.assertEqual(model.grid_index, expected_index)
                    self.assertAlmostEqual(model.calibration_risk, means[expected_index])

    def test_evaluation_uses_sample_macro_metrics(self):
        model = calibrate(np.full((9, 2), .5), np.ones((9, 2)), 1 - np.eye(2), alpha=.1)
        metrics = model.evaluate([[1, 0], [1, 1]], [[1, 0], [0, 1]])
        self.assertEqual(metrics["precision_macro"], .75)
        self.assertEqual(metrics["recall_macro"], 1)
        self.assertAlmostEqual(metrics["f1_macro"], 5 / 6)
        self.assertEqual(metrics["set_size_mean"], 1.5)

    def test_evaluation_does_not_change_calibration(self):
        model = calibrate([[.8, .1]] * 10, [[1, 0]] * 10, 1 - np.eye(2))
        initial = (model.lambda_hat, model.calibration_risk, model.corrected_risk)
        model.evaluate([[.4, .8]], [[0, 1]])
        model.evaluate([[.4, .8]], [[1, 0]])
        self.assertEqual((model.lambda_hat, model.calibration_risk, model.corrected_risk), initial)
        with self.assertRaises(FrozenInstanceError):
            model.alpha = .8

    def test_roundtrip_and_label_order(self):
        cost = 1 - np.eye(2)
        model = calibrate([[.75, 0]] * 3, [[1, 0]] * 3, cost, label_names=["a", "b"])
        cost[0, 1] = .5
        self.assertEqual(model.cost_matrix[0, 1], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "csc.json"
            model.save(path)
            restored = CalibratedCSC.load(path)
            np.testing.assert_array_equal(model.predict([[.8, .1]]), restored.predict([[.8, .1]]))
            self.assertEqual(restored.predict_labels([[.8, .1]]), [["a"]])
            self.assertFalse({"scores", "targets", "patient_ids"}.intersection(json.loads(path.read_text())))
            with self.assertRaisesRegex(ValueError, "column order"):
                restored.predict([[.8, .1]], label_names=["b", "a"])
            payload = json.loads(path.read_text())
            payload["format_version"] = 100
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "version"):
                CalibratedCSC.load(path)

    def test_empty_inference_batch(self):
        model = calibrate([[.5]], [[1]], [[0]])
        self.assertEqual(model.predict(np.empty((0, 1))).shape, (0, 1))
        self.assertEqual(model.predict_labels(np.empty((0, 1))), [])


class ValidationTests(unittest.TestCase):
    def test_invalid_inputs(self):
        valid = dict(scores=[[.5, .5]], targets=[[1, 0]], cost_matrix=1 - np.eye(2))
        cases = [
            {"scores": [[np.nan, 0]]}, {"scores": [[np.inf, 0]]},
            {"scores": [[1.1, 0]]}, {"scores": [[-.1, 0]]}, {"scores": [[1j, 0]]},
            {"scores": [.5, .5]}, {"targets": [[.5, 0]]},
            {"targets": [[np.nan, 0]]}, {"targets": [[1]]},
            {"scores": np.empty((0, 2)), "targets": np.empty((0, 2))},
            {"scores": np.empty((1, 0)), "targets": np.empty((1, 0))},
            {"cost_matrix": [[0, 1]]}, {"cost_matrix": [[0, 2], [1, 0]]},
            {"cost_matrix": [[0, -1], [1, 0]]}, {"cost_matrix": [[.1, 1], [1, 0]]},
            {"cost_matrix": [[0, np.nan], [1, 0]]},
            {"alpha": 0}, {"alpha": 1}, {"alpha": np.nan},
            {"beta": -1}, {"beta": 1}, {"beta": np.nan},
            {"label_names": ["x", "x"]}, {"label_names": ["x"]},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                calibrate(**dict(valid, **case))

    def test_all_zero_cost_is_valid(self):
        model = calibrate([[.5, 0]] * 10, [[0, 1]] * 10, np.zeros((2, 2)), alpha=.1)
        self.assertEqual(model.lambda_hat, .5)
        self.assertEqual(model.calibration_risk, 0)


if __name__ == "__main__":
    unittest.main()
