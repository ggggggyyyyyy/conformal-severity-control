import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from csc import CalibratedCSC


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = np.asarray(["A10BA", "C09AA"])
        np.savez(self.root / "calibration.npz", scores=np.tile([.8, .1], (10, 1)),
                 targets=np.tile([1, 0], (10, 1)), label_names=self.names)

    def run_cli(self, *args, success=True):
        result = subprocess.run([sys.executable, "-m", "csc", *map(str, args)],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def fit(self):
        self.run_cli("calibrate", "--data", self.root / "calibration.npz", "--atc-level", 4,
                     "--alpha", .25, "--output", self.root / "model.json")

    def test_separate_calibration_prediction_and_evaluation(self):
        self.fit()
        path = self.root / "model.json"
        original_model = path.read_text()
        # Object targets cannot be loaded without pickle. Prediction must not read them.
        np.savez(self.root / "new.npz", scores=[[.9, .2]], label_names=self.names,
                 targets=np.array([{"unused": True}], dtype=object))
        self.run_cli("predict", "--model", path, "--data", self.root / "new.npz",
                     "--output", self.root / "sets.npz")
        with np.load(self.root / "sets.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["selected"], [[1, 0]])
            np.testing.assert_array_equal(data["label_names"], self.names)
        np.savez(self.root / "test.npz", scores=[[.9, .2]], targets=[[0, 1]], label_names=self.names)
        self.run_cli("evaluate", "--model", path, "--data", self.root / "test.npz",
                     "--output", self.root / "metrics.json")
        metrics = json.loads((self.root / "metrics.json").read_text())
        self.assertEqual(metrics["severity_mean"], 1)
        self.assertEqual(path.read_text(), original_model)

    def test_reordered_columns_are_rejected(self):
        self.fit()
        np.savez(self.root / "new.npz", scores=[[.2, .9]], label_names=self.names[::-1])
        result = self.run_cli("predict", "--model", self.root / "model.json", "--data",
                              self.root / "new.npz", "--output", self.root / "sets.npz", success=False)
        self.assertIn("column order", result.stderr)
        self.assertFalse((self.root / "sets.npz").exists())

    def test_custom_cost_and_no_targets_at_inference(self):
        np.save(self.root / "cost.npy", [[0, .5], [.75, 0]])
        self.run_cli("calibrate", "--data", self.root / "calibration.npz", "--cost-matrix",
                     self.root / "cost.npy", "--output", self.root / "custom.json")
        model = CalibratedCSC.load(self.root / "custom.json")
        self.assertEqual(model.cost_matrix[0, 1], .5)
        np.savez(self.root / "scores.npz", scores=[[.9, .2]], label_names=self.names)
        self.run_cli("predict", "--model", self.root / "custom.json", "--data",
                     self.root / "scores.npz", "--output", self.root / "sets.npz")

    def test_input_file_cannot_be_output(self):
        path = self.root / "calibration.npz"
        contents = path.read_bytes()
        self.run_cli("calibrate", "--data", path, "--atc-level", 4, "--output", path, success=False)
        self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
