"""Separate commands for calibration, prediction, and final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import CalibratedCSC, calibrate
from .costs import atc_cost_matrix


def _load_data(path, *, targets=False):
    with np.load(path, allow_pickle=False) as data:
        required = {"scores", "label_names"} | ({"targets"} if targets else set())
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"input archive is missing: {', '.join(sorted(missing))}")
        result = {key: data[key] for key in required}
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate and apply severity-controlled prediction sets.")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("calibrate", help="fit a threshold using calibration data only")
    fit.add_argument("--data", required=True, type=Path, help="calibration NPZ: scores, targets, label_names")
    costs = fit.add_mutually_exclusive_group(required=True)
    costs.add_argument("--atc-level", type=int, choices=range(1, 6), help="construct ATC costs from label_names")
    costs.add_argument("--cost-matrix", type=Path, help="bounded K x K NPY matrix in label_names order")
    fit.add_argument("--alpha", type=float, default=0.25)
    fit.add_argument("--beta", type=float, default=0.0)
    fit.add_argument("--output", required=True, type=Path, help="output calibration JSON")
    for command, help_text in (("predict", "apply a saved threshold to scores"),
                               ("evaluate", "report metrics on a separate evaluation split")):
        action = sub.add_parser(command, help=help_text)
        action.add_argument("--model", required=True, type=Path)
        action.add_argument("--data", required=True, type=Path)
        action.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = [args.data]
        if args.command == "calibrate" and args.cost_matrix:
            inputs.append(args.cost_matrix)
        if args.command != "calibrate":
            inputs.append(args.model)
        if args.output.resolve() in [path.resolve() for path in inputs]:
            raise ValueError("output must be different from input files")
        data = _load_data(args.data, targets=args.command != "predict")
        if args.command == "calibrate":
            if args.atc_level:
                cost = atc_cost_matrix(data["label_names"], level=args.atc_level)
            else:
                cost = np.load(args.cost_matrix, allow_pickle=False)
            model = calibrate(data["scores"], data["targets"], cost, alpha=args.alpha,
                              beta=args.beta, label_names=data["label_names"])
            model.save(args.output)
            print(json.dumps({"lambda_hat": model.lambda_hat, "score_threshold": model.score_threshold,
                              "calibration_risk": model.calibration_risk,
                              "corrected_risk": model.corrected_risk,
                              "full_set_fallback": model.full_set_fallback}))
        else:
            model = CalibratedCSC.load(args.model)
            if args.command == "predict":
                selected = model.predict(data["scores"], label_names=data["label_names"])
                with args.output.open("wb") as handle:
                    np.savez_compressed(handle, selected=selected, label_names=np.asarray(model.label_names))
                print(f"Wrote {len(selected)} prediction sets to {args.output}")
            else:
                metrics = model.evaluate(data["scores"], data["targets"], label_names=data["label_names"])
                args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                print(json.dumps(metrics))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
