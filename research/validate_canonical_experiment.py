"""Validate that canonical website exports belong to one experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir
    manifest = read_json(root / "manifest.json")
    evaluation = read_json(root / "evaluation.json")
    index = read_json(root / "examples" / "index.json")
    experiment_id = manifest["experiment_id"]
    if evaluation["experiment_id"] != experiment_id or index["experiment_id"] != experiment_id:
        raise ValueError("Manifest, evaluation, and example index use different experiment identifiers")
    classes = manifest["class_order"]
    matrix = evaluation["held_out_test"]["confusion_matrix"]
    if len(matrix) != len(classes) or any(len(row) != len(classes) for row in matrix):
        raise ValueError("Confusion matrix dimensions do not match manifest class order")
    for entry in index["examples"]:
        detail = read_json(root / "examples" / f"{entry['id']}.json")
        if detail["experiment_id"] != experiment_id:
            raise ValueError(f"Example {entry['id']} uses a different experiment identifier")
        if set(detail["probabilities"]) != set(classes):
            raise ValueError(f"Example {entry['id']} probabilities do not match class order")
        contributions = sum(item["value"] for item in detail["shap"]["contributions"])
        reconstructed = detail["shap"]["baseline"] + contributions
        if abs(reconstructed - detail["shap"]["output"]) > 1e-6:
            raise ValueError(f"Example {entry['id']} SHAP values do not sum to its declared output")
    print(f"Validation passed: {len(index['examples'])} examples from {experiment_id}")


if __name__ == "__main__":
    main()
