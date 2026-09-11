"""Train one traceable multiclass CIC-IDS2017 experiment and export website data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


EXPERIMENT_ID = "cicids2017-canonical-v1"
RANDOM_STATE = 42
LABEL_COLUMN = "Label"
EXCLUDED_COLUMNS = frozenset({
    "Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Destination Port", "Protocol", "Timestamp", "Label", "Label_Binary",
})


@dataclass(frozen=True)
class RunSettings:
    data_dir: str
    output_dir: str
    seed: int
    max_rows_per_file: int
    cv_folds: int
    test_size: float
    examples: int
    lime_samples: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-rows-per-file", type=int, required=True)
    parser.add_argument("--cv-folds", type=int, required=True)
    parser.add_argument("--test-size", type=float, required=True)
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--lime-samples", type=int, required=True)
    return parser.parse_args()


def normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_quotas(counts: pd.Series, maximum: int) -> dict[str, int]:
    if maximum < len(counts):
        raise ValueError("max_rows_per_file must be at least the number of labels in a source file")
    raw = counts / counts.sum() * maximum
    quotas = pd.Series(np.floor(raw.to_numpy()).astype(int), index=raw.index).clip(lower=1)
    remaining = maximum - int(quotas.sum())
    for label in raw.sub(quotas).sort_values(ascending=False).index[:remaining]:
        quotas.loc[label] += 1
    return {str(label): int(value) for label, value in quotas.items()}


def source_label_counts(path: Path) -> pd.Series:
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(path, usecols=lambda name: name.strip() == LABEL_COLUMN, chunksize=100_000):
        column = chunk.columns[0]
        labels = normalize_labels(chunk[column]).dropna()
        for label, count in labels.value_counts().items():
            counts[str(label)] = counts.get(str(label), 0) + int(count)
    if not counts:
        raise ValueError(f"No labels found in {path}")
    return pd.Series(counts).sort_index()


def stratified_reservoir(path: Path, maximum: int, seed: int) -> pd.DataFrame:
    quotas = class_quotas(source_label_counts(path), maximum)
    reservoirs: dict[str, list[dict[str, Any]]] = {label: [] for label in quotas}
    seen: dict[str, int] = {label: 0 for label in quotas}
    generator = np.random.default_rng(seed)
    for chunk in pd.read_csv(path, chunksize=25_000):
        chunk.columns = chunk.columns.str.strip()
        if LABEL_COLUMN not in chunk.columns:
            raise ValueError(f"{path} does not contain a {LABEL_COLUMN!r} column")
        chunk[LABEL_COLUMN] = normalize_labels(chunk[LABEL_COLUMN])
        for row in chunk.to_dict(orient="records"):
            label = str(row[LABEL_COLUMN])
            if label not in quotas:
                continue
            seen[label] += 1
            reservoir = reservoirs[label]
            quota = quotas[label]
            if len(reservoir) < quota:
                reservoir.append(row)
            else:
                replacement = int(generator.integers(0, seen[label]))
                if replacement < quota:
                    reservoir[replacement] = row
    sampled = [row for rows in reservoirs.values() for row in rows]
    if not sampled:
        raise ValueError(f"Sampling returned no rows from {path}")
    result = pd.DataFrame(sampled)
    result["_source_file"] = path.name
    return result


def load_data(data_dir: Path, maximum: int, seed: int) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    samples: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    for position, path in enumerate(files):
        samples.append(stratified_reservoir(path, maximum, seed + position))
        sources.append({"file": path.name, "sha256": sha256(path)})
    return pd.concat(samples, ignore_index=True), sources


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"Expected {LABEL_COLUMN!r} after loading data")
    labels = normalize_labels(frame[LABEL_COLUMN])
    valid = labels.notna() & labels.ne("")
    frame = frame.loc[valid]
    labels = labels.loc[valid]
    candidates = frame.drop(columns=[column for column in EXCLUDED_COLUMNS if column in frame.columns])
    numeric = candidates.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    before_duplicates = len(numeric)
    keep = ~numeric.duplicated(keep="first")
    numeric = numeric.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)
    counts = labels.value_counts()
    supported = counts[counts >= 2].index
    numeric = numeric.loc[labels.isin(supported)].reset_index(drop=True)
    labels = labels.loc[labels.isin(supported)].reset_index(drop=True)
    if numeric.empty or labels.nunique() < 2:
        raise ValueError("Cleaning left fewer than two represented classes")
    return numeric, labels, {"input_rows": len(frame), "duplicate_feature_rows_removed": before_duplicates - len(numeric)}


def pipelines(seed: int) -> dict[str, Pipeline]:
    def pipeline(model: Any) -> Pipeline:
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", model)])
    return {
        "Logistic Regression": pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "Extra Trees": pipeline(ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "XGBoost": pipeline(xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_estimators=300, learning_rate=.05, max_depth=6, subsample=.8, colsample_bytree=.8, random_state=seed, n_jobs=-1)),
        "LightGBM": pipeline(lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=.05, num_leaves=63, subsample=.8, colsample_bytree=.8, class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)),
    }


def json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def local_shap(classifier: Any, transformed: np.ndarray, class_index: int) -> tuple[np.ndarray, float]:
    explanation = shap.TreeExplainer(classifier)
    values = np.asarray(explanation.shap_values(transformed))
    base_values = np.asarray(explanation.expected_value)
    if values.ndim != 3:
        raise ValueError(f"Expected three-dimensional multiclass SHAP values, received shape {values.shape}")
    if values.shape[0] == transformed.shape[0] and values.shape[2] > class_index:
        return values[0, :, class_index], float(base_values[class_index])
    if values.shape[0] > class_index and values.shape[1] == transformed.shape[0]:
        return values[class_index, 0, :], float(base_values[class_index])
    raise ValueError(f"SHAP class dimensions do not match transformed data: {values.shape}")


def export_examples(output_dir: Path, pipeline: Pipeline, features: pd.DataFrame, labels: np.ndarray, predictions: np.ndarray, classes: list[str], example_count: int, lime_samples: int, seed: int) -> None:
    details_dir = output_dir / "examples"
    details_dir.mkdir(parents=True, exist_ok=True)
    probabilities = pipeline.predict_proba(features)
    candidates = pd.DataFrame({"actual": labels, "prediction": predictions})
    candidates["correct"] = candidates.actual == candidates.prediction
    selected_indices: list[int] = []
    for correct in [False, True]:
        for label in sorted(candidates.actual.unique()):
            matching = candidates.index[(candidates.correct == correct) & (candidates.actual == label)].tolist()
            if matching:
                selected_indices.append(matching[0])
    generator = np.random.default_rng(seed)
    remaining = [index for index in candidates.index if index not in selected_indices]
    if len(selected_indices) < example_count:
        selected_indices.extend(generator.choice(remaining, size=min(example_count - len(selected_indices), len(remaining)), replace=False).tolist())
    selected_indices = sorted(selected_indices[:example_count])
    transformed_train = pipeline.named_steps["scaler"].transform(pipeline.named_steps["imputer"].transform(features))
    lime = LimeTabularExplainer(transformed_train, feature_names=features.columns.tolist(), class_names=classes, mode="classification", random_state=seed)
    index: list[dict[str, Any]] = []
    classifier = pipeline.named_steps["classifier"]
    for number, row_index in enumerate(selected_indices):
        target = int(np.argmax(probabilities[row_index]))
        transformed_row = transformed_train[row_index : row_index + 1]
        shap_values, baseline = local_shap(classifier, transformed_row, target)
        lime_explanation = lime.explain_instance(transformed_row[0], classifier.predict_proba, labels=[target], num_features=min(15, features.shape[1]), num_samples=lime_samples)
        identifier = f"flow-{number:03d}"
        contributions = [{"feature": feature, "value": float(value)} for feature, value in zip(features.columns, shap_values)]
        lime_weights = [{"feature": features.columns[position], "weight": float(weight)} for position, weight in lime_explanation.as_map()[target]]
        payload = {"experiment_id": EXPERIMENT_ID, "id": identifier, "actual_label": str(labels[row_index]), "prediction": str(predictions[row_index]), "probabilities": {classes[position]: float(value) for position, value in enumerate(probabilities[row_index])}, "raw_features": {column: float(features.iloc[row_index][column]) for column in features.columns}, "shap": {"target_class": classes[target], "units": "raw model score", "baseline": baseline, "output": float(baseline + shap_values.sum()), "contributions": contributions}, "lime": {"target_class": classes[target], "score": float(lime_explanation.score), "weights": lime_weights}}
        json_write(details_dir / f"{identifier}.json", payload)
        index.append({"experiment_id": EXPERIMENT_ID, "id": identifier, "actual_label": payload["actual_label"], "prediction": payload["prediction"], "correct": payload["actual_label"] == payload["prediction"], "available_explanations": ["shap", "lime"]})
    json_write(details_dir / "index.json", {"experiment_id": EXPERIMENT_ID, "examples": index})


def main() -> None:
    args = arguments()
    settings = RunSettings(str(args.data_dir), str(args.output_dir), args.seed, args.max_rows_per_file, args.cv_folds, args.test_size, args.examples, args.lime_samples)
    if not 0 < args.test_size < 1:
        raise ValueError("test-size must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, sources = load_data(args.data_dir, args.max_rows_per_file, args.seed)
    features, labels_raw, cleaning = prepare_features(raw)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels_raw)
    classes = encoder.classes_.tolist()
    class_counts = pd.Series(labels_raw).value_counts().sort_index()
    if int(class_counts.min()) < args.cv_folds:
        raise ValueError(f"The smallest class has {int(class_counts.min())} rows; cv-folds={args.cv_folds} is not possible")
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=args.test_size, stratify=labels, random_state=args.seed)
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    results: list[dict[str, Any]] = []
    model_map = pipelines(args.seed)
    for name, model in model_map.items():
        cv = cross_validate(model, train_features, train_labels, cv=folds, scoring=scoring, n_jobs=1)
        results.append({"model": name, "macro_f1_mean": float(cv["test_macro_f1"].mean()), "macro_f1_std": float(cv["test_macro_f1"].std()), "balanced_accuracy_mean": float(cv["test_balanced_accuracy"].mean()), "balanced_accuracy_std": float(cv["test_balanced_accuracy"].std())})
    winner = max(results, key=lambda item: item["macro_f1_mean"])
    selected = model_map[str(winner["model"])]
    selected.fit(train_features, train_labels)
    predictions = selected.predict(test_features)
    evaluation = {"experiment_id": EXPERIMENT_ID, "selected_model": winner["model"], "cross_validation": results, "held_out_test": {"macro_f1": float(f1_score(test_labels, predictions, average="macro")), "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)), "classes": classes, "confusion_matrix": confusion_matrix(test_labels, predictions, labels=np.arange(len(classes))).tolist(), "per_class": classification_report(test_labels, predictions, target_names=classes, output_dict=True, zero_division=0)}}
    manifest = {"experiment_id": EXPERIMENT_ID, "settings": asdict(settings), "dataset": {"sources": sources, "class_counts_after_cleaning": class_counts.to_dict()}, "cleaning": cleaning, "feature_schema": features.columns.tolist(), "class_order": classes, "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "shap": shap.__version__}}
    joblib.dump(selected, args.output_dir / "model_pipeline.joblib")
    joblib.dump(encoder, args.output_dir / "label_encoder.joblib")
    json_write(args.output_dir / "manifest.json", manifest)
    json_write(args.output_dir / "evaluation.json", evaluation)
    json_write(args.output_dir / "feature_glossary.json", {"experiment_id": EXPERIMENT_ID, "features": [{"name": name, "unit": "dataset unit", "definition": "CIC-IDS2017 numeric flow feature."} for name in features.columns]})
    export_examples(args.output_dir, selected, test_features.reset_index(drop=True), encoder.inverse_transform(test_labels), encoder.inverse_transform(predictions), classes, args.examples, args.lime_samples, args.seed)
    print(f"Canonical experiment complete: {args.output_dir}")


if __name__ == "__main__":
    main()
