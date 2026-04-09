# 1. IMPORT LIBRARIES

# Standard / utility libraries
import os
import json
from pathlib import Path
import time
import joblib

# Data and math
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn ML utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Imbalanced handling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Gradient boosting
import xgboost as xgb
import lightgbm as lgb

# 2. PATHS AND GLOBAL SETTINGS

PROJECT_ROOT = Path("/Users/sanyawadhawan/Desktop/INTRUTION_DETECTION")
DATA_PATH = PROJECT_ROOT / "preprocessed_cicids2017_nozerocols.csv"
OUTPUT_DIR = PROJECT_ROOT / "model_outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

for p in (OUTPUT_DIR, MODELS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CORR_THRESHOLD = 0.95

np.random.seed(RANDOM_STATE)

# 3. LOAD DATA AND DEFINE TARGET

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

TARGET = "Label"
if TARGET not in df.columns:
    raise ValueError(f"Expected target column '{TARGET}' not found!")

# Quick class distribution
print("Class distribution (raw):")
print(df[TARGET].value_counts())

# Handle infinities and missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_df = df.drop(columns=[TARGET])
y_raw = df[TARGET].astype(str)

numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
if X_df[numeric_cols].isna().any().any():
    print("Imputing numeric missing values (median)...")
    imp = SimpleImputer(strategy="median")
    X_df[numeric_cols] = imp.fit_transform(X_df[numeric_cols])

# Clip extreme values
X_df[numeric_cols] = X_df[numeric_cols].clip(-1e12, 1e12)

# Drop non-numeric columns
non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("Dropping non-numeric columns (expected none):", non_numeric)
    X_df.drop(columns=non_numeric, inplace=True)

# Feature pruning
nunique = X_df.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    X_df.drop(columns=const_cols, inplace=True)
    print(f"Dropped {len(const_cols)} constant columns")

corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
if to_drop:
    X_df.drop(columns=to_drop, inplace=True)
    print(f"Dropped {len(to_drop)} highly correlated columns")

print("Remaining features:", X_df.shape[1])

# Encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
target_names = le.classes_.tolist()
X = X_df.values

# 4. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 5. PREPROCESSING PIPELINE

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocess = ColumnTransformer([
    ("num", numeric_transformer, list(range(X.shape[1]))),
], remainder="drop")

# 6. DEFINE MODELS

pipe_logreg = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=500, solver="saga", multi_class="auto",
                               class_weight="balanced", random_state=RANDOM_STATE))
])

pipe_extratrees = Pipeline([
    ("preprocess", preprocess),
    ("clf", ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1))
])

pipe_xgb = Pipeline([
    ("preprocess", preprocess),
    ("clf", xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        use_label_encoder=False
    ))
])

pipe_lgbm = Pipeline([
    ("preprocess", preprocess),
    ("clf", lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

models = [
    (pipe_logreg, "LogisticRegression"),
    (pipe_extratrees, "ExtraTrees"),
    (pipe_xgb, "XGBoost"),
    (pipe_lgbm, "LightGBM")
]

# 7. CROSS-VALIDATION

scoring = {
    "macro_f1": make_scorer(f1_score, average="macro"),
    "bal_acc": make_scorer(balanced_accuracy_score)
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def evaluate_with_cv(model, name):
    print(f"\nRunning CV for model: {name}")
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    summary = {
        "model": name,
        "macro_f1_mean": float(np.mean(res["test_macro_f1"])),
        "macro_f1_std": float(np.std(res["test_macro_f1"])),
        "bal_acc_mean": float(np.mean(res["test_bal_acc"])),
        "bal_acc_std": float(np.std(res["test_bal_acc"])),
        "train_macro_f1_mean": float(np.mean(res["train_macro_f1"])),
        "train_bal_acc_mean": float(np.mean(res["train_bal_acc"]))
    }
    print("CV summary:", summary)
    return summary

cv_results = []
for mdl, name in models:
    try:
        cv_results.append(evaluate_with_cv(mdl, name))
    except Exception as e:
        print(f"Error during CV for {name}: {e}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(REPORTS_DIR / "cv_summary.csv", index=False)
print("\nCV results saved:", REPORTS_DIR / "cv_summary.csv")

# 8. SELECT BEST MODEL AND TRAIN ON FULL DATA

best_row = cv_df.sort_values("macro_f1_mean", ascending=False).iloc[0]
best_model_name = best_row["model"]
print("\nSelected best model:", best_model_name)

model_map = {name: mdl for mdl, name in models}
best_model = model_map[best_model_name]

best_model.fit(X_train, y_train)

# Feature importance for tree-based models
if best_model_name in ["ExtraTrees", "XGBoost", "LightGBM"]:
    feat_imp = best_model.named_steps['clf'].feature_importances_
    feat_df = pd.DataFrame({"feature": X_df.columns, "importance": feat_imp}).sort_values("importance", ascending=False).head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=feat_df, palette="viridis")
    plt.title(f"Top 20 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"feature_importance_{best_model_name}.png", dpi=150)
    plt.show()

# 9. EVALUATE ON TEST SET

y_pred = best_model.predict(X_test)
try:
    y_proba = best_model.predict_proba(X_test)
except Exception:
    y_proba = None

test_bal_acc = balanced_accuracy_score(y_test, y_pred)
test_macro_f1 = f1_score(y_test, y_pred, average="macro")
print("\nTest Balanced Accuracy:", test_bal_acc)
print("Test Macro F1:", test_macro_f1)
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
plt.show()

# 10. SAVE ARTIFACTS

joblib.dump(best_model, MODELS_DIR / f"{best_model_name}_pipeline.joblib")
cv_df.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
metrics = {
    "best_model": best_model_name,
    "test_balanced_accuracy": test_bal_acc,
    "test_macro_f1": test_macro_f1,
    "classes": target_names
}
with open(REPORTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

if y_proba is not None:
    proba_df = pd.DataFrame(y_proba)
    proba_df.to_parquet(REPORTS_DIR / "test_proba.parquet")

