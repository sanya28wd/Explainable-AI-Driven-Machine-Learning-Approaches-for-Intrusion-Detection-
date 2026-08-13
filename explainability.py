from pathlib import Path
import json
import base64
import html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "preprocessed_cicids2017_nozerocols.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports" / "modeling outputs"
EXPLAIN_DIR = OUTPUT_DIR / "reports" / "explainability"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CORR_THRESHOLD = 0.95

NUM_TOP_FEATURES = 20
NUM_EXAMPLES = 3
MAX_SHAP_SAMPLES = 2000

np.random.seed(RANDOM_STATE)

metrics_path = REPORTS_DIR / "metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError(f"{metrics_path} not found. Run model.py first.")

with metrics_path.open("r") as f:
    metrics = json.load(f)

best_model_name = metrics["best_model"]

model_path = MODELS_DIR / f"{best_model_name}_pipeline.joblib"
if not model_path.exists():
    raise FileNotFoundError(f"{model_path} not found. Ensure model.py saved it.")

print("Loading model pipeline...")
best_model = joblib.load(model_path)
le = joblib.load(MODELS_DIR / "label_encoder.joblib")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
TARGET = "Label"
if TARGET not in df.columns:
    raise ValueError(f"Expected target column '{TARGET}' not found.")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

X_df = df.drop(columns=[TARGET])
y_raw = df[TARGET].astype(str)
y = le.transform(y_raw)

numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

if X_df[numeric_cols].isna().any().any():
    print("Imputing missing numeric values...")
    imp = SimpleImputer(strategy="median")
    X_df[numeric_cols] = imp.fit_transform(X_df[numeric_cols])

X_df[numeric_cols] = X_df[numeric_cols].clip(-1e12, 1e12)

non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric}")
    X_df.drop(columns=non_numeric, inplace=True)

nunique = X_df.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    print(f"Dropping {len(const_cols)} constant columns")
    X_df.drop(columns=const_cols, inplace=True)

corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
if to_drop:
    print(f"Dropping {len(to_drop)} highly correlated columns")
    X_df.drop(columns=to_drop, inplace=True)

feature_names = list(X_df.columns)
X = X_df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
class_names = list(le.classes_)


print("Computing SHAP global feature importance...")
model_for_shap = best_model
try:
    model_for_shap = best_model.named_steps["clf"]
except Exception:
    pass

n_samples = min(MAX_SHAP_SAMPLES, X_train.shape[0])
sample_idx = np.random.choice(np.arange(X_train.shape[0]), size=n_samples, replace=False)
X_shap = X_train[sample_idx]

explainer_global = shap.TreeExplainer(model_for_shap)
shap_values = explainer_global.shap_values(X_shap)

if isinstance(shap_values, list):
    sv_stack = np.stack(shap_values, axis=0)
    abs_mean = np.mean(np.abs(sv_stack), axis=(0, 1))
else:
    abs_mean = np.mean(np.abs(shap_values), axis=0)

abs_mean = np.asarray(abs_mean).ravel()
if len(abs_mean) != len(feature_names):
    min_len = min(len(feature_names), len(abs_mean))
    feature_names = feature_names[:min_len]
    abs_mean = abs_mean[:min_len]

shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": abs_mean}).sort_values(
    "mean_abs_shap", ascending=False
)
top_shap = shap_df.head(NUM_TOP_FEATURES).copy()
total_importance = top_shap["mean_abs_shap"].sum()
top_shap["percent"] = 0.0 if total_importance == 0 else 100 * top_shap["mean_abs_shap"] / total_importance


feature_explanations = {
    "act_data_pkt_fwd": "High volume of forward application data; check for rate-limiting and suspicious flows.",
    "Packet Length Variance": "Variable packet sizes; may indicate tunneling or obfuscation.",
    "Bwd Packet Length Max": "Large backward packets; monitor for exfiltration.",
    "Fwd Packet Length Max": "Oversized forward packets; enforce MTU limits.",
    "Total Length of Fwd Packets": "Large forward data transfers; combine with duration for anomaly detection.",
    "PSH Flag Count": "Abnormal PSH flags; possible flooding or custom attack tool.",
    "Active Std": "Variable active times; may indicate scanning or bursty DoS.",
    "Total Fwd Packets": "Excessive forward packets; consider throttling/blocking.",
    "Fwd Packet Length Mean": "Deviation from normal packet size; potential attack signal.",
    "Bwd Header Length": "Unusual header sizes; check for protocol abuse.",
    "DEFAULT": "Layered network security: IDS/IPS, firewalls, monitoring."
}

def get_explanation(feat: str) -> str:
    if feat in feature_explanations:
        return feature_explanations[feat]
    for key in feature_explanations:
        if key != "DEFAULT" and key.lower() in feat.lower():
            return feature_explanations[key]
    return feature_explanations["DEFAULT"]

top_shap["explanation"] = top_shap["feature"].apply(get_explanation)
top_shap.to_csv(EXPLAIN_DIR / "shap_top_features.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="mean_abs_shap", y="feature", data=top_shap, color="steelblue")
plt.title(f"Top {NUM_TOP_FEATURES} features by mean absolute SHAP")
plt.xlabel("mean |SHAP value|")
plt.tight_layout()
shap_global_img = EXPLAIN_DIR / "shap_global_top_features.png"
plt.savefig(shap_global_img, dpi=150)
plt.close()


background = X_train[np.random.choice(X_train.shape[0], size=min(500, X_train.shape[0]), replace=False)]
local_explainer = shap.Explainer(lambda x: best_model.predict_proba(x), background)

waterfall_files = []
num_waterfall = min(NUM_EXAMPLES, X_test.shape[0])

for i in range(num_waterfall):
    x_row = X_test[i : i + 1]
    try:
        shap_exp = local_explainer(x_row)
        pred_probs = best_model.predict_proba(x_row)[0]
        pred_class_idx = int(np.argmax(pred_probs))
        pred_label = class_names[pred_class_idx]
        pred_prob = float(pred_probs[pred_class_idx])

        if shap_exp.values.ndim == 3:  # multi-class
            single_exp = shap.Explanation(
                values=shap_exp.values[0, :, pred_class_idx],
                base_values=shap_exp.base_values[0, pred_class_idx],
                data=shap_exp.data[0],
                feature_names=feature_names,
            )
        else:
            single_exp = shap.Explanation(
                values=shap_exp.values[0],
                base_values=shap_exp.base_values[0],
                data=shap_exp.data[0],
                feature_names=feature_names,
            )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(single_exp, show=False, max_display=15)
        wf_path = EXPLAIN_DIR / f"shap_waterfall_sample_{i}.png"
        plt.tight_layout()
        plt.savefig(wf_path, dpi=150, bbox_inches="tight")
        plt.close()

        waterfall_files.append((wf_path.name, pred_label, pred_prob))
        print(f"Saved SHAP waterfall plot for sample {i} -> {wf_path.name}")

    except Exception as e:
        import traceback
        print(f"Failed to create SHAP waterfall plot for sample {i}: {e}")
        traceback.print_exc()


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8") if path.exists() else ""

html_lines = [
    "<html><head><meta charset='utf-8'>",
    f"<title>Explainability Report — Intrusion Detection ({best_model_name})</title>",
    "<style>",
    "body { font-family: Arial, sans-serif; margin: 20px; }",
    "h1, h2, h3 { font-family: Arial, sans-serif; }",
    "table { border-collapse: collapse; width: 100%; margin-top: 10px; }",
    "th, td { border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; }",
    "th { background-color: #f2f2f2; text-align: left; }",
    "</style>",
    "</head><body>",
    f"<h1>Explainability Report — Intrusion Detection ({best_model_name})</h1>",
    "<h2>Global Feature Importance (Top 20)</h2>",
]

if shap_global_img.exists():
    html_lines.append(
        f"<img src='data:image/png;base64,{img_b64(shap_global_img)}' "
        "style='max-width:900px; display:block; margin-bottom:20px;'>"
    )

html_lines.append("<h2>Top 20 Features — Percent Contribution & Explanation</h2>")
html_lines.append("<table>")
html_lines.append("<tr><th>Feature</th><th>Percent</th><th>Explanation</th></tr>")
for _, row in top_shap.iterrows():
    feat = html.escape(str(row["feature"]))
    pct = f"{row['percent']:.2f}%"
    explanation = html.escape(str(row.get("explanation", "")))
    html_lines.append(f"<tr><td>{feat}</td><td>{pct}</td><td>{explanation}</td></tr>")
html_lines.append("</table>")

html_lines.append("<h2>Local SHAP Waterfall Plots</h2>")
if waterfall_files:
    html_lines.append("<ul>")
    for fname, pred_label, pred_prob in waterfall_files:
        html_lines.append(
            f"<li>Predicted: {pred_label} ({pred_prob:.2f}) - "
            f"<img src='./{html.escape(fname)}' style='max-width:600px; display:block; margin-bottom:10px;'></li>"
        )
    html_lines.append("</ul>")
else:
    html_lines.append("<p>No SHAP waterfall plots were generated.</p>")

html_lines.append("<p>Report generated by <code>explainability.py</code>.</p>")
html_lines.append("</body></html>")

report_path = EXPLAIN_DIR / "explainability_report.html"
report_path.write_text("\n".join(html_lines), encoding="utf-8")
print("Explainability report saved to:", report_path)
print("Explainability outputs folder:", EXPLAIN_DIR.resolve())
