# Explainable Intrusion Detection Lab

An explainable, multiclass network-intrusion-detection project built with the CIC-IDS2017 benchmark. It compares four machine-learning models, evaluates the selected model on a fixed held-out test set, and publishes recorded SHAP and LIME explanations for curated network flows.

## Live research explorer

[Open the interactive website](https://sanya28wd.github.io/Explainable-AI-Driven-Machine-Learning-Approaches-for-Intrusion-Detection-/)

The website is a static Stage 1 research explorer. It loads recorded, versioned results; it does not run live inference in the browser.

## Canonical experiment

The canonical run is `cicids2017-canonical-v1`.

- Dataset: eight CIC-IDS2017 CSV source files, checksum recorded in the experiment manifest.
- Sampling: seeded stratified reservoir sampling, capped at 100,000 rows per file.
- Labels: original multiclass attack labels are preserved.
- Leakage controls: identifiers, IPs, ports, timestamps, protocol, targets, and derived targets are excluded from features.
- Evaluation: fixed stratified 80/20 development/test split; five-fold CV on the development set; selection by mean macro F1.
- Explanations: SHAP is calculated on the fitted model's transformed feature space; LIME is generated for the curated held-out examples.

The first canonical run sampled 800,000 flows before cleaning and retained 14 classes. LightGBM was selected by cross-validation macro F1; see the exported evaluation file or interactive site for the exact metrics and per-class outcomes.

## Repository layout

```text
archive/                         Local CIC-IDS2017 CSV inputs (not committed)
research/run_canonical_experiment.py
research/validate_canonical_experiment.py
website/                         React/TypeScript research explorer
website/public/research/         Versioned static website exports
artifacts/canonical-v1/          Local model and experiment artifacts (not committed)
```

## Reproduce the canonical experiment

### 1. Get the dataset

Download CIC-IDS2017 from the [official dataset page](https://www.unb.ca/cic/datasets/ids-2017.html), extract the CSV files, and place them in `archive/`. The raw dataset is not included in this repository.

### 2. Set up Python

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-canonical.txt
```

### 3. Train and export

```bash
python research/run_canonical_experiment.py \
  --data-dir archive \
  --output-dir artifacts/canonical-v1 \
  --seed 42 \
  --max-rows-per-file 100000 \
  --cv-folds 5 \
  --test-size 0.2 \
  --examples 60 \
  --lime-samples 2000
```

The full run can take several hours because it compares four models across five folds and generates 60 local explanations.

### 4. Validate exports

```bash
python research/validate_canonical_experiment.py \
  --artifact-dir artifacts/canonical-v1
```

Expected output:

```text
Validation passed: 60 examples from cicids2017-canonical-v1
```

## Website development

```bash
cd website
npm ci
npm run dev
```

The website is deliberately separate from the offline Python training workflow. It ships compact JSON exports only; it does not ship the raw dataset, model binary, or complete prediction table.

## Limitations

This is a historical, controlled benchmark. Random held-out performance does not guarantee performance on new production networks. SHAP and LIME describe learned model behavior and should not be treated as evidence of causation.

## License and attribution

Use the CIC-IDS2017 dataset in accordance with its source terms and cite the Canadian Institute for Cybersecurity / University of New Brunswick dataset documentation in academic or professional work.
