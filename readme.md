# Intrusion Detection System

This project implements a machine learning-based network intrusion detection system using the CIC-IDS2017 dataset. It includes both model training and explainability components to help understand model decisions.

# Project Structure

NTRUTION_DETECTION
Preprocessing.py.  #preprocessing file 
model.py # Main model training and evaluation script 
explainability.py # Model explainability and interpretation 
model_outputs/ # Directory containing model outputs 
models/ # Saved models and preprocessors
reports/ # Evaluation metrics and explainability reports 
preprocessed_cicids2017_nozerocols.csv # Preprocessed dataset


# Data Preprocessing

The preprocessing pipeline includes:

1. Data Cleaning
   - Replace infinite values with NaN
   - Fill missing values with 0
   - Remove duplicate rows
   - Strip whitespace from column names

2. Label Processing
   - Convert target labels to binary classification
   - Create 'Label_Binary' column (0 = BENIGN, 1 = Malicious)

### Usage
```bash
python Preprocessing.py

## Model Training (model.py)

#Features
- Implements multiple machine learning models for network intrusion detection
- Handles class imbalance and performs feature selection
- Includes hyperparameter tuning and cross-validation
- Saves trained models and evaluation metrics

# Models Implemented
1. Logistic Regression
2. Extra Trees Classifier
3. XGBoost
4. LightGBM

# Usage
```bash
python model.py

#Model Performance
The best performing model and its metrics are saved in model_outputs/reports/metrics.json

#Explainability Analysis (explainability.py)

#Features
-Generates SHAP (SHapley Additive exPlanations) values for model interpretability
-Creates visualizations of feature importance
-Provides human-readable explanations for model decisions
-Generates an HTML report with interactive visualizations

#Outputs
- SHAP summary plots
- Feature importance rankings
- Local explanation examples
- Interactive HTML report

#Usage
bash
python explainability.py

# Requirements
Python 3.7+
# Required packages 
numpy
pandas
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
matplotlib
seaborn
joblib

#Getting Started

1. Clone the repository
2. Install dependencies listed under Requirements section
3. Place your dataset as preprocessed_cicids2017_nozerocols.csv in the project root
4. Run python model.py to train and evaluate models
5. Run python explainability.py to generate model explanations

