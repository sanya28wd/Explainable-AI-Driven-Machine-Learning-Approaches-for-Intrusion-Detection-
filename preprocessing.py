import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

import warnings

warnings.filterwarnings('ignore')

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = PROJECT_ROOT / "preprocessed_cicids2017_nozerocols.csv"

# List all CSV files in the current directory

files = [

   'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',

   'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',

   'Friday-WorkingHours-Morning.pcap_ISCX.csv',

   'Monday-WorkingHours.pcap_ISCX.csv',

   'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',

   'Thursday-WorkingHours-Morning-WebAttack.pcap_ISCX.csv',

   'Tuesday-WorkingHours.pcap_ISCX.csv',

   'Wednesday-workingHours.pcap_ISCX.csv'

]

 

existing_files = [f for f in files if os.path.exists(f)]

 

def load_and_sample_files(files, rows_per_file=100000):

   dfs = []

   for file in files:

       if os.path.exists(file):

           df = pd.read_csv(file, nrows=rows_per_file)

           dfs.append(df)

   combined_df = pd.concat(dfs, ignore_index=True)

   return combined_df

 

df = load_and_sample_files(existing_files, rows_per_file=100000)

 

# Clean up columns (fix extra spaces)

df.columns = df.columns.str.strip()

 

# Data cleaning

def clean_data(df):

   # Replace infinite values with NaN

   df = df.replace([np.inf, -np.inf], np.nan)

   # Fill missing values with 0

   df = df.fillna(0)

   # Remove duplicates

   df = df.drop_duplicates()

   return df

 

df = clean_data(df)

 

# Encode labels

def encode_labels(df):

   df['Label_Binary'] = (df['Label'] != 'BENIGN').astype(int)

   return df

 

df = encode_labels(df)

 

# Feature Engineering

def feature_engineering(df):

   drop_cols = [

       'Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP',

       'Source Port', 'Destination Port', 'Protocol', 'Timestamp', 'Label'

   ]

   drop_cols = [col for col in drop_cols if col in df.columns]

   if drop_cols:

       df = df.drop(columns=drop_cols)

   # Drop any remaining non-numeric columns except Label_Binary

   non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

   non_numeric = [col for col in non_numeric if col != 'Label_Binary']

   if non_numeric:

       df = df.drop(columns=non_numeric)

   return df

 

df = feature_engineering(df)

 

# Drop features with all 0 values

zero_cols = [col for col in df.columns if (df[col] == 0).all() and col != 'Label_Binary']

print("Columns with all zeros:", zero_cols)

df = df.drop(columns=zero_cols)

 

# Separate features and labels

X = df.drop(columns=['Label_Binary'])

y = df['Label_Binary']

 

# Train-Test Split

X_temp, X_test, y_temp, y_test = train_test_split(

   X, y, test_size=0.2, random_state=42, stratify=y

)

X_train, X_val, y_train, y_val = train_test_split(

   X_temp, y_temp, test_size=0.1875, random_state=42, stratify=y_temp

)

 

# Handle class imbalance with SMOTE (train set only)

smote = SMOTE(random_state=42, k_neighbors=5)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

 

# Feature Standardization

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_balanced)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

 

# Save preprocessed data

np.savez(

   'preprocessed_cicids2017.npz',

   X_train=X_train_scaled,

   y_train=y_train_balanced,

   X_val=X_val_scaled,

   y_val=y_val,

   X_test=X_test_scaled,

   y_test=y_test,

   feature_names=np.array(X.columns.tolist(), dtype=object)

)

 

# Save processed dataset and scaler for later use

import pickle

# Save the fully prepared dataset in CSV format for downstream model training.
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved processed dataset to: {OUTPUT_CSV}")

with open(PROJECT_ROOT / 'scaler_cicids2017.pkl', 'wb') as f:

   pickle.dump(scaler, f)

print(f"Saved scaler to: {PROJECT_ROOT / 'scaler_cicids2017.pkl'}")