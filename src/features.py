# src/features.py
"""
Stage 2: Feature Engineering
- Encodes categorical variables (City, Property Type)
- Splits into train/test sets
- Saves as numpy arrays for faster loading
"""
import pandas as pd
import numpy as np
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def create_features(in_csv, out_dir, test_size, random_state):
    """Engineer features and split data"""
    
    print(f"Loading cleaned data from {in_csv}...")
    df = pd.read_csv(in_csv)
    
    # Separate features and target
    # Assuming 'price' is the target column (adjust if different)
    if 'price' in df.columns:
        target_col = 'price'
    elif 'Price' in df.columns:
        target_col = 'Price'
    else:
        # Find the price column (common variations)
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        if price_cols:
            target_col = price_cols[0]
        else:
            raise ValueError("Cannot find price column in dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical columns
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Save arrays
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train.values)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test.values)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test.values)
    
    # Save label encoders and feature names
    os.makedirs("models", exist_ok=True)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(list(X.columns), "models/model_features.pkl")
    
    # Create field map (for Flask form)
    feature_field_map = {col: col for col in X.columns}
    joblib.dump(feature_field_map, "models/feature_field_map.pkl")
    
    print("Feature engineering complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    create_features(args.in_csv, args.out_dir, args.test_size, args.random_state)