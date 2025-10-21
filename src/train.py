# src/train.py
"""
Stage 3: Model Training
- Trains a Random Forest model
- Saves trained model
"""
import numpy as np
import yaml
import argparse
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(data_dir, model_out, n_estimators, max_depth, random_state):
    """Train Random Forest model"""
    
    print("Loading training data...")
    X_train = np.load(f"{data_dir}/X_train.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    
    print(f"Training Random Forest with {n_estimators} trees...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    print(f"Saving model to {model_out}...")
    joblib.dump(model, model_out)
    
    # Also save to models/ for Flask app
    joblib.dump(model, "models/house_price_model.pkl")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_out", default="model.pkl")
    args = parser.parse_args()
    
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    train_params = params["train"]
    train_model(
        args.data_dir,
        args.model_out,
        train_params["n_estimators"],
        train_params["max_depth"],
        train_params["random_state"]
    )