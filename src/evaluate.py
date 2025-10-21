# src/evaluate.py
"""
Stage 4: Model Evaluation
- Tests model on test set
- Calculates metrics (RMSE, R², MAE)
- Saves metrics to JSON
"""
import numpy as np
import json
import argparse
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(data_dir, model_path, out_path):
    """Evaluate model performance"""
    
    print("Loading test data and model...")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    model = joblib.load(model_path)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    metrics = {
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mae": float(mae)
    }
    
    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:,.2f}")
    
    # Save metrics
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--out", default="metrics/eval.json")
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model, args.out)