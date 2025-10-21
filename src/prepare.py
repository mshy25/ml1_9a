# src/prepare.py
"""
Stage 1: Data Preparation
- Loads raw dataset
- Cleans missing values
- Removes duplicates
- Saves cleaned data
"""
import pandas as pd
import yaml
import argparse
import os

def prepare_data(out_dir, random_seed):
    """Clean and prepare the dataset"""
    
    # Load raw data
    print("Loading raw dataset...")
    df = pd.read_csv("data/Pakistan_House_Price_Dataset.csv")
    print(f"Original shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Handle missing values (simple approach: drop rows with missing values)
    df = df.dropna()
    print(f"After removing missing values: {df.shape}")
    
    # Save cleaned data
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "cleaned_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()
    
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    prepare_data(args.out_dir, params["prepare"]["random_seed"])