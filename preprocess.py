import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

def main(input_dir, output_dir):
    # Find the first CSV file in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not input_files:
        raise FileNotFoundError("No CSV file found in input directory.")
    input_path = os.path.join(input_dir, input_files[0])

    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original fraud values: {df['is_fraud'].unique()}")
    
    # Convert 'is_fraud' to 0/1
    df['is_fraud'] = df['is_fraud'].map({'no': 0, 'yes': 1})
    print(f"Converted fraud values: {df['is_fraud'].unique()}")
    
    # For XGBoost, we need to prepare data in the right format
    # Target column should be first, then features
    # Remove non-numeric columns and rearrange
    feature_columns = ['amount', 'customer_age']  # Keep only numeric features
    target_column = 'is_fraud'
    
    # Create clean dataset with target first, then features
    processed_df = df[[target_column] + feature_columns].copy()
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Processed columns: {processed_df.columns.tolist()}")
    print(f"Sample of processed data:\n{processed_df.head()}")
    
    # Split data 80/20 (train/validation)
    train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42, stratify=processed_df[target_column])
    
    # Save train and validation sets (no header for XGBoost)
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    
    # XGBoost expects CSV without header, target in first column
    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    
    print(f"Training data saved to {train_path} ({len(train_df)} samples)")
    print(f"Validation data saved to {val_path} ({len(val_df)} samples)")
    print(f"Output directory contents: {os.listdir(output_dir)}")
    print(f"Training data sample (first 5 rows):")
    print(train_df.head().to_string(index=False))
    print(f"Validation data sample (first 5 rows):")
    print(val_df.head().to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=os.environ.get('SM_INPUT_DATA_INPUT', '/opt/ml/processing/input'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_OUTPUT', '/opt/ml/processing/output'))
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
