import pandas as pd
import xgboost as xgb
import os
import joblib
import argparse

def main(input_dir, model_dir):
	# Find the processed CSV file
	input_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
	if not input_files:
		raise FileNotFoundError("No CSV file found in input directory.")
	input_path = os.path.join(input_dir, input_files[0])

	df = pd.read_csv(input_path)
	X = df.drop(['is_fraud', 'transaction_id'], axis=1)
	y = df['is_fraud']

	model = xgb.XGBClassifier()
	model.fit(X, y)

	os.makedirs(model_dir, exist_ok=True)
	model_path = os.path.join(model_dir, 'model.joblib')
	joblib.dump(model, model_path)
	print(f"Model saved to {model_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/processing/output'))
	parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
	args = parser.parse_args()
	main(args.input_dir, args.model_dir)
