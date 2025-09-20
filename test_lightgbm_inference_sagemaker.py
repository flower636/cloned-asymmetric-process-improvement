# test.py 

import pandas as pd
import joblib
import os
import boto3

# Set up S3 client using boto3
s3_client = boto3.client('s3')

# Get S3 paths from environment variables
test_path = os.getenv('TEST_DATA_PATH', '')
if test_path == '':
    print("No test data path provided - Terminating")
    assert False

model_path = os.getenv('MODEL_PATH', '')
if model_path == '':
    print("No model path provided - Terminating")
    assert False

# Parse the S3 path to get the bucket and key for the test data
test_bucket_name = test_path.split('/')[2]
test_key = '/'.join(test_path.split('/')[3:])

# Download the test data from S3 to a local file
download_test_path = '/tmp/test.csv'
s3_client.download_file(test_bucket_name, test_key, download_test_path)

# Load the test data into pandas
df_test = pd.read_csv(download_test_path)

# Separate features (assuming the target column is last)
X_test = df_test#.iloc[:, :-1].values

# Parse the S3 path to get the bucket and key for the model
model_bucket_name = model_path.split('/')[2]
model_key = '/'.join(model_path.split('/')[3:])

# Download the model from S3 to a local file
local_model_path = '/tmp/lightgbm_model.pkl'
s3_client.download_file(model_bucket_name, model_key, local_model_path)

# Load the model
model = joblib.load(filename=local_model_path)

# Make predictions
y_pred = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Print the predictions
print(f"Predictions: {y_pred}")
