# Requirements:
# pip install tensorflow
# pip install 'tensorflow[and-cuda]'     # For GPU support
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# Check GPU availability:
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import json
import time
from datetime import datetime, timezone

def fetch_dataset(dataset_path, target_column, feature_categories):
    """
    Loads dataset from CSV, extracts the target column, and optionally one-hot encodes categorical features.
    """
    df = pd.read_csv(dataset_path)
    target = df[target_column].values.reshape(-1, 1)
    features = df.drop(target_column, axis=1)

    if feature_categories:
        features = pd.get_dummies(features, columns=feature_categories)

    return features, target
 
def normalize_data(features):
    """
    Normalizes features using StandardScaler.
    """
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features, scaler

def evaluate_model(model, features, target, loss_history, training_time, dataset):
    """
    Evaluates the model and returns performance metrics.
    """
    start_time = time.time() 
    predictions = model.predict(features)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # in milliseconds

    mse = mean_squared_error(target, predictions)
    r2 = r2_score(target, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    return {
        "features": features.tolist(),
        "target": target.tolist(),
        "predictions": predictions.tolist(),
        "loss_history": loss_history,
        "training_time_ms": training_time,
        "inference_time_ms": inference_time,
        "mse": mse,
        "r2": r2
    }

def train_model(features, target):
    """
    Trains a simple linear regression model using TensorFlow.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

    start_time = time.time()
    history = model.fit(features, target, epochs=200, batch_size=4096, verbose=0)
    end_time = time.time()

    training_time = (end_time - start_time) * 1000  # in milliseconds
    loss_history = history.history['loss']
    return model, training_time, loss_history

def run(dataset_path, target_column, feature_categories, feature_index_to_train_on, dataset):
    """
    Executes the linear regression training and evaluation pipeline.
    """
    features, target = fetch_dataset(dataset_path, target_column, feature_categories)

    # Use only one feature column based on index
    single_feature = features.iloc[:, feature_index_to_train_on].values.reshape(-1, 1)
 
    normalized_features, _ = normalize_data(single_feature)
  
    model, training_time, loss_history = train_model(normalized_features, target)

    results = evaluate_model(model, normalized_features, target, loss_history, training_time, dataset)
    del model
    return results

def process(dataset, executionTries, sample, result_item_id):
    """
    Orchestrates the full experiment pipeline:
    - Loads the appropriate dataset
    - Runs training and evaluation
    - Returns metadata and results
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_mapping = {
        1: os.path.join(base_dir, "../../datasets/house_price/sample_10%.csv"),
        2: os.path.join(base_dir, "../../datasets/house_price/sample_50%.csv"),
        3: os.path.join(base_dir, "../../datasets/house_price/sample_100%.csv"),
    }
    dataset_name = {
        1: "sample_10%",
        2: "sample_50%",
        3: "sample_100%",
    }
    dataset_path = dataset_mapping[dataset]

    print(f"Using dataset: {dataset_path}")

    target_column = "price"
    feature_categories = []  # Add categorical columns here if needed
    feature_index_to_train_on = 0  # Index of the feature to train on

    start_time = time.time() 
    results = run(dataset_path, target_column, feature_categories, feature_index_to_train_on, dataset_name[dataset])
    end_time = time.time()
    
    experiments_path = "linear_regression/training_result/" + str(result_item_id)
    sdt = datetime.fromtimestamp(start_time, tz=timezone.utc)
    edt = datetime.fromtimestamp(end_time, tz=timezone.utc)

    return {
        'experiment': {
            'try': int(executionTries),
            'type': "Linear Regression Python gpu",
            'sample': sample,
            'title': "Linear Regression Python gpu " + sample,
            'start': sdt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            'end': edt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            'platform': "python_gpu",
            'result_item_id': result_item_id,
            'location': experiments_path,
            'try_path': experiments_path + "/" + str(executionTries),
            'experiment_path': experiments_path + "/" + str(executionTries) + "/python_gpu",
            'result_path':  experiments_path + "/" + str(executionTries) + "/python_gpu/" +  "python_gpu_" + dataset_name[dataset] + ".json"
        },
        'results': results
    }
