# pip install tensorflow
# pip install 'tensorflow[and-cuda]'
# verify using GPU: python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# pip install pandas
# pip install matplotlib
# pip install scikit-learn

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

def fetch_dataset(dataset_path, target_column, feature_categories):
    """Fetches and preprocesses the dataset."""
    df = pd.read_csv(dataset_path)

    target = df[target_column].values.reshape(-1, 1)
    features = df.drop(target_column, axis=1)

    # One-hot encode categorical features (if any)
    if feature_categories:
        features = pd.get_dummies(features, columns=feature_categories)

    return features, target

def save_objects_json(features, target, predictions, loss_history, training_time, inferenceTime, mse, r2, filename="regression_data.json"):
    #save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    save_path = os.path.join("../../training_result/python", filename)
    with open(save_path, "w") as file:
        json.dump({
            "features": features.tolist(),
            "target": target.tolist(),
            "predictions": predictions.tolist(),
            "loss_history": loss_history,
            "training_time_ms": training_time,
            "inference_time_ms": inferenceTime,
            "mse": mse,
            "r2": r2
        }, file)
    print(f"Data saved to {save_path}")


def normalize_data(features):
    """Normalizes the features."""
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features, scaler  # Return scaler for prediction



def evaluate_model(model, features, target, loss_history, training_time, dataset):
    """Evaluates the model and get predictions."""
    start_time = time.time() 
    predictions = model.predict(features)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000

    mse = mean_squared_error(target, predictions)
    r2 = r2_score(target, predictions)

    print(f"predictions: {predictions}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    save_objects_json(features, target, predictions, loss_history, training_time, inference_time, mse, r2, "python_gpu_" + dataset + ".json")

    return predictions

def train_model(features, target):
    """Trains a linear regression model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(features.shape[1],))
    ])

    #model.compile(optimizer='sgd', loss='mean_squared_error')
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')
    start_time = time.time()
    history = model.fit(features, target, epochs=200, batch_size=4096, verbose=0)  # No verbose output for cleaner run
    #history = model.fit(features, target, epochs=200, verbose=0)  # No verbose output for cleaner run
    end_time = time.time()
    training_time = (end_time - start_time) * 1000
    loss_history = history.history['loss']
    return model, training_time, loss_history

def run(dataset_path, target_column, feature_categories, feature_index_to_train_on, dataset):
    """Runs the linear regression pipeline."""
    features, target = fetch_dataset(dataset_path, target_column, feature_categories)
 
    # Extract the feature
    single_feature = features.iloc[:, feature_index_to_train_on].values.reshape(-1, 1)

    # test(single_feature, target)

    print(f"features {single_feature}")
    print(f"target {target}")
    
    normalized_features, feature_scaler = normalize_data(single_feature)

    print(f"normalized_features {normalized_features}")

    # Train the model
    model, training_time, loss_history = train_model(normalized_features, target)

    # Evaluate the model and get predictions
    predictions = evaluate_model(model, normalized_features, target, loss_history, training_time, dataset)

    # Plot the regression line and save it
    # plot_regression_line(normalized_features, target, predictions)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a dataset size.")
    parser.add_argument(
        "option",
        type=int,
        choices=[1, 2, 3],
        help="Choose dataset size: 1 for 10%, 2 for 50%, 3 for 100%",
    )
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = ""
    dataset_mapping = {
        1: os.path.join(base_dir, "../../datasets/house_price/sample_10%.csv"),  # Go up two levels to reach the datasets folder
        2: os.path.join(base_dir, "../../datasets/house_price/sample_50%.csv"),
        3: os.path.join(base_dir, "../../datasets/house_price/sample_100%.csv"),
    }
    dataset_name = {
        1: "sample_10%",
        2: "sample_50%",
        3: "sample_100%",
    }
    dataset_path = dataset_mapping[args.option]

    print(f"Using dataset: {dataset_path}")
    #target_column = "fare_amount"
    target_column = "price"
    feature_categories = []  # Replace with your categorical feature columns
    feature_index_to_train_on = 0 

    run(dataset_path, target_column, feature_categories, feature_index_to_train_on, dataset_name[args.option])