import tensorflow as tf
import numpy as np
import time
import json
import argparse
import os
from datetime import datetime, timezone

# Load MNIST dataset from JSON files, allowing partial loading via train_percentage
def load_mnist(train_percentage=1.0):
    path = 'neural_network/datasets/'

    # Load JSON-encoded image and label arrays
    with open(path + 'mnist_train_images.json', 'r') as f:
        train_images_data = json.load(f)
    with open(path + 'mnist_train_labels.json', 'r') as f:
        train_labels_data = json.load(f)
    with open(path + 'mnist_test_images.json', 'r') as f:
        test_images_data = json.load(f)
    with open(path + 'mnist_test_labels.json', 'r') as f:
        test_labels_data = json.load(f)

    # Convert to numpy arrays
    train_images = np.array(train_images_data)
    train_labels = np.array(train_labels_data)
    test_images = np.array(test_images_data)
    test_labels = np.array(test_labels_data)

    # Reduce data size if train_percentage < 1.0
    num_train_samples = int(train_images.shape[0] * train_percentage)
    num_test_samples = int(test_images.shape[0] * train_percentage)

    train_images = train_images[:num_train_samples]
    train_labels = train_labels[:num_train_samples]
    test_images = test_images[:num_test_samples]
    test_labels = test_labels[:num_test_samples]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

# Predict a single input and measure inference time
def predict_and_measure(model, input_tensor):
    start_time = time.time()
    predictions = model.predict(input_tensor)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # ms

    predicted_class = np.argmax(predictions, axis=1)[0]
    return {'predicted_class': predicted_class, 'inference_time': inference_time}

# Train a simple neural network on MNIST data and collect performance metrics
def train_model(train_percentage):
    data = load_mnist(train_percentage)
    train_images, train_labels = data['train_images'], data['train_labels']
    test_images, test_labels = data['test_images'], data['test_labels']

    # Define a simple feedforward neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Lists to collect metrics during training
    loss_values, accuracy_values = [], []
    val_loss_values, val_accuracy_values = [], []

    # Train the model and track metrics at each epoch
    start_time = time.time()
    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[tf.keras.callbacks.LambdaCallback(
                  on_epoch_end=lambda epoch, logs: (
                      loss_values.append(logs['loss']),
                      accuracy_values.append(logs['accuracy']),
                      val_loss_values.append(logs['val_loss']),
                      val_accuracy_values.append(logs['val_accuracy'])
                  )
              )])
    end_time = time.time()
    training_time = (end_time - start_time) * 1000  # ms
    print('Training time:', training_time, 'milliseconds')

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_images, test_labels)
    print('Loss:', loss)
    print('Accuracy:', accuracy)

    # Test prediction on a single image
    sample_image = test_images[0:1, :]
    prediction_result = predict_and_measure(model, sample_image)
    print('Predicted class:', prediction_result['predicted_class'])
    print('Inference time:', prediction_result['inference_time'], 'milliseconds')

    # Return all results and metrics
    return {
        'loss_values': loss_values,
        'accuracy_values': accuracy_values,
        'val_loss_values': val_loss_values,
        'val_accuracy_values': val_accuracy_values,
        'training_time_ms': training_time,
        'inference_time_ms': prediction_result['inference_time'],
        'loss': loss,
        'accuracy': accuracy
    } 

# Perform a training run and format results for saving and reporting
def process(dataset, executionTries, sample, result_item_id):
    dataset_perc = {
        1: 0.1,
        2: 0.5,
        3: 1.0,
    }
    dataset_name = {
        1: "sample_10%",
        2: "sample_50%",
        3: "sample_100%",
    }

    start_time = time.time()
    results = train_model(dataset_perc[dataset])
    end_time = time.time()

    sdt = datetime.fromtimestamp(start_time, tz=timezone.utc)
    edt = datetime.fromtimestamp(end_time, tz=timezone.utc)

    experiments_path = f"neural_network/training_result/{result_item_id}"

    return {
        'experiment': {
            'try': int(executionTries),
            'type': "Neural Network Python gpu",
            'sample': sample,
            'title': f"Neural Network Python gpu {sample}",
            'start': sdt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            'end': edt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            'platform': "python_gpu",
            'result_item_id': result_item_id,
            'location': experiments_path,
            'try_path': f"{experiments_path}/{executionTries}",
            'experiment_path': f"{experiments_path}/{executionTries}/python_gpu",
            'result_path': f"{experiments_path}/{executionTries}/python_gpu/python_gpu_sample_{int(dataset_perc[dataset] * 100)}%.json"
        },
        'results': results
    }