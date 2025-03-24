import tensorflow as tf
import numpy as np
import time
import json
import argparse
import os

def load_mnist(train_percentage=1.0):
    path = '../../datasets/'
    with open(path + 'mnist_train_images.json', 'r') as f:
        train_images_data = json.load(f)
    with open(path + 'mnist_train_labels.json', 'r') as f:
        train_labels_data = json.load(f)
    with open(path + 'mnist_test_images.json', 'r') as f:
        test_images_data = json.load(f)
    with open(path + 'mnist_test_labels.json', 'r') as f:
        test_labels_data = json.load(f)

    train_images = np.array(train_images_data)
    train_labels = np.array(train_labels_data)
    test_images = np.array(test_images_data)
    test_labels = np.array(test_labels_data)

    num_train_samples = int(train_images.shape[0] * train_percentage)
    num_test_samples = int(test_images.shape[0] * train_percentage)

    train_images = train_images[:num_train_samples, :]
    train_labels = train_labels[:num_train_samples, :]
    test_images = test_images[:num_test_samples, :]
    test_labels = test_labels[:num_test_samples, :]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

def predict_and_measure(model, input_tensor):
    start_time = time.perf_counter()
    predictions = model.predict(input_tensor)
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    predicted_class = np.argmax(predictions, axis=1)[0]

    return {'predicted_class': predicted_class, 'inference_time': inference_time}

def train_model(train_percentage):
    data = load_mnist(train_percentage)
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    loss_values = []
    accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    start_time = time.perf_counter()
    history = model.fit(train_images, train_labels, epochs=10,
                      validation_data=(test_images, test_labels),
                      callbacks=[tf.keras.callbacks.LambdaCallback(
                          on_epoch_end=lambda epoch, logs: (
                              loss_values.append(logs['loss']),
                              accuracy_values.append(logs['accuracy']),
                              val_loss_values.append(logs['val_loss']),
                              val_accuracy_values.append(logs['val_accuracy'])
                          )
                      )])
    end_time = time.perf_counter()
    training_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print('Training time:', training_time, 'milliseconds')

    eval_result = model.evaluate(test_images, test_labels)
    loss = eval_result[0]
    accuracy = eval_result[1]

    print('Loss:', loss)
    print('Accuracy:', accuracy)

    sample_image = test_images[0:1, :]
    prediction_result = predict_and_measure(model, sample_image)
    predicted_class = prediction_result['predicted_class']
    inference_time = prediction_result['inference_time']

    print('Predicted class:', predicted_class)
    print('Inference time:', inference_time, 'milliseconds')

    result = {
        'loss_values': loss_values,
        'accuracy_values': accuracy_values,
        'val_loss_values': val_loss_values,
        'val_accuracy_values': val_accuracy_values,
        'training_time_ms': training_time,
        'inference_time_ms': inference_time,
        'loss': loss,
        'accuracy': accuracy
    }
    print(result)
    save_objects_json(result, "python_gpu_sample_" + str(int(train_percentage * 100)) + "%.json")
    return result


def save_objects_json(result, filename="neural_network.json"):
    save_path = os.path.join("../../training_result/python", filename)
    with open(save_path, "w") as file:
        json.dump(result, file)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose a dataset size.")
    parser.add_argument(
        "option",
        type=int,
        choices=[1, 2, 3],
        help="Choose dataset size: 1 for 10%, 2 for 50%, 3 for 100%",
    )
    args = parser.parse_args() 
    
    dataset_perc = {
        1: 0.1,  # Go up two levels to reach the datasets folder
        2: 0.5,
        3: 1.0,
    }
    dataset_name = {
        1: "sample_10%",
        2: "sample_50%",
        3: "sample_100%",
    } 

    train_model(dataset_perc[args.option])

