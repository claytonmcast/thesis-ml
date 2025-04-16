use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::from_value;
use web_sys::{console, window};

use std::io::{self, Read, Cursor};
use flate2::read::GzDecoder;
use bincode;

use nabla_ml::nab_array::NDArray;
use nabla_ml::nab_model::NabModel;
use nabla_ml::nab_layers::NabLayer; 
use nabla_ml::nab_model::reset_node_id;

// Define structure to hold the training results
#[derive(Serialize, Deserialize)]
struct TrainingResults {
    loss_values: Vec<f64>,
    accuracy_values: Vec<f64>,
    val_loss_values: Vec<f64>,
    val_accuracy_values: Vec<f64>,
    training_time_ms: f64,
    inference_time_ms: f64,
    loss: f64,
    accuracy: f64
}

// Structure for serializing NDArray
#[derive(Serialize, Deserialize)]
struct SerializableNDArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}

// Function to load NAB data from bytes (gzip compressed)
pub fn load_nab_from_bytes(bytes: &[u8]) -> io::Result<NDArray> {
    let cursor = Cursor::new(bytes);
    let mut decoder = GzDecoder::new(cursor);  // Decompress the Gz file
    let mut serialized_data = Vec::new();
    decoder.read_to_end(&mut serialized_data)?;  // Read the decompressed data into a vector

    // Deserialize the data into a SerializableNDArray and then into a NDArray
    let serializable_array: SerializableNDArray = bincode::deserialize(&serialized_data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(NDArray::new(serializable_array.data, serializable_array.shape))
}

// Function to load MNIST data from byte arrays and split into training and testing sets
fn get_mnist_data(
    train_images_nab: &[u8],  // Byte data for training images
    train_labels_nab: &[u8],  // Byte data for training labels
    train_percent: f64        // Percentage of the data to use for training
) -> Result<((NDArray, NDArray), (NDArray, NDArray)), JsValue> {  
    // Load the image and label data
    let images = load_nab_from_bytes(train_images_nab)
        .map_err(|e| JsValue::from_str(&format!("Error loading images: {}", e)))?;
    let labels = load_nab_from_bytes(train_labels_nab)
        .map_err(|e| JsValue::from_str(&format!("Error loading labels: {}", e)))?;

    // Determine the split between training and testing sets
    let num_samples = images.shape()[0];
    let train_size = ((train_percent / 100.0) * num_samples as f64).round() as usize;

    // Split the data into training and testing sets
    let train_images = NDArray::new(
        images.data()[..train_size * images.shape()[1] * images.shape()[2]].to_vec(),
        vec![train_size, images.shape()[1], images.shape()[2]],
    );

    let test_images = NDArray::new(
        images.data()[train_size * images.shape()[1] * images.shape()[2]..].to_vec(),
        vec![num_samples - train_size, images.shape()[1], images.shape()[2]],
    );

    let train_labels = NDArray::new(
        labels.data()[..train_size].to_vec(),
        vec![train_size],
    );

    let test_labels = NDArray::new(
        labels.data()[train_size..].to_vec(),
        vec![num_samples - train_size],
    );

    Ok(((train_images, train_labels), (test_images, test_labels)))
}

// Entry point for fetching and training the model
#[wasm_bindgen]
pub async fn fetch_and_train(
    train_images: &[u8],  // Byte data for training images
    train_labels: &[u8],  // Byte data for training labels
    train_percent: f64    // Percentage of the data to use for training
) -> Result<JsValue, JsValue> {
    console::log_1(&format!("{:?}", "Entered function").into());
    //console::log_1(&format!("Train images first 10 bytes: {:?}", &train_images[0..10]).into());
    //console::log_1(&format!("Train labels first 10 bytes: {:?}", &train_labels[0..10]).into());

    // Step 1: Load dataset from bytes
    let ((x_train, y_train), (x_test, y_test)) = get_mnist_data(train_images, train_labels, train_percent)?;
    
    // Step 2: Normalize input data (scale pixels to 0-1)
    console::log_1(&"Normalizing data...".into());
    let x_train = x_train.divide_scalar(255.0);
    let x_test = x_test.divide_scalar(255.0);

    // Step 3: Reshape input data to 2D (flatten images to 784-dimensional vectors)
    let x_train = x_train.reshape(&[x_train.shape()[0], 784])
        .expect("Failed to reshape training data");
    let x_test = x_test.reshape(&[x_test.shape()[0], 784])
        .expect("Failed to reshape test data");

    // Step 4: One-hot encode target data (labels)
    console::log_1(&"One-hot encoding targets...".into());
    let y_train = NDArray::one_hot_encode(&y_train);
    let y_test = NDArray::one_hot_encode(&y_test);

    // Print the shapes of the data
    //console::log_1(&format!("x_train shape: {:?}", x_train.shape()).into());
    //console::log_1(&format!("y_train shape: {:?}", y_train.shape()).into());
    //console::log_1(&format!("x_test shape: {:?}", x_test.shape()).into());
    //console::log_1(&format!("y_test shape: {:?}", y_test.shape()).into());

    // Step 5: Create model architecture
    console::log_1(&"Creating model...".into());

    // Reset the node ID counter before defining the model
    reset_node_id();

    // Define the model architecture
    let input = NabModel::input(vec![784]);
    let dense1 = NabLayer::dense(784, 32, Some("relu"), Some("dense1"));
    let x = input.apply(dense1);

    let dense2 = NabLayer::dense(32, 32, Some("relu"), Some("dense2"));
    let x = x.apply(dense2);

    let output_layer = NabLayer::dense(32, 10, Some("softmax"), Some("output"));
    let output = x.apply(output_layer); 
    
    let mut model = NabModel::new_functional(vec![input], vec![output]);
    
    // Print the model summary
    model.summary();

    // Compile the model with the specified optimizer, learning rate, loss function, and metrics
    model.compile(
        "sgd",                       // Optimizer: SGD
        0.1,                         // Learning rate
        "categorical_crossentropy",  // Loss function
        vec!["accuracy".to_string()] // Metrics
    );

    // Step 6: Train the model
    let start_time = window().unwrap().performance().unwrap().now();
    let history = model.fit(
        &x_train,
        &y_train,
        32,             // Batch size
        10,             // Number of epochs
        Some((&x_test, &y_test)) // Validation data
    );
    let end_time = window().unwrap().performance().unwrap().now();
    let training_time = end_time - start_time;

    // Step 7: Evaluate the final model
    let eval_metrics = model.evaluate(&x_test, &y_test, 32);

    // Measure inference time
    let start_time = window().unwrap().performance().unwrap().now();
    let _predictions = model.predict(&x_test);
    let end_time = window().unwrap().performance().unwrap().now();
    let inference_time = end_time - start_time;

    // Print final results
    //console::log_1(&format!("Final test accuracy: {:.2}%", eval_metrics["accuracy"] * 100.0).into());
    
    // Create a TrainingResults struct with results
    let results_with_time = TrainingResults {
        loss_values: history["loss"].clone(),
        accuracy_values: history["accuracy"].clone(),
        val_loss_values: history["val_loss"].clone(),
        val_accuracy_values: history["val_accuracy"].clone(),
        training_time_ms: training_time, 
        inference_time_ms: inference_time,
        loss: eval_metrics["loss"].clone(), 
        accuracy: eval_metrics["accuracy"].clone()
    };

    // Serialize and return results
    Ok(serde_wasm_bindgen::to_value(&results_with_time).unwrap())
}
