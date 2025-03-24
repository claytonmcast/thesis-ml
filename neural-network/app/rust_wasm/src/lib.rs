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

#[derive(Serialize, Deserialize)]
struct SerializableNDArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}


pub fn load_nab_from_bytes(bytes: &[u8]) -> io::Result<NDArray> {
    let cursor = Cursor::new(bytes);
    let mut decoder = GzDecoder::new(cursor);
    let mut serialized_data = Vec::new();
    decoder.read_to_end(&mut serialized_data)?;
    let serializable_array: SerializableNDArray = bincode::deserialize(&serialized_data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(NDArray::new(serializable_array.data, serializable_array.shape))
}

fn get_mnist_data(
    train_images_nab: &[u8], // Training images as bytes
    train_labels_nab: &[u8], // Training labels as bytes
    train_percent: f64
) -> Result<((NDArray, NDArray), (NDArray, NDArray)), JsValue> {  
    let images = load_nab_from_bytes(train_images_nab)
        .map_err(|e| JsValue::from_str(&format!("Error loading images: {}", e)))?;
    let labels = load_nab_from_bytes(train_labels_nab)
        .map_err(|e| JsValue::from_str(&format!("Error loading labels: {}", e)))?;
    let num_samples = images.shape()[0];
    let train_size = ((train_percent / 100.0) * num_samples as f64).round() as usize;

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

#[wasm_bindgen]
pub async fn fetch_and_train(
    train_images: &[u8],
    train_labels: &[u8],
    train_percent: f64
) -> Result<JsValue, JsValue> {
    console::log_1(&format!("{:?}", "Entered function").into());
    console::log_1(&format!("Train images first 10 bytes: {:?}", &train_images[0..10]).into());
    console::log_1(&format!("Train labels first 10 bytes: {:?}", &train_labels[0..10]).into());

    // Step 1: Load dataset from bytes
    let ((x_train, y_train), (x_test, y_test)) = get_mnist_data(train_images, train_labels, train_percent)?;
    
    // Step 2: Normalize input data (scale pixels to 0-1)
    println!("Normalizing data...");
    let x_train = x_train.divide_scalar(255.0);
    let x_test = x_test.divide_scalar(255.0);

    // Step 3: Reshape input data
    let x_train = x_train.reshape(&[x_train.shape()[0], 784])
        .expect("Failed to reshape training data");
    let x_test = x_test.reshape(&[x_test.shape()[0], 784])
        .expect("Failed to reshape test data");

    // Step 3: One-hot encode target data
    println!("One-hot encoding targets...");
    let y_train = NDArray::one_hot_encode(&y_train);
    let y_test = NDArray::one_hot_encode(&y_test);

    println!("Data shapes:");
    println!("x_train: {:?}", x_train.shape());
    println!("y_train: {:?}", y_train.shape());
    println!("x_test: {:?}", x_test.shape());
    println!("y_test: {:?}", y_test.shape());

    // Step 3: Create model architecture
    println!("Creating model...");

    // Reset node ID counter before test
    reset_node_id();
            
            
    let input = NabModel::input(vec![784]);
    let dense1 = NabLayer::dense(784, 32, Some("relu"), Some("dense1"));
    let x = input.apply(dense1);

    let dense2 = NabLayer::dense(32, 32, Some("relu"), Some("dense2"));
    let x = x.apply(dense2);

    let output_layer = NabLayer::dense(32, 10, Some("softmax"), Some("output"));
    let output = x.apply(output_layer); 
    
    let mut model = NabModel::new_functional(vec![input], vec![output]);
    
   
    model.summary();

    model.compile(
        "sgd",                      
        0.1,                        
        "categorical_crossentropy", 
        vec!["accuracy".to_string()]
    );


    // Train model
    let start_time = window().unwrap().performance().unwrap().now();
    let history = model.fit(
        &x_train,
        &y_train,
        32,             // Increase batch size from 32 to 64
        10,             // Increase epochs from 2 to 10
        Some((&x_test, &y_test))
    );
    let end_time = window().unwrap().performance().unwrap().now();
    let training_time = end_time - start_time;

    
    // Step 6: Evaluate final model
    let eval_metrics = model.evaluate(&x_test, &y_test, 32);

    let start_time = window().unwrap().performance().unwrap().now();
    let predictions = model.predict(&x_test);
    let end_time = window().unwrap().performance().unwrap().now();
    let inference_time = end_time - start_time;

    // Print final results
    println!("Final test accuracy: {:.2}%", eval_metrics["accuracy"] * 100.0);
    
    // Create a TrainingResults struct with placeholder values
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