use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::from_value;
use ndarray::{Array2, Array1, Axis};
use linfa::prelude::*;
use linfa_linear::{LinearRegression, FittedLinearRegression};
use web_sys::{console, window};
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
struct TrainingResult {
    features: Vec<Vec<f64>>,
    target: Vec<Vec<f64>>,
    predictions: Vec<Vec<f64>>, 
    loss_history: Vec<f64>,
    training_time_ms: f64,
    inference_time_ms: f64,
    mse: f64,
    r2: f64,
}

fn normalize_data(features: &Array2<f64>) -> Array2<f64> {
    let mean = features.mean_axis(Axis(0)).unwrap();
    let std = features.std_axis(Axis(0), 0.0);
    (features - &mean) / (&std + 1e-7)
}

#[wasm_bindgen]
pub async fn fetch_and_train(
    features: JsValue,
    target: JsValue,
) -> Result<JsValue, JsValue> {
 
    let features_vec: Vec<f64> = from_value(features)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize features: {}", e)))?;
    let target_vec: Vec<f64> = from_value(target)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize target: {}", e)))?;

    
    let num_samples = target_vec.len();
    let num_features = features_vec.len() / num_samples;
    let features_array = Array2::from_shape_vec((num_samples, num_features), features_vec)
        .map_err(|e| JsValue::from_str(&format!("Failed to create features array: {}", e)))?;
    let target_array = Array1::from_vec(target_vec);

    let normalized_features = normalize_data(features_array);
    let dataset = linfa::Dataset::new(normalized_features.clone(), target.clone());
    let model = LinearRegression::new();
    
    let start_time = window().unwrap().performance().unwrap().now(); 
    let trained_model = model
        .fit(&dataset)
        .map_err(|e| JsValue::from_str(&format!("Failed to train model: {}", e)))?;
    let end_time = window().unwrap().performance().unwrap().now(); // Record end time
    let training_time = end_time - start_time; // Calculate training time in milliseconds
    

    let start_time = window().unwrap().performance().unwrap().now(); 
    let predictions = model.predict(&normalized_features);
    let end_time = window().unwrap().performance().unwrap().now(); // Record end time
    let inference_time = end_time - start_time; // Calculate training time in milliseconds
    
    let mse = predictions.mean_squared_error(target).map_err(|e| JsValue::from_str(&format!("Failed to calculate MSE: {}", e)))?; // Handle Result
    let target_mean = target.mean().unwrap();
    let mut ss_total = 0.0;
    let mut ss_res = 0.0;

    for i in 0..target.len() {
        ss_total += (target[i] - target_mean).powi(2);
        ss_res += (predictions[i] - target[i]).powi(2);
    }

    let r_squared = 1.0 - (ss_res / ss_total);

    
    let result = TrainingResult {
        features: normalized_features,
        target: target.into_iter().map(|x| vec![x]).collect(),
        predictions: predictions.clone().into_raw_vec().iter().map(|&x| vec![x]).collect(),
        loss_history: ???,
        training_time_ms: training_time,
        inference_time_ms: inference_time,
        mse: mse,
        r2: r_squared,
    };

    Ok(serde_wasm_bindgen::to_value(&result).unwrap())
}