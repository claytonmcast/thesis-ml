use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::from_value;
use web_sys::window;
use nabla_ml::nab_array::NDArray;
use nabla_ml::nabla::Nabla;

/// Struct to serialize the training result to JS
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

/// Calculate mean of each column (feature) in the dataset
fn calculate_mean(features: &[Vec<f64>]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    let mut means = vec![0.0; num_cols];

    for col in 0..num_cols {
        for row in 0..num_rows {
            means[col] += features[row][col];
        }
        means[col] /= num_rows as f64;
    }

    means
}

/// Calculate standard deviation of each feature column
fn calculate_std_dev(features: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    let mut std_devs = vec![0.0; num_cols];

    for col in 0..num_cols {
        for row in 0..num_rows {
            let diff = features[row][col] - means[col];
            std_devs[col] += diff * diff;
        }
        std_devs[col] = (std_devs[col] / num_rows as f64).sqrt();
    }

    std_devs
}

/// Normalize each feature column using Z-score normalization
fn normalize_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new();
    }

    let means = calculate_mean(features);
    let std_devs = calculate_std_dev(features, &means);
    let num_rows = features.len();
    let num_cols = features[0].len();

    let mut normalized = vec![vec![0.0; num_cols]; num_rows];
    for row in 0..num_rows {
        for col in 0..num_cols {
            normalized[row][col] = (features[row][col] - means[col]) / std_devs[col];
        }
    }

    normalized
}

/// Mean Squared Error between predictions and targets
fn calculate_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "MSE input size mismatch");

    let sum_squared_error: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();

    sum_squared_error / predictions.len() as f64
}

/// R² score (coefficient of determination)
fn calculate_r_squared(predictions: &[f64], targets: &[f64]) -> f64 {
    assert_eq!(predictions.len(), targets.len(), "R² input size mismatch");

    let n = predictions.len() as f64;
    let target_mean = targets.iter().sum::<f64>() / n;

    let sst: f64 = targets.iter().map(|&t| (t - target_mean).powi(2)).sum();
    let ssr: f64 = predictions.iter().zip(targets).map(|(&p, &t)| (p - t).powi(2)).sum();

    if sst == 0.0 {
        return if ssr == 0.0 { 1.0 } else { 0.0 };
    }

    1.0 - (ssr / sst)
}

/// WebAssembly entry point: Fetches JS data, trains a linear model, and returns performance metrics
#[wasm_bindgen]
pub async fn fetch_and_train(features_js: JsValue, target_js: JsValue) -> Result<JsValue, JsValue> {
    // Deserialize JS input to Rust Vec<Vec<f64>>
    let features: Vec<Vec<f64>> = from_value::<Vec<f64>>(features_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize features: {}", e)))?
        .into_iter()
        .map(|x| vec![x]) // Convert to 2D
        .collect();

    let target: Vec<f64> = from_value(target_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize target: {}", e)))?; 

    // Normalize features for stable training
    let normalized_features_x = normalize_features(&features);

    // Convert data to NDArray
    let X = NDArray::from_matrix(normalized_features_x.clone());
    let y = NDArray::from_vec(target.clone());

    // Measure training time
    let start_time = window().unwrap().performance().unwrap().now();
    let (theta, history) = Nabla::linear_regression(&X, &y, 0.1, 200);
    let end_time = window().unwrap().performance().unwrap().now();
    let training_time = end_time - start_time;

    // Measure inference time
    let start_time = window().unwrap().performance().unwrap().now();
    let final_predictions = X.clone() * theta[0] + theta[1];
    let end_time = window().unwrap().performance().unwrap().now();
    let inference_time = end_time - start_time;

    // Calculate evaluation metrics
    let mse = calculate_mse(&final_predictions.data, &y.data);
    let r2 = calculate_r_squared(&final_predictions.data, &y.data);

    // Prepare result struct
    let result = TrainingResult {
        features: normalized_features_x,
        target: target.into_iter().map(|x| vec![x]).collect(),
        predictions: final_predictions
            .data
            .into_iter()
            .map(|x| vec![x])
            .collect(),
        loss_history: history,
        training_time_ms: training_time,
        inference_time_ms: inference_time,
        mse,
        r2,
    };

    // Serialize to JS value and return
    Ok(serde_wasm_bindgen::to_value(&result).unwrap())
}
