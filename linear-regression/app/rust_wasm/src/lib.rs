use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::from_value;
use web_sys::{console, window}; 
use nabla_ml::nab_array::NDArray;   

 /*    downloadJson({
        features,
        target,
        predictions: predArray,
        trainingTime,
        inferenceTime,
        mse,
        r2
    }, fileName)   */
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
 
fn calculate_mean(features: &[Vec<f64>]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new(); // Handle empty input
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    let mut means = vec![0.0; num_cols];

    for col in 0..num_cols {
        let mut sum = 0.0;
        for row in 0..num_rows {
            sum += features[row][col];
        }
        means[col] = sum / num_rows as f64;
    }

    means
}

fn calculate_std_dev(features: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new(); // Handle empty input
    }

    let num_rows = features.len();
    let num_cols = features[0].len();
    let mut std_devs = vec![0.0; num_cols];

    for col in 0..num_cols {
        let mut sum_squared_diff = 0.0;
        for row in 0..num_rows {
            sum_squared_diff += (features[row][col] - means[col]).powi(2);
        }
        std_devs[col] = (sum_squared_diff / num_rows as f64).sqrt();
    }

    std_devs
}

fn calculate_mse(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.len() != targets.len() {
        panic!("Predictions and targets must have the same length.");
    }

    let n = predictions.len() as f64;
    let sum_squared_errors: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            let error = pred - target;
            error * error
        })
        .sum();

    sum_squared_errors / n
}

fn calculate_r_squared(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.len() != targets.len() {
        panic!("Predictions and targets must have the same length.");
    }

    let n = predictions.len() as f64;

    // Calculate the mean of the targets
    let target_mean: f64 = targets.iter().sum::<f64>() / n;

    // Calculate the total sum of squares (SST)
    let sst: f64 = targets
        .iter()
        .map(|&target| {
            let diff = target - target_mean;
            diff * diff
        })
        .sum();

    // Calculate the sum of squared residuals (SSR)
    let ssr: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            let error = pred - target;
            error * error
        })
        .sum();

    // Calculate R-squared
    if sst == 0.0 {
        // Handle the case where SST is zero (all targets are the same).
        // In this scenario, R-squared is typically undefined or set to 1.0 if SSR is also 0.0.
        if ssr == 0.0{
            return 1.0;
        }else {
            return 0.0;
        }
    }

    1.0 - (ssr / sst)
}

fn normalize_features(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() {
        return Vec::new(); // Handle empty input
    }

    let means = calculate_mean(features);
    let std_devs = calculate_std_dev(features, &means);
    let num_rows = features.len();
    let num_cols = features[0].len();
    let mut normalized_features = vec![vec![0.0; num_cols]; num_rows];

    for row in 0..num_rows {
        for col in 0..num_cols {
            normalized_features[row][col] = (features[row][col] - means[col]) / std_devs[col];
        }
    }

    normalized_features
}
 
use nabla_ml::nabla::Nabla;
 
#[wasm_bindgen]
pub async fn fetch_and_train(
    features_js: JsValue,
    target_js: JsValue
) -> Result<JsValue, JsValue> {
    let features: Vec<Vec<f64>> = from_value::<Vec<f64>>(features_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to deserialize features: {}", e)))?
            .into_iter()
            .map(|x| vec![x])
            .collect();
    let target: Vec<f64> = from_value(target_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize target: {}", e)))?;
 
    
    console::log_1(&format!("features {:?}", features).into());
    let normalized_features_x: Vec<Vec<f64>> = normalize_features(&features); 
    let X =  NDArray::from_matrix(normalized_features_x.clone()); 
    let y = NDArray::from_vec(target.clone());
    console::log_1(&format!("X {:?}", X).into());
    console::log_1(&format!("Y {:?}", y).into()); 

    let start_time = window().unwrap().performance().unwrap().now();
    let (theta, history) = Nabla::linear_regression(&X, &y, 0.1, 200);
    let end_time = window().unwrap().performance().unwrap().now();
    let training_time = end_time - start_time;

    let start_time = window().unwrap().performance().unwrap().now();
    let final_predictions = X.clone() * theta[0] + theta[1];
    let end_time = window().unwrap().performance().unwrap().now();
    let inference_time = end_time - start_time;

    let mse = calculate_mse(&final_predictions.data, &y.data);
    let r2 = calculate_r_squared(&final_predictions.data, &y.data);
    let result = TrainingResult {
        features: normalized_features_x,
        target: target.into_iter().map(|x| vec![x]).collect(),
        predictions: final_predictions.data.into_iter().map(|x| vec![x]).collect(),
        loss_history: history,
        training_time_ms: training_time,
        inference_time_ms: inference_time,
        mse: mse,
        r2: r2,
    };

    Ok(serde_wasm_bindgen::to_value(&result).unwrap())

    // console::log_1(&format!("theta {:?}", theta).into());
    // console::log_1(&format!("history {:?}", history).into());
         
    // let normalized_features_x: Vec<Vec<f64>> = normalize_features(&features);
    // let X: Vec<f64> = normalized_features_x.clone().into_iter().flatten().collect(); 
    // let X = NDArray::from_vec(X);
    // let y = NDArray::from_vec(target.clone());
    // console::log_1(&format!("X {:?}", X).into());
    // console::log_1(&format!("Y {:?}", y).into());
        
    // // Apply linear regression
    // let m = X.size() as f64;
    // let mut theta_0 = 0.0;
    // let mut theta_1 = 0.0;
    // let learning_rate = 0.01;
    // let num_epochs = 200;
    // let mut loss_history = Vec::with_capacity(num_epochs);
 
    // let start_time = window().unwrap().performance().unwrap().now();
    // // Training loop
    // for epoch in 0..num_epochs {
    //     let predictions = X.clone() * theta_1 + theta_0;
    //     let errors = predictions.clone() - y.clone();
        
    //     // Convert errors to Vec<f64> if needed
    //     let errors_vec: Vec<f64> = (0..errors.size()).map(|i| errors.get(i)).collect();

    //     // Compute gradients
    //     let d_theta_0 = (1.0 / m) * errors_vec.iter().sum::<f64>();
    //     let d_theta_1 = (1.0 / m) * X.data().iter().zip(errors_vec.iter()).map(|(&x, &e)| x * e).sum::<f64>();
        
    //     // Update parameters
    //     theta_0 -= learning_rate * d_theta_0;
    //     theta_1 -= learning_rate * d_theta_1;
        
    //     // Compute and store loss
    //     let mse = (1.0 / (2.0 * m)) * errors_vec.iter().map(|e| e * e).sum::<f64>();
    //     loss_history.push(mse);
    // }
    // let end_time = window().unwrap().performance().unwrap().now();
    // let training_time = end_time - start_time;
 
    // let start_time = window().unwrap().performance().unwrap().now();
    // let final_predictions = X.clone() * theta_1 + theta_0;
    // let end_time = window().unwrap().performance().unwrap().now();
    // let inference_time = end_time - start_time;

    // console::log_1(&format!("Final Predictions: {:?}", final_predictions).into());
 
    // // Calculate MSE and R-squared
    // let mse = calculate_mse(&final_predictions.data, &y.data);
    // let r2 = calculate_r_squared(&final_predictions.data, &y.data);

    // let result = TrainingResult {
    //     features: normalized_features_x,
    //     target: target.into_iter().map(|x| vec![x]).collect(),
    //     predictions: final_predictions.data.into_iter().map(|x| vec![x]).collect(),
    //     loss_history,
    //     training_time_ms: training_time,
    //     inference_time_ms: inference_time,
    //     mse: mse,
    //     r2: r2,
    // };

    // Ok(serde_wasm_bindgen::to_value(&result).unwrap())
} 

#[wasm_bindgen]
pub async fn fetch_and_train2(
    features_js: JsValue,
    target_js: JsValue
) -> Result<JsValue, JsValue> {
    let features: Vec<Vec<f64>> = from_value::<Vec<f64>>(features_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to deserialize features: {}", e)))?
            .into_iter()
            .map(|x| vec![x])
            .collect();
    let target: Vec<f64> = from_value(target_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to deserialize target: {}", e)))?;
 
  
    console::log_1(&format!("features {:?}", features).into());
    let normalized_features_x: Vec<Vec<f64>> = normalize_features(&features);
    let X: Vec<f64> = normalized_features_x.clone().into_iter().flatten().collect(); 
    let X = NDArray::from_vec(X);
    let y = NDArray::from_vec(target.clone());
    console::log_1(&format!("X {:?}", X).into());
    console::log_1(&format!("Y {:?}", y).into());
        
    // Apply linear regression
    let m = X.size() as f64;
    let mut theta_0 = 0.0;
    let mut theta_1 = 0.0;
    let learning_rate = 0.01;
    let num_epochs = 200;
    let mut loss_history = Vec::with_capacity(num_epochs);
 
    let start_time = window().unwrap().performance().unwrap().now();
    // Training loop
    for epoch in 0..num_epochs {
        let predictions = X.clone() * theta_1 + theta_0;
        let errors = predictions.clone() - y.clone();
        
        // Convert errors to Vec<f64> if needed
        let errors_vec: Vec<f64> = (0..errors.size()).map(|i| errors.get(i)).collect();

        // Compute gradients
        let d_theta_0 = (1.0 / m) * errors_vec.iter().sum::<f64>();
        let d_theta_1 = (1.0 / m) * X.data().iter().zip(errors_vec.iter()).map(|(&x, &e)| x * e).sum::<f64>();
        
        // Update parameters
        theta_0 -= learning_rate * d_theta_0;
        theta_1 -= learning_rate * d_theta_1;
        
        // Compute and store loss
        let mse = (1.0 / (2.0 * m)) * errors_vec.iter().map(|e| e * e).sum::<f64>();
        loss_history.push(mse);
    }
    let end_time = window().unwrap().performance().unwrap().now();
    let training_time = end_time - start_time;
 
    let start_time = window().unwrap().performance().unwrap().now();
    let final_predictions = X.clone() * theta_1 + theta_0;
    let end_time = window().unwrap().performance().unwrap().now();
    let inference_time = end_time - start_time;

    console::log_1(&format!("Final Predictions: {:?}", final_predictions).into());
 
    // Calculate MSE and R-squared
    let mse = calculate_mse(&final_predictions.data, &y.data);
    let r2 = calculate_r_squared(&final_predictions.data, &y.data);

    let result = TrainingResult {
        features: normalized_features_x,
        target: target.into_iter().map(|x| vec![x]).collect(),
        predictions: final_predictions.data.into_iter().map(|x| vec![x]).collect(),
        loss_history,
        training_time_ms: training_time,
        inference_time_ms: inference_time,
        mse: mse,
        r2: r2,
    };

    Ok(serde_wasm_bindgen::to_value(&result).unwrap())
} 