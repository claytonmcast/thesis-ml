# Thesis ML Web App

A browser-based machine learning demo built for my Master's thesis:  
**"Exploring the Feasibility of Browser-Based Machine Learning: Challenges and Dataset Limitations"**.

---

## 🚀 Features

- Train machine learning models (e.g., linear regression, NN) directly in the browser using WebAssembly.
- Select different dataset sizes and model complexities.
- Collect performance metrics (training time, inference time, MSE, R², accuracy, loss).
- Plot results


## Run Web App

### Installing Requirements

python -m venv .venv                            # create an environment for python
for windows: .venv\Scripts\activate             # access the environment for windows
for ubuntu:  source .venv\bin\activate          # access the environement for ubuntu
pip install 'tensorflow[and-cuda]'              # For NVIDIA GPU support, this takes long time to download and install everything
pip install pandas
pip install matplotlib
pip install scikit-learn
# Check GPU availability:
# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# navigate to thesis-ml
python server.py                                # run server

## 📁 Project Structure

The following is the breakdown of the project structure and its contents:

- **assets**
  - `alpine_interface.js`: Alpine component to generate the grid based on data
  - `app.js`: Manages API calls to the server and general functions
  - `module_binding.js`: Binds the modules for the runAll function
  - `style.css`: Basic styles for the interface

- **linear_regression**
  - **app/python**
    - `linear_Regression.py`: Python script for training the linear regression model using TensorFlow
  - **rust_wasm**
    - **pkg**
      - `rust_wasm_bg.wasm`: WASM binary for running the model in the browser
      - `rust_wasm.js`: JavaScript code to run the WASM
    - **src**
      - `lib.rs`: Rust code for the linear regression model
      - `rust_wasm_app.js`: Binds buttons from interface to call the WASM
  - **tensorflow_js**
    - `tensorflow_js_app.js`: Binds buttons for training the model with TensorFlow.js
  - **datasets**
    - `house_price/`: Contains datasets for house rent price training
      - `sample_10%.csv`
      - `sample_50%.csv`
      - `sample_100%.csv`
  - **plot**
    - `linear_regression_plot.py`: Plots the graphs and stores metrics of the results
  - **training_result**
    - `[1]/[1]/[platform]`: Directory structure for storing results based on transaction ID, try and platform
    - `confidence_interval_metric.json`: Stores the confidence interval for metrics across all tries

- **neural_network**
  - **app/python**
    - `neural_network.py`: Python script for training the neural network using TensorFlow
  - **rust_wasm**
    - **pkg**
      - `rust_wasm_bg.wasm`: WASM binary for running the neural network in the browser
      - `rust_wasm.js`: JavaScript code to run the WASM
    - **src**
      - `lib.rs`: Rust code for the neural network model
      - `rust_wasm_app.js`: Binds buttons from interface to call the WASM
  - **tensorflow_js**
    - `tensorflow_js_app.js`: Binds buttons for training the model with TensorFlow.js
  - **datasets**
    - `mnist/`: Dataset for the neural network training
      - `mnist_train_images.json`
      - `mnist_train_labels.json`
  - **plot**
    - `neural_network_plot.py`: Plots the graphs and stores metrics of the results
  - **training_result**
    - `[1]/[1]/[platform]`: Directory structure for storing results based on transaction ID, try, and platform
    - `confidence_interval_metric.json`: Stores the confidence interval for metrics across all tries

- `blank.html`: Used for certain scenarios where there is an exception and redirects to clean resources
- `index.html`: The main interface to train the datasets
- `README.md`: Explanation of the application
- `result_list.json`: Database to store training status
- `server.py`: Runs the website and API for executing features


## Debug WASM

To build WASM use the following command: wasm-pack build --target web # Note that you might need to install extra libs in your system
