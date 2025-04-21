# 🧠 Thesis ML Web App

A browser-based machine learning demo built for my Master's thesis:  
**"Exploring the Feasibility of Browser-Based Machine Learning: Challenges and Dataset Limitations"**

This project demonstrates training ML models like linear regression and neural networks **directly in the browser** using WebAssembly, TensorFlow.js, and Python APIs. It evaluates the feasibility of in-browser ML by testing dataset size limitations, model complexity, and performance.

---

## 🚀 Features

- 🧮 Train models like Linear Regression and Neural Networks
- 🌐 WebAssembly + TensorFlow.js browser-based training
- 📊 Collect metrics: training time, inference time, MSE, R², accuracy, and loss
- 📈 Plot and visualize performance metrics
- ⚙️ Select dataset size and model complexity

---

## 🛠 Run the Web App

### 1. Extract Required Dataset

> 🔓 Make sure to extract `mnist_train_images.rar` at:
```
thesis-ml/neural_network/datasets/mnist_train_images.rar
```

---

### 2. Set Up Python Environment

```bash
python -m venv .venv                         # Create virtual environment

# For Windows
.venv\Scripts\activate

# For Ubuntu/macOS
source .venv/bin/activate
```

---

### 3. Install Requirements

```bash
pip install 'tensorflow[and-cuda]'          # For NVIDIA GPU support (optional, slow to install)
pip install pandas matplotlib scikit-learn
```

To check if TensorFlow detects your GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### 4. Run the Server

```bash
cd thesis-ml
python server.py
```

---

## 📁 Project Structure

```
thesis-ml/
│
├── assets/
│   ├── alpine_interface.js                         # Alpine.js grid logic
│   ├── app.js                                      # API handlers and utilities
│   ├── module_binding.js                           # Model bindings for runAll
│   └── style.css                                   # Basic styles for the interface
│
├── linear_regression/
│   ├── app/
│   │   └── python/
│   │       └── linear_Regression.py                # Python script for training the linear regression model using TensorFlow
│   ├── rust_wasm/
│   │   ├── pkg/
│   │   │   ├── rust_wasm_bg.wasm                   # WASM binary for running the model in the browser
│   │   │   └── rust_wasm.js                        # JavaScript code to run the WASM
│   │   └── src/
│   │       ├── lib.rs                              # Rust code for the linear regression model
│   │       └── rust_wasm_app.js                    # Binds buttons from interface to call the WASM
│   ├── tensorflow_js/
│   │   └── tensorflow_js_app.js                    # Binds buttons for training the model with TensorFlow.js
│   ├── datasets/
│   │   └── house_price/                            # Contains datasets for house rent price training
│   │       ├── sample_10%.csv
│   │       ├── sample_50%.csv
│   │       └── sample_100%.csv
│   ├── plot/
│   │   └── linear_regression_plot.py               # Plots the graphs and stores metrics of the results
│   └── training_result/                            # Directory structure for storing results based on process ID and platform
│       └── [process]                               # [proccess] is an id of the running process,
│           ├── [try]/[platform]                    # [try] is the current try, [platform] is the platform being executed
│           │   └── [comparisons graph|metric]      # here keeps comparison graphs and metric data files
│           └── confidence_interval_metric.json     # Stores the confidence interval for metrics across all tries
│
├── neural_network/
│   ├── app/
│   │   └── python/
│   │       └── neural_network.py                   # Python script for training the neural network using TensorFlow
│   ├── rust_wasm/
│   │   ├── pkg/
│   │   │   ├── rust_wasm_bg.wasm                   # WASM binary for running the neural network in the browser
│   │   │   └── rust_wasm.js                        # JavaScript code to run the WASM
│   │   └── src/
│   │       ├── lib.rs                              # Rust code for the neural network model
│   │       └── rust_wasm_app.js                    # Binds buttons from interface to call the WASM
│   ├── tensorflow_js/
│   │   └── tensorflow_js_app.js                    # Binds buttons for training the model with TensorFlow.js
│   ├── datasets/                                   # Dataset for the neural network training
│   │   ├── nab/                                    # this folder holds the samples to train the model for mnist dataset to be used in rust
│   │   │   ├── mnsit_iamges.nab
│   │   │   └── mnist_labels.nab
│   │   ├── mnist_test_images.json
│   │   ├── mnist_test_labels.json
│   │   ├── mnist_train_images.json
│   │   ├── mnist_train_images.rar                  # this is a rar holding mnist_train_images.json to reduce the size and can be uploaded on github
│   │   └── mnist_train_labels.json
│   ├── plot/
│   │   └── neural_network_plot.py
│   └── training_result/                            # Directory structure for storing results based on process ID and platform
│       └── [process]                               # [proccess] is an id of the running process,
│           ├── [try]/[platform]                    # [try] is the current try, [platform] is the platform being executed
│           │   └── [comparisons graph|metric]      # here keeps comparison graphs and metric data files
│           └── confidence_interval_metric.json     # Stores the confidence interval for metrics across all tries
│
├── index.html                  # Main interface
├── blank.html                  # blank.html is used for certain scenarios where there is an exception and redirects to it to clean resources
├── server.py                   # Runs the website and api to execute features
├── result_list.json            # Training result tracker
└── README.md                   # This file
```

---

## 🧪 Build & Debug WebAssembly (WASM)

> ⚠️ **Note:** You do **not** need to run this to use the app.  
> This step is only required **if you make code changes in the Rust modules**.

To compile WebAssembly modules from Rust:

```bash
wasm-pack build --target web
```

> 🛠 Make sure `wasm-pack` is installed:  
> Install via [Rust](https://rustup.rs):  
> `cargo install wasm-pack`

---

## 🎓 Academic Context

Built for my Master's thesis at Malta College of Arts, Science & Technology
