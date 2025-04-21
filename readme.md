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
    - `[1]/[1]/[platform]`: Directory structure for storing results based on transaction ID and platform
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
    - `[1]/[1]/[platform]`: Directory structure for storing results based on transaction ID and platform
    - `confidence_interval_metric.json`: Stores the confidence interval for metrics across all tries

- `blank.html`: Used for certain scenarios where there is an exception and redirects to clean resources
- `index.html`: The main interface to train the datasets
- `README.md`: Explanation of the application
- `result_list.json`: Database to store training status
- `server.py`: Runs the website and API for executing features


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
│   └── training_result/                            # Directory structure for storing results based on transaction ID and platform
│       ├── [transaction]/[try]/[platform]/
│       └── [transaction]/confidence_interval_metric.json       # Stores the confidence interval for metrics across all tries
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
│   ├── datasets/
│   │   └── mnist/                                  # Dataset for the neural network training
│   │       ├── mnist_train_images.json
│   │       └── mnist_train_labels.json
│   ├── plot/
│   │   └── neural_network_plot.py
│   └── training_result/
│       ├── [transaction]/[try]/[platform]/
│       └── [transaction]/confidence_interval_metric.json       # Stores the confidence interval for metrics across all tries
│
├── index.html                  # Main interface
├── blank.html                  # Redirect fallback on error
├── server.py                   # Flask API server
├── result_list.json            # Training result tracker
└── README.md                   # This file
```

---

## 🧪 Build & Debug WebAssembly (WASM)

To compile WebAssembly modules from Rust:

```bash
wasm-pack build --target web
```

> 🛠 Make sure `wasm-pack` is installed:  
> Install via [Rust](https://rustup.rs):  
> `cargo install wasm-pack`

---

## 📬 Contact

Built with ❤️ for my Master's thesis at [Your University Name].  
If you'd like to discuss the project or collaboration, feel free to reach out!
