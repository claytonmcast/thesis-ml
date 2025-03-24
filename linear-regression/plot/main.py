import os
import json
import matplotlib.pyplot as plt
import re
import numpy
import csv

def plot_regression_line(title, features, target, predictions, save_path):
    """Plots the actual data and the regression line and saves it."""
    # Flatten the features list of lists
    #flat_features = [item[0] for item in features]

    plt.figure(figsize=(10, 6))
    plt.scatter(features, target, label='Data Points', color='blue')
    plt.plot(features, predictions, label='Regression Line', color='red')
    plt.xlabel("Features")
    plt.ylabel("Target/Predictions")
    plt.title("Regression Plot: " + title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"plot_regression_line saved to {save_path}")

    
def plot_loss_history(title, loss_history, save_path):
    plt.figure(figsize=(10, 6)) 
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History: ' + title)
    plt.legend()  # Add a legend
    plt.grid(True)  # Add grid lines
    plt.savefig(save_path)
    plt.close()
    print(f"plot_loss_history saved to {save_path}")

def plot_grouped_bar_comparisons(data, metrics, dataset_sizes, platforms, save_path):
    """
    Plots grouped bar charts to compare platforms for each dataset size.
    """
    print(data)
    for metric in metrics:
        x = numpy.arange(len(dataset_sizes))  # the label locations
        width = 0.8 / len(platforms)  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size as needed

        for i, platform in enumerate(platforms):
            values = [data[metric][size].get(platform, 0) for size in dataset_sizes]
            offset = (i - len(platforms) / 2 + 0.5) * width
            rects = ax.bar(x + offset, values, width, label=platform)
            ax.bar_label(rects, padding=3, fontsize=8)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison across Dataset Sizes')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_sizes)
        ax.legend()

        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f"{metric}_comparison.png")) #save the figure
        plt.close(fig) #close the figure to prevent memory issues.
        print(f"Loss history plot saved to {save_path}")

def create_csv(file_name, data):
    headers = ['Platform', 'Dataset Size', 'Training Time (ms)', 'Inference Time (ms)', 'MSE', 'R2']
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        # Sort dataset sizes in the desired order
        dataset_sizes_order = ['10%', '50%', '100%']

        for dataset_size in dataset_sizes_order:
            if dataset_size in data['training_time']: #check if the dataset size exists in the data.
                training_times = data['training_time'][dataset_size]
                for platform, training_time in training_times.items():
                    inference_time = data['inference_time'][dataset_size][platform]
                    accuracy = data['mse'][dataset_size][platform]
                    loss = data['r2'][dataset_size][platform]
                    row = [platform, dataset_size, training_time, inference_time, accuracy, loss]
                    writer.writerow(row)

def process_json_files(root_folder):
    """Processes JSON files in subfolders, plots regression lines, and saves them."""

    metrics = ["training_time", "inference_time", "mse", "r2"]
    dataset_sizes = ["10%", "50%", "100%"]
    platforms = ["python_gpu", "rust_wasm_cpu", "tensorflow_js_cpu", "tensorflow_js_webgpu", "tensorflow_js_wasm"]

    metric_data = {}

    for framework_folder in os.listdir(root_folder):
        framework_path = os.path.join(root_folder, framework_folder)
        if os.path.isdir(framework_path):
            for json_file in os.listdir(framework_path):
                if json_file.endswith(('.json')):
                    file_path = os.path.join(framework_path, json_file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        match = re.match(r"([a-zA-Z0-9_]+)_sample_(\d+)%.json", os.path.basename(file_path)) 
                        platform = match.group(1)
                        percentage = match.group(2) + "%"

                        for metric in metrics:
                            if metric not in metric_data:
                                metric_data[metric] = {}
                            if percentage not in metric_data[metric]:
                                metric_data[metric][percentage] = {}
                        
                        features = data.get('features')
                        target = data.get('target')
                        predictions = data.get('predictions')
                        loss_history = data.get('loss_history')  
                        metric_data["training_time"][percentage][platform] = data.get('training_time_ms')  
                        metric_data["inference_time"][percentage][platform] = data.get('inference_time_ms')
                        metric_data["mse"][percentage][platform] = data.get('mse')
                        metric_data["r2"][percentage][platform] = data.get('r2') 
                        
                        plot_filename = os.path.splitext(json_file)[0] + "_loss_history.png"
                        save_path = os.path.join(framework_path, plot_filename)
                        plot_loss_history(platform + " sample " + percentage, loss_history, save_path)

                        if features and target and predictions:
                            # Ensure features, target, and predictions are lists of numbers
                            if all(isinstance(item, list) and len(item) == 1 and isinstance(item[0], (int, float)) for item in features) and \
                               all(isinstance(item, list) and len(item) == 1 and isinstance(item[0], (int, float)) for item in target) and \
                               all(isinstance(item, list) and len(item) == 1 and isinstance(item[0], (int, float)) for item in predictions):

                                flat_features = [item[0] for item in features] 
                                flat_target = [item[0] for item in target]
                                flat_predictions = [item[0] for item in predictions]

                                # generate the save path.
                                plot_filename = os.path.splitext(json_file)[0] + "_regression_line.png"
                                save_path = os.path.join(framework_path, plot_filename)

                                plot_regression_line(platform + " sample " + percentage, flat_features, flat_target, flat_predictions, save_path)
                            else:
                                print(f"Warning: features, target, or predictions in {file_path} are not valid lists of numbers.")

                        else:
                            print(f"Warning: Missing 'features', 'target', or 'predictions' in {file_path}")

                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON in {file_path}")
                    except FileNotFoundError:
                        print(f"Error: File not found: {file_path}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {file_path}: {e}")
                        
    plot_grouped_bar_comparisons(metric_data, metrics, dataset_sizes, platforms, root_folder)
    create_csv(os.path.join(root_folder, "metric.csv"), metric_data)

process_json_files('../training_result')