import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from pathlib import Path
from scipy import stats

def plot_loss(title, loss, val_loss, save_path):
    """Plots and saves the training and validation loss over epochs."""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(loss, color='blue', label='Training Loss')
    plt.plot(val_loss, color='red', label='Validation Loss')
    plt.title('Model Loss: ' + title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")

    
def plot_accuracy(title, accuracy, val_accuracy, save_path):
    """Plots and saves the training and validation accuracy over epochs."""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(accuracy, color='blue', label='Training Accuracy')
    plt.plot(val_accuracy, color='red', label='Validation Accuracy')
    plt.title('Model Accuracy: ' + title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy plot saved to {save_path}")

    
def plot_grouped_bar_comparisons(data, metrics, dataset_sizes, platforms, save_path):
    """
    Plots grouped bar charts to compare platforms for each dataset size.
    """
    for metric in metrics:
        x = np.arange(len(dataset_sizes))  # the label locations
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
        print(f"Comparison saved to {os.path.join(save_path, f"{metric}_comparison.png")}")

def create_csv(file_name, data):
    """Generates a CSV file with training metrics for each platform and dataset size."""
    headers = ['Platform', 'Dataset Size', 'Training Time (ms)', 'Inference Time (ms)', 'Accuracy', 'Loss']
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        dataset_sizes_order = ['10%', '50%', '100%']
        for dataset_size in dataset_sizes_order:
            if dataset_size in data['training_time']:  # Check if the dataset size exists in the data
                training_times = data['training_time'][dataset_size]
                for platform, training_time in training_times.items():
                    inference_time = data['inference_time'][dataset_size][platform]
                    accuracy = data['accuracy'][dataset_size][platform]
                    loss = data['loss'][dataset_size][platform]
                    writer.writerow([platform, dataset_size, training_time, inference_time, accuracy, loss])


def process_json_files(root_folder):
    """Processes JSON files in the root folder, extracts metrics, and generates plots and CSV."""
    metrics = ["training_time", "inference_time", "loss", "accuracy"]
    platform_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    predefined_platforms = ["python_gpu", "rust_wasm_cpu", "tensorflow_js_cpu", "tensorflow_js_webgpu", "tensorflow_js_wasm"]

    # Retrieve dataset sizes from the first platform folder
    first_platform_folder = os.path.join(root_folder, platform_folders[0])
    json_files = [re.search(r'(\d+%)', f.name).group(1) for f in Path(first_platform_folder).iterdir()
                  if f.is_file() and f.suffix == '.json' and f.stem.endswith(('10%', '50%', '100%'))]
    dataset_sizes = sorted(json_files, key=lambda x: int(x.replace('%', '')))  # Sort dataset sizes
    platforms = sorted(platform_folders, key=lambda x: predefined_platforms.index(x) if x in predefined_platforms else len(predefined_platforms))

    metric_data = {metric: {size: {} for size in dataset_sizes} for metric in metrics}

    """Processes JSON files in subfolders, plots regression lines, and saves them."""
    for framework_folder in os.listdir(root_folder):
        framework_path = os.path.join(root_folder, framework_folder)
        if os.path.isdir(framework_path):
            for json_file in os.listdir(framework_path):
                if json_file.endswith(('.json')):
                    file_path = os.path.join(framework_path, json_file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        match = re.match(r"([a-zA-Z0-9_]+)_sample_(\d+)%.json", os.path.basename(file_path).replace("nn_mnist_", "")) 
                        platform = match.group(1)
                        percentage = match.group(2) + "%"

                        for metric in metrics:
                            if metric not in metric_data:
                                metric_data[metric] = {}
                            if percentage not in metric_data[metric]:
                                metric_data[metric][percentage] = {}

                        loss_values = data.get('loss_values')
                        accuracy_values = data.get('accuracy_values')
                        val_loss_values = data.get('val_loss_values')
                        val_accuracy_values = data.get('val_accuracy_values')
                        metric_data["training_time"][percentage][platform] = round(data.get('training_time_ms') / 1000, 4)
                        metric_data["inference_time"][percentage][platform] = round(data.get('inference_time_ms') / 1000, 4)
                        metric_data["accuracy"][percentage][platform] = data.get('accuracy')
                        metric_data["loss"][percentage][platform] = data.get('loss') 
                        # print(loss_values)
                        # print(accuracy_values)
                        # print(val_loss_values)
                        # print(val_accuracy_values)
                        if loss_values and accuracy_values and val_loss_values and val_accuracy_values:
                            # Ensure loss_values, accuracy_values, val_loss_values and val_accuracy_values are lists of numbers
                            if all(isinstance(item, (float, int)) for item in loss_values) and \
                               all(isinstance(item, (float, int)) for item in accuracy_values) and \
                               all(isinstance(item, (float, int)) for item in val_loss_values) and \
                               all(isinstance(item, (float, int)) for item in val_accuracy_values):
 
                                # generate the save path.
                                plot_filename = os.path.splitext(json_file)[0].replace("nn_mnist_", "")
                                save_path = os.path.join(framework_path, plot_filename)

                                plot_loss(platform + " sample " + percentage, loss_values, val_loss_values, save_path + "_loss.png")
                                plot_accuracy(platform + " sample " + percentage, accuracy_values, val_accuracy_values, save_path + "_accuracy.png")
                            else:
                                print(f"Warning: loss_values, accuracy_values, val_loss_values, or val_accuracy_values in {file_path} are not valid lists of numbers.")

                        else:
                            print(f"Warning: Missing 'loss_values', 'accuracy_values', 'val_loss_values', or val_accuracy_values in {file_path}")

                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON in {file_path}")
                    except FileNotFoundError:
                        print(f"Error: File not found: {file_path}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {file_path}: {e}")
    
    plot_grouped_bar_comparisons(metric_data, metrics, dataset_sizes, platforms, root_folder)
    create_csv(os.path.join(root_folder, "metric.csv"), metric_data)
    return metric_data

# Function to calculate the confidence interval
def calculate_confidence_interval(data, confidence=0.95): 
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    n = len(data)
    
    # t-distribution critical value for 95% confidence level
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    # Standard error of the mean
    standard_error = std_dev / np.sqrt(n)
    
    # Confidence interval
    margin_of_error = t_critical * standard_error
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    print(data, ci_lower, ci_upper)
    return ci_lower, ci_upper

def save_confidence_interval(data, location): 
    # List of metrics (training_time, inference_time, mse, r2)
    metrics = ["training_time", "inference_time", "loss", "accuracy"]
    
    # Store results in a hierarchical JSON format
    json_data = {}

    # Loop through each metric and calculate confidence intervals
    for metric in metrics:
        print(f"Metric: {metric}")
        
        # Get the percentages available for the metric (e.g., "10%", "50%")
        percentages = list(data[metric].keys())  # get the percentages (keys under each metric)
        
        for percentage in percentages:
            # Get the platforms for the current percentage (e.g., tensorflow_js_webgpu)
            platforms = list(data[metric][percentage].keys())  # get the platforms
            for platform in platforms:
                # Get the values for the platform at the given percentage
                values = data[metric][percentage][platform]
                
                # Calculate the confidence interval
                ci_lower, ci_upper = calculate_confidence_interval(values)
                
                # Initialize the platform and dataset size if not present
                if platform not in json_data:
                    json_data[platform] = {}

                if percentage not in json_data[platform]:
                    json_data[platform][percentage] = {
                        "training_time": {},
                        "inference_time": {},
                        "loss": {},
                        "accuracy": {}
                    }
                
                # Add the confidence interval for the current metric
                if metric == "training_time":
                    json_data[platform][percentage]["training_time"] = {
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper
                    }
                elif metric == "inference_time":
                    json_data[platform][percentage]["inference_time"] = {
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper
                    }
                elif metric == "loss":
                    json_data[platform][percentage]["loss"] = {
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper
                    }
                elif metric == "accuracy":
                    json_data[platform][percentage]["accuracy"] = {
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper
                    }

    # Print the resulting JSON
    json_string = json.dumps(json_data, indent=4)
    with open(location, "w") as outfile:
        json.dump(json_data, outfile, indent=4)

def plot(result_item_id, tries):
    """
    Runs the process_json_files function multiple times based on the number of tries.
    """
    result_item_location = 'neural_network/training_result/' + str(result_item_id)
    metrics = ["training_time", "inference_time", "loss", "accuracy"] 
    metric_results = {}
    for i in range(tries): 
        data = process_json_files(result_item_location + '/' + str(i+1))
        for metric in metrics:  
            # Get the percentages available for the metric (e.g., "10%", "50%")
            percentages = list(data[metric].keys())  # get the percentages (keys under each metric) 

            for percentage in percentages: 
                # Get the platforms for the current percentage (e.g., tensorflow_js_webgpu)
                platforms = list(data[metric][percentage].keys())  # get the platforms

                for platform in platforms:
                    # Get the value for the platform at the given percentage
                    value = data[metric][percentage][platform] 

                    if metric not in metric_results:
                        metric_results[metric] = {}
                    if percentage not in metric_results[metric]:
                        metric_results[metric][percentage] = {}
                    if platform not in metric_results[metric][percentage]:
                        metric_results[metric][percentage][platform] = []
                    metric_results[metric][percentage][platform].append(value)

    save_confidence_interval(metric_results, result_item_location + "/confidence_interval_metric.json") 
