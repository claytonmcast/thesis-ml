import os
import json
import matplotlib.pyplot as plt
import numpy
import re
import csv

def plot_loss(title, loss, val_loss, save_path):
    """Plots the actual data and the regression line and saves it."""
    # Flatten the features list of lists
    #flat_features = [item[0] for item in features]

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
    print(f"plot_loss saved to {save_path}")

    
def plot_accuracy(title, accuracy, val_accuracy, save_path):
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
    print(f"plot_accuracy plot saved to {save_path}")

    
def plot_grouped_bar_comparisons(data, metrics, dataset_sizes, platforms, save_path):
    """
    Plots grouped bar charts to compare platforms for each dataset size.
    """
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
    headers = ['Platform', 'Dataset Size', 'Training Time (ms)', 'Inference Time (ms)', 'Accuracy', 'Loss']
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
                    accuracy = data['accuracy'][dataset_size][platform]
                    loss = data['loss'][dataset_size][platform]
                    row = [platform, dataset_size, training_time, inference_time, accuracy, loss]
                    writer.writerow(row)

def process_json_files(root_folder):
    metrics = ["training_time", "inference_time", "loss", "accuracy"]
    dataset_sizes = ["10%", "50%", "100%"]
    platforms = ["python_gpu", "rust_wasm_cpu", "tensorflow_js_cpu", "tensorflow_js_webgpu", "tensorflow_js_wasm"]

    metric_data = {}

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
                        metric_data["training_time"][percentage][platform] = data.get('training_time_ms')  
                        metric_data["inference_time"][percentage][platform] = data.get('inference_time_ms')
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

process_json_files('../training_result')