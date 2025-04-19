# pip install rarfile
# sudo apt install unrar

import http.server
import socketserver
import json
import os
import socket
import rarfile
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from linear_regression.app.python import linear_regression
from linear_regression.plot import linear_regression_plot
from neural_network.app.python import neural_network
from neural_network.plot import neural_network_plot

def extract_if_not_exists(target_file, rar_path):
    if os.path.exists(target_file):
        print(f"{target_file} already exists. Skipping extraction.")
        return

    with rarfile.RarFile(rar_path) as rf:
        print(f"Extracting {rar_path}...")
        rf.extractall(path=os.path.dirname(rar_path))
        print("Extraction complete.")

# this extract a rar which is big for github to have as a raw file
extract_if_not_exists('neural_network/datasets/mnist_train_images.json', 'neural_network/datasets/mnist_train_images.rar')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler): 
    # Handle GET requests
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/new_result_item':
            result = self.new_result_item()  # Create a new result item
            self.response(result)
        elif parsed_path.path == '/api/run_python':
            self.run_python()  # Run Python script
        elif parsed_path.path == '/api/plot_linear_regression':
            self.plot_linear_regression()  # Plot linear regression
        elif parsed_path.path == '/api/plot_neural_network':
            self.plot_neural_network()  # Plot neural network
        else:
            super().do_GET()  # Default behavior for other GET requests

    # Handle POST requests
    def do_POST(self):
        if self.path == '/api/save_json_object':
            self.save_json_object()  # Save a JSON object to file
        elif self.path == '/api/append_experiment':
            self.append_experiment()  # Append experiment to result list
        elif self.path == '/api/update_result_item':
            self.update_result_item()  # Update result item information
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')  # Handle invalid routes

    def save_json_file(self, location, data):
        # Ensure the directory exists 
        # Get directory name from location
        directory = os.path.dirname(location)

        # Only try to create the directory if it's not empty (i.e., not just a filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save the JSON file
        with open(location, 'w') as file:
            json.dump(data, file, indent=4)

    # Save the JSON object from the request body
    def save_json_object(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data) 
            location = data['location']
            jsonObject = data['jsonObject']
           
            self.save_json_file(location, jsonObject)
            self.response({})

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')

    # Plot the linear regression graph based on query parameters
    def plot_linear_regression(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        id = query_params.get('id', [None])[0]
        tries = query_params.get('tries', [None])[0]
        linear_regression_plot.plot(int(id), int(tries))
        self.response({})
    
    # Plot the neural network graph based on query parameters
    def plot_neural_network(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        id = query_params.get('id', [None])[0]
        tries = query_params.get('tries', [None])[0]
        neural_network_plot.plot(int(id), int(tries))
        self.response({})
    
    # Retrieve the result list from the JSON file
    def get_result_list(self):
        if not os.path.exists('result_list.json'):
            return []
        with open('result_list.json', 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []

    # Run the Python model (linear regression or neural network)
    def run_python(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        type = query_params.get('type', [None])[0]
        retry = query_params.get('try', [None])[0]
        sample = query_params.get('sample', [None])[0]
        dataset = int(query_params.get('dataset', [None])[0])
        result_item_id = int(query_params.get('result_item_id', [None])[0])
 
        if(type == 'Linear Regression Python GPU'):
            data = linear_regression.process(dataset, retry, sample, result_item_id)
        else :
            data = neural_network.process(dataset, retry, sample, result_item_id)

        self.append_experiment_to_result_list(data)

        self.response({})

    # Send a JSON response back to the client
    def response(self, response_obj):
        response_json = json.dumps(response_obj).encode('utf-8')

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json)

    # Create a new result item and append it to the result list
    def new_result_item(self):
        data = self.get_result_list()
        sorted_data = sorted(data, key=lambda x: x['id'])
        if sorted_data:
            last_id = sorted_data[-1]['id']
            new_id = last_id + 1
        else:
            new_id = 1

        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        new_item = {'id': new_id, 'tries': int(query_params.get('tries', [None])[0]), "isRunAll": query_params.get('isRunAll', [None])[0], "start": query_params.get('start', [None])[0],  'experiments': []}
        data.append(new_item)
        self.save_json_file('result_list.json', data)
        return new_item

    # Append the experiment data to the result list
    def append_experiment_to_result_list(self, data):
        experiment = data['experiment'];
        self.save_json_file(experiment['result_path'], data['results'])

        result_list = self.get_result_list()
        for item in result_list:
            if item.get('id') == experiment['result_item_id']:
                if 'experiments' in item and isinstance(item['experiments'], list):
                    item['experiments'].append(experiment)
                else:
                    item['experiments'] = [experiment]
                break 

        self.save_json_file("result_list.json", result_list)

    # Append experiment data (received in the request) to the result list
    def append_experiment(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body) 
            self.append_experiment_to_result_list(data)
 
            self.response({})
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return
        
    # Update the result item with new start and end timestamps
    def update_result_item(self): 
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body) 
            result_list = self.get_result_list()
            for item in result_list:
                if item.get('id') == data['result_item_id']:
                    item['end'] = data['end'];
                    break 
            self.response({})
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return
 
        self.save_json_file("result_list.json", result_list)


    def end_headers(self):
        # Set MIME type for .wasm files
        #if self.path.endswith('.wasm'):
         #   self.send_header('Content-Type', 'application/wasm')
        # Set caching headers for specific files
        if self.path.endswith((".nab", "sample_10%.csv", "sample_50%.csv", "mnist_test_images.json", "mnist_test_labels.json", "mnist_test_labels.json", "mnist_train_images.json", "mnist_train_labels.json")):
            print("------------------------------------------")
            print("header", self.path)
            print("------------------------------------------")
            expires = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
            self.send_header("Cache-Control", "public, max-age=2592000")  # 30 days
            self.send_header("Expires", expires)
            
        if self.path.endswith("result_list.json"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

# Function to start the server and handle retries in case of errors
def start_server(handler, port=8001, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            with socketserver.TCPServer(("", port), handler) as httpd:
                print(f"Serving at port {port}")
                httpd.serve_forever()
                return  # Successfully started the server
        except socket.error as e:
            # If port is already in use, try the next one
            print(f"Port {port} is in use. Trying next port...")
            port += 1
            retries += 1
    
    print("Max retries reached. Could not start the server.")
    return None


# Start the server on port 8001
start_server(MyHTTPRequestHandler, port=8001)