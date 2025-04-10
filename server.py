import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/new_result_item':
            self.new_result_item()
        else:
            super().do_GET()
 
    def do_POST(self):
        if self.path == '/api/save_json_object':
            self.save_json_object()
        if self.path == '/api/append_experiment':
            self.append_experiment()
        else:
            super().do_POST()

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


    def save_json_object(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data) 
            location = data['location']
            jsonObject = data['jsonObject']
           
            self.save_json_file(location, jsonObject)

            response_json = json.dumps({}).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
    
    def get_result_list(self):
        if not os.path.exists('result_list.json'):
            return []
        with open('result_list.json', 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []

            
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

        new_item = {'id': new_id, 'tries': int(query_params.get('tries', [None])[0]), 'experiments': []}
        data.append(new_item)
        self.save_json_file('result_list.json', data)
        response_json = json.dumps(new_item).encode('utf-8')

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json)

    def append_experiment(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body) 
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

            
            response_json = json.dumps({}).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return


    def end_headers(self):
        # Set MIME type for .wasm files
        #if self.path.endswith('.wasm'):
         #   self.send_header('Content-Type', 'application/wasm')
        super().end_headers()

# Use the custom handler to serve files
handler = MyHTTPRequestHandler
port = 8004
with socketserver.TCPServer(("", port), handler) as httpd:
    print("Serving at port " + str(port))
    httpd.serve_forever()
