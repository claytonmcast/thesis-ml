import http.server
import socketserver

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Set MIME type for .wasm files
        #if self.path.endswith('.wasm'):
         #   self.send_header('Content-Type', 'application/wasm')
        super().end_headers()

# Use the custom handler to serve files
handler = MyHTTPRequestHandler
with socketserver.TCPServer(("", 8001), handler) as httpd:
    print("Serving at port 8001")
    httpd.serve_forever()
