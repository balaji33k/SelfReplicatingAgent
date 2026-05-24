import requests
from requests.auth import HTTPBasicAuth
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
import socket
import pytest

# Constants
HOST = 'localhost'
PORT = 8080


class MockServerRequestHandler(BaseHTTPRequestHandler):
    """
    A mock HTTP server to simulate a 401 challenge and authentication.
    """

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"OK")

    def do_POST(self):
        if self.headers.get('Authorization') is None:
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="Test Realm"')
            self.end_headers()
        else:
            # Simulate successful authentication on retry
            auth = self.headers.get('Authorization').split(" ")[1]
            if auth == "Basic dXNlcjpwYXNzd29yZA==":  # user:password in base64
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Authenticated")
            else:
                 self.send_response(401)
                 self.send_header('WWW-Authenticate', 'Basic realm="Test Realm"')
                 self.end_headers()


class MockServer(threading.Thread):
    """
    A thread that runs the mock HTTP server.
    """

    def __init__(self, host=HOST, port=PORT):
        super().__init__()
        self.httpd = HTTPServer((host, port), MockServerRequestHandler)
        self.url = f"http://{host}:{port}"
        self._shutdown_flag = threading.Event()

    def run(self):
        while not self._shutdown_flag.is_set():
            self.httpd.handle_request() #Handle one request at a time for testing
        self.httpd.server_close()


    def shutdown(self):
        self._shutdown_flag.set()
        # Need to make a request to unblock handle_request
        try:
            requests.get(self.url)  # Any request will do to unblock
        except requests.exceptions.RequestException:
            pass  # Ignore connection errors during shutdown


def chunked_upload(url, auth, data):
    """
    Simulates a chunked upload with authentication.
    """
    try:
        response = requests.post(url, auth=auth, data=data, stream=True)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def test_chunked_upload_with_401():
    """
    Test case to verify chunked upload with 401 authentication challenge.
    """
    # Start the mock server
    server = MockServer()
    server.start()
    time.sleep(0.1)  # Give the server some time to start


    # Define the data to be uploaded (as a generator for chunked upload)
    def data_generator():
        yield b"This is the first chunk.\n"
        yield b"This is the second chunk.\n"
        yield b"This is the third chunk.\n"

    # Define the authentication
    auth = HTTPBasicAuth('user', 'password')

    # Perform the chunked upload
    response = chunked_upload(server.url, auth, data_generator())

    # Assertions to verify correct behavior
    assert response is not None, "Response should not be None"

    if response:
        assert response.status_code == 200, f"Expected 200 OK, but got {response.status_code}"
        assert response.text == "Authenticated", "Expected 'Authenticated' in response"


    # Shutdown the server
    server.shutdown()
    server.join()
    print("Test completed successfully.")


if __name__ == '__main__':
    test_chunked_upload_with_401()