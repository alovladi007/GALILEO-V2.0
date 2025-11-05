#!/usr/bin/env python3
"""
Simple HTTP server for serving the emulator dashboard
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    """Start the HTTP server"""
    print("=" * 60)
    print("  Optical Bench Emulator - Dashboard Server")
    print("=" * 60)
    print(f"\nServing dashboard at: http://localhost:{PORT}")
    print(f"Dashboard URL: http://localhost:{PORT}/dashboard.html")
    print("\nMake sure the WebSocket server is running on ws://localhost:8765")
    print("Start it with: python server.py")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            # Try to open browser automatically
            try:
                webbrowser.open(f'http://localhost:{PORT}/dashboard.html')
            except:
                pass
            
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
