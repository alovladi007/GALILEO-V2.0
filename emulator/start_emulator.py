#!/usr/bin/env python3
"""
Master Startup Script for Optical Bench Emulator
Starts both WebSocket server and HTTP dashboard server
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Process handles
processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Shutting down emulator system...")
    
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    
    print("✓ All processes stopped")
    sys.exit(0)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import numpy
        import websockets
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        return False

def start_websocket_server():
    """Start the WebSocket emulator server"""
    print("🚀 Starting WebSocket emulator server...")
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    processes.append(proc)
    return proc

def start_dashboard_server():
    """Start the HTTP dashboard server"""
    print("🚀 Starting dashboard HTTP server...")
    proc = subprocess.Popen(
        [sys.executable, "dashboard_server.py"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    processes.append(proc)
    return proc

def main():
    """Main startup routine"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "█" * 60)
    print("  OPTICAL BENCH EMULATOR - SYSTEM STARTUP")
    print("█" * 60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start servers
    ws_server = start_websocket_server()
    time.sleep(1)  # Give WebSocket server time to start
    
    dashboard_server = start_dashboard_server()
    time.sleep(1)  # Give dashboard server time to start
    
    print("\n" + "=" * 60)
    print("  ✓ EMULATOR SYSTEM RUNNING")
    print("=" * 60)
    print("\n📡 WebSocket Server: ws://localhost:8765")
    print("🌐 Dashboard: http://localhost:8080/dashboard.html")
    print("\n💡 The dashboard should open automatically in your browser")
    print("   If not, manually navigate to the URL above")
    print("\n⚙️  Available demo scripts:")
    print("   • python demo_basic.py      - Basic operation demo")
    print("   • python demo_events.py     - Event injection demo")
    print("   • python demo_streaming.py  - Streaming data demo")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 60 + "\n")
    
    # Keep running and monitor processes
    try:
        while True:
            # Check if processes are still running
            for proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  Process {proc.pid} exited unexpectedly")
                    print("Shutting down...")
                    signal_handler(None, None)
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    # Change to script directory
    os.chdir(Path(__file__).parent)
    main()
