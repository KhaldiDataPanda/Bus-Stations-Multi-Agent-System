"""
Launcher script for the Traffic Routing Dashboard System
Starts the simulation, API server, and Streamlit dashboard
"""
import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def run_command(command, name, cwd=None):
    """Run a command in a separate process"""
    print(f"Starting {name}...")
    try:
        if cwd:
            process = subprocess.Popen(
                command, 
                shell=True, 
                cwd=cwd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
        return process
    except Exception as e:
        print(f"Error starting {name}: {e}")
        return None

def main():
    print("🚌 Traffic Routing Dashboard System Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("Error: Please run this script from the App directory")
        sys.exit(1)
    
    processes = []
    
    try:
        # Start API server
        print("\n1. Starting FastAPI server...")
        api_process = run_command(
            "python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload",
            "FastAPI Server"
        )
        if api_process:
            processes.append(("FastAPI Server", api_process))
            time.sleep(3)  # Give the API server time to start
        
        # Start Streamlit dashboard
        print("\n2. Starting Streamlit dashboard...")
        dashboard_process = run_command(
            "streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0",
            "Streamlit Dashboard"
        )
        if dashboard_process:
            processes.append(("Streamlit Dashboard", dashboard_process))
            time.sleep(2)
        
        print("\n" + "=" * 50)
        print("🎉 System Started Successfully!")
        print("📊 Dashboard: http://localhost:8501")
        print("🔌 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("\nTo start the simulation, use the dashboard controls or run:")
        print("python main.py")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for processes and monitor them
        while True:
            time.sleep(1)
            # Check if any process has died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n⚠️  {name} has stopped unexpectedly")
                    break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        
        # Terminate all processes
        for name, process in processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
        
        print("✅ All services stopped")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Clean up processes
        for name, process in processes:
            try:
                process.terminate()
            except:
                pass

if __name__ == "__main__":
    main()
