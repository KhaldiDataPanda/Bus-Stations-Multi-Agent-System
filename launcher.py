"""
Enhanced Launcher for Traffic Routing Dashboard System with RL
Starts the API server and Streamlit dashboard (simulation started from dashboard)
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
    print("🚌 Enhanced Traffic Routing Dashboard System Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("Error: Please run this script from the App directory")
        sys.exit(1)
    
    processes = []
    
    try:
        # Start API server
        print("\n🔧 Starting API server...")
        api_process = run_command("python api_server.py", "API Server")
        if api_process:
            processes.append(("API Server", api_process))
            print("✅ API Server started on http://localhost:8000")
        else:
            print("❌ Failed to start API server")
            return
        
        # Wait a moment for API server to start
        time.sleep(3)
        
        # Start Streamlit dashboard
        print("\n🖥️  Starting Streamlit dashboard...")
        dashboard_process = run_command("streamlit run dashboard.py --server.port 8501", "Dashboard")
        if dashboard_process:
            processes.append(("Dashboard", dashboard_process))
            print("✅ Dashboard started on http://localhost:8501")
        else:
            print("❌ Failed to start dashboard")
            return
        
        print("\n" + "=" * 60)
        print("🎉 SYSTEM READY!")
        print("=" * 60)
        print("📊 Dashboard: http://localhost:8501")
        print("🔗 API Docs: http://localhost:8000/docs")
        print("\n📝 INSTRUCTIONS:")
        print("1. Open the dashboard at http://localhost:8501")
        print("2. Go to 'Line Creation' page to create bus lines")
        print("3. Click on the map to add stations (minimum 2 stations per line)")
        print("4. Create multiple lines as needed")
        print("5. Go to 'Simulation Control' to start the RL-enhanced simulation")
        print("6. Monitor real-time progress and RL performance on various dashboard pages")
        print("\n⚠️  The simulation uses online RL training - performance improves over time!")
        print("\n🛑 Press Ctrl+C to stop all services")
        print("=" * 60)
        
        # Monitor processes
        while True:
            time.sleep(5)
            
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n⚠️  {name} stopped unexpectedly")
                    # Try to restart
                    if name == "API Server":
                        new_process = run_command("python api_server.py", "API Server")
                        if new_process:
                            processes = [(n, p) for n, p in processes if n != name]
                            processes.append((name, new_process))
                            print(f"✅ {name} restarted")
                    elif name == "Dashboard":
                        new_process = run_command("streamlit run dashboard.py --server.port 8501", "Dashboard")
                        if new_process:
                            processes = [(n, p) for n, p in processes if n != name]
                            processes.append((name, new_process))
                            print(f"✅ {name} restarted")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down system...")
        
    finally:
        # Clean shutdown
        for name, process in processes:
            print(f"Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("\n✅ System shutdown complete")
        print("Thank you for using the Enhanced Traffic Routing System! 🚌")

if __name__ == "__main__":
    main()
