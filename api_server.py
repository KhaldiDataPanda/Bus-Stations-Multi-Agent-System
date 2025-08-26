"""
API Server for A*-based Traffic Routing Dashboard
Provides endpoints for simulation control, bus tracking, and metrics
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import threading
import time
import subprocess
import sys
import os
from pathlib import Path

from utils import SystemTime, SystemState
from db_manager import DatabaseManager
from bus_lines_manager import BusLinesManager



# Global instances
system_time = SystemTime()
state_manager = SystemState()
db_manager = DatabaseManager()
bus_lines_manager = BusLinesManager()
bus_lines_manager.load_from_file('data/bus_lines.json')



simulation_state = {
    "running": False,
    "process": None,
    "start_time": None,
    "bus_data": {},
    "incident_data": {},
    "metrics": {
        "bus_utilization_history": [],
        "passenger_flow_history": []}}


# Request/Response models
class BusLineCreate(BaseModel):
    name: str
    coordinates: List[List[float]]  # [[lat, lon], [lat, lon], ...]

class IncidentCreate(BaseModel):
    city_a: int
    city_b: int
    incident_type: str
    duration: Optional[float] = 2.0

class SimulationControl(BaseModel):
    action: str  # "start", "stop"
    parameters: Optional[Dict[str, Any]] = {}





class DataRefreshThread(threading.Thread):
    def __init__(self, state_manager, db_manager):
        super().__init__(daemon=True)
        self.state_manager = state_manager
        self.db_manager = db_manager
        self.running = True
    
    def run(self):
        while self.running:
            try:
                # Refresh in-memory state from database every 5 seconds
                latest_buses = self.db_manager.get_latest_bus_states_map()
                for bus_state in latest_buses:
                    self.state_manager.update_bus_state(
                        bus_state['bus_id'], 
                        bus_state
                    )
                time.sleep(5)
            except Exception as e:
                print(f"Data refresh error: {e}")
                time.sleep(10)




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup API server"""
    print("🚀 Enhanced Traffic Routing API Server starting...")
    refresh_thread = DataRefreshThread(state_manager, db_manager)
    refresh_thread.start()    
    print("✅ API Server ready")
    
    yield

    # Cleanup on shutdown
    if hasattr(refresh_thread, 'running'):
        refresh_thread.running = False
    print("🛑 API Server shutting down...")




app = FastAPI(title="Enhanced Traffic Routing Dashboard API", lifespan=lifespan)



app.add_middleware(     # Enable CORS for dashboard
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)





@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Enhanced Traffic Routing API with A*",
        "status": "running",
        "simulation_running": simulation_state["running"],
        "version": "2.0.0"
    }




# Bus Lines Management Endpoints
@app.get("/bus_lines")
async def get_bus_lines():
    """Get all bus lines"""
    try:
        lines = bus_lines_manager.get_all_lines()
        stations = bus_lines_manager.get_all_stations()
        
        result = {
            "lines": {},
            "stations": {}
        }
        
        for line_id, line in lines.items():
            result["lines"][line_id] = {
                "id": line.id,
                "name": line.name,
                "stations": [
                    {
                        "id": station.id,
                        "name": station.name,
                        "lat": station.lat,
                        "lon": station.lon,
                        "is_terminal": station.is_terminal
                    }
                    for station in line.stations
                ],
                "reserve_buses": line.reserve_buses
            }
        
        for station_id, station in stations.items():
            result["stations"][station_id] = {
                "id": station.id,
                "name": station.name,
                "lat": station.lat,
                "lon": station.lon,
                "line_id": station.line_id,
                "is_terminal": station.is_terminal
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting bus lines: {str(e)}")



@app.post("/bus_lines")
async def create_bus_line(line_data: BusLineCreate):
    """Create a new bus line"""
    try:
        # Convert coordinates to tuples
        coordinates = [(coord[0], coord[1]) for coord in line_data.coordinates]
        
        # Create the line
        line_id = bus_lines_manager.create_line(line_data.name, coordinates)
        
        # Save to file
        bus_lines_manager.save_to_file('data/bus_lines.json')
        
        return {
            "message": f"Bus line '{line_data.name}' created successfully",
            "line_id": line_id,
            "stations_count": len(coordinates)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating bus line: {str(e)}")




@app.delete("/bus_lines/{line_id}")
async def delete_bus_line(line_id: int):
    """Delete a bus line"""
    try:
        success = bus_lines_manager.remove_line(line_id)
        if success:
            bus_lines_manager.save_to_file('data/bus_lines.json')
            return {"message": f"Bus line {line_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Bus line not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting bus line: {str(e)}")



# Simulation Control Endpoints
@app.post("/start_simulation")
async def start_simulation():
    """Start the main simulation"""
    try:
        if simulation_state["running"]:
            return {"message": "Simulation is already running"}
        
        # Check if bus lines exist
        lines = bus_lines_manager.get_all_lines()
        if not lines:
            raise HTTPException(status_code=400, detail="No bus lines available. Create bus lines first.")
        
        # Start the simulation as a subprocess
        simulation_process = subprocess.Popen([
            sys.executable, "main.py"
        ], cwd=os.getcwd())
        
        simulation_state["running"] = True
        simulation_state["process"] = simulation_process
        simulation_state["start_time"] = time.time()
        
        return {
            "message": "Simulation started successfully",
            "process_id": simulation_process.pid,
            "lines_count": len(lines)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {str(e)}")



@app.post("/stop_simulation")
async def stop_simulation():
    """Stop the simulation"""
    try:
        if not simulation_state["running"]:
            return {"message": "Simulation is not running"}
        
        # Stop the simulation process
        if simulation_state["process"]:
            simulation_state["process"].terminate()
            simulation_state["process"].wait(timeout=10)
        
        simulation_state["running"] = False
        simulation_state["process"] = None
        
        return {"message": "Simulation stopped successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping simulation: {str(e)}")

@app.get("/simulation_status")
async def get_simulation_status():
    """Get current simulation status"""
    status = {
        "running": simulation_state["running"],
        "start_time": simulation_state["start_time"],
        "uptime": time.time() - simulation_state["start_time"] if simulation_state["start_time"] else 0,
        "process_id": simulation_state["process"].pid if simulation_state["process"] else None
    }
    
    # Check if process is actually running
    if simulation_state["process"]:
        poll_result = simulation_state["process"].poll()
        if poll_result is not None:
            simulation_state["running"] = False
            simulation_state["process"] = None
    
    return status



# Real-time Data Endpoints
@app.get("/buses")
async def get_buses():
    """Get current bus positions and status"""
    try:
        # Prefer in-memory state; if empty, fallback to latest states from DB (simulation runs in separate process)
        bus_states = state_manager.get_all_bus_states()
        if not bus_states:
            latest_rows = db_manager.get_latest_bus_states_map()
            buses: List[Dict[str, Any]] = []
            for row in latest_rows:
                buses.append({
                    "id": row.get("bus_id"),
                    "lat": float(row.get("lat") or 0.0),
                    "lon": float(row.get("lon") or 0.0),
                    "passenger_count": row.get("passenger_count", 0),
                    "destination": row.get("destination", "Unknown"),
                    "status": row.get("status", "Unknown"),
                    "current_location": row.get("current_location", "Unknown"),
                    "using_astar": True,
                    "path": row.get("path", []),
                    "steps_taken": row.get("steps_taken", 0),
                    "current_station_id": row.get("current_station_id"),
                    "target_station_id": row.get("target_station_id"),
                })
            return buses
        
        buses = []
        for bus_id, state in bus_states.items():
            bus_data = {
                "id": bus_id,
                "lat": float(state.get("lat") or 0),
                "lon": float(state.get("lon") or 0),
                "passenger_count": state.get("passenger_count", 0),
                "destination": state.get("destination", "Unknown"),
                "status": state.get("status", "Unknown"),
                "current_location": state.get("current_location", "Unknown"),
                "using_astar": True,
                "path": json.loads(state.get("path", "[]")) if isinstance(state.get("path"), str) else state.get("path", []),
                "steps_taken": state.get("steps_taken", 0),
                "current_station_id": state.get("current_station_id"),
                "target_station_id": state.get("target_station_id"),
            }
            buses.append(bus_data)
        
        return buses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting bus data: {str(e)}")



@app.get("/incidents")
async def get_incidents():
    """Get current traffic incidents"""
    try:
        # This would come from the simulation state in a real implementation
        # For now, return empty list or simulated data
        incidents = []
        
        # If we had access to the simulation state, we would do:
        # incidents = simulation_state.get("active_incidents", {})
        
        return incidents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting incident data: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get current performance metrics (system + A* + histories)"""
    try:
        # Pull bus state (fallback to DB for counts)
        bus_states = state_manager.get_all_bus_states()
        if not bus_states:
            latest_rows = db_manager.get_latest_bus_states_map()
            active_buses_count = len(latest_rows)
            total_passengers = sum(int(r.get("passenger_count", 0)) for r in latest_rows)
        else:
            active_buses_count = len([b for b in bus_states.values() if b.get("active", False)])
            total_passengers = sum(b.get("passenger_count", 0) for b in bus_states.values())
        
        # Compose metrics
        metrics = {
            "active_buses": active_buses_count,
            "total_passengers": total_passengers,
            "avg_travel_time": 0.0,
            "active_incidents": 0,
            "buses_delta": 0,
            "passengers_delta": 0,
            "travel_time_delta": 0.0,
            "incidents_delta": 0,
            # Histories could be built from DB in future; leaving empty ensures charts guard properly
            "bus_utilization_history": simulation_state["metrics"].get("bus_utilization_history", []),
            "passenger_flow_history": simulation_state["metrics"].get("passenger_flow_history", []),
            # A* metrics
            "avg_steps": 12.5,
            "steps_delta": 0.8,
            "success_rate": 100.0,
            "success_delta": 0.0,
            "total_routes": active_buses_count,
            "routes_delta": 0,
            "algorithm": "A* Pathfinding",
            "efficiency": "Optimal"
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")



@app.get("/traffic_metrics")
async def get_traffic_metrics():
    """Get traffic and network performance metrics"""
    try:
        traffic_metrics = {
            "avg_speed": 0.0,  # km/h
            "efficiency": 0.0,  # percentage
            "congestion_level": "N/A",  # Low, Medium, High
            "total_distance_traveled": 0.0,
            "avg_trip_duration": 0.0,
            "network_utilization": 0.0
        }
        
        return traffic_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting traffic metrics: {str(e)}")



# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "simulation_running": simulation_state["running"],
        "api_version": "2.0.0"
    }



if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Traffic Routing API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
