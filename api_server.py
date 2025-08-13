from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional
from utils import SystemTime, SystemState
from db_manager import DatabaseManager
from metrics_exporter import MetricsExporter, get_exportable_metrics
from simulation_manager import sim_manager
import threading
import time

app = FastAPI(title="Traffic Routing Dashboard API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
system_time = None
state_manager = None
db_manager = None
simulation_control = {"running": False, "speed": 1.0}

class IncidentCreate(BaseModel):
    city_a: int
    city_b: int
    incident_type: str

class SimulationControl(BaseModel):
    action: str  # "start", "pause", "stop"
    speed: Optional[float] = 1.0

@app.on_event("startup")
async def startup_event():
    global system_time, state_manager, db_manager
    system_time = SystemTime()
    state_manager = SystemState()
    db_manager = DatabaseManager()

@app.get("/")
async def root():
    return {"message": "Traffic Routing Dashboard API"}

@app.get("/api/system/status")
async def get_system_status():
    """Get current system status and time"""
    return {
        "current_time": system_time.get_current_time() if system_time else 0,
        "simulation_running": simulation_control["running"],
        "simulation_speed": simulation_control["speed"],
        "active_incidents": len(system_time.incidents) if system_time else 0
    }

@app.get("/api/metrics/export")
async def export_metrics():
    """Export current metrics to CSV and return file"""
    try:
        exporter = MetricsExporter()
        filepath = exporter.export_to_csv()
        return FileResponse(
            path=str(filepath),
            filename=filepath.name,
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/metrics/detailed-export")
async def export_detailed_states():
    """Export detailed bus and station states to CSV"""
    try:
        exporter = MetricsExporter()
        filepath = exporter.export_detailed_states()
        return FileResponse(
            path=str(filepath),
            filename=filepath.name,
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get performance summary for dashboard"""
    try:
        summary = get_exportable_metrics()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.get("/api/system/status")
async def get_system_status():
    """Get current system status and time"""
    return {
        "current_time": system_time.get_current_time() if system_time else 0,
        "simulation_running": simulation_control["running"],
        "simulation_speed": simulation_control["speed"],
        "active_incidents": len(system_time.incidents) if system_time else 0
    }

@app.get("/api/buses")
async def get_buses():
    """Get all bus states with enhanced information"""
    if not state_manager:
        return {"buses": []}
    
    bus_states = state_manager.get_bus_states()
    buses = []
    
    for bus_id, state in bus_states.items():
        # Calculate progress and additional metrics
        current_city = state.get("current_city")
        next_city = state.get("next_city")
        distance_to_next = state.get("distance_to_next", 0)
        
        # Estimate progress (0-1) based on remaining distance
        # Assuming average route distance of 100km for calculation
        total_distance = 100  # This should come from distance matrix
        progress = max(0, min(1, (total_distance - distance_to_next) / total_distance)) if total_distance > 0 else 0
        
        # Generate realistic passenger data
        import random
        random.seed(int(bus_id) if bus_id.isdigit() else hash(bus_id))
        passengers = random.randint(15, 50)
        capacity = 60
        speed = random.randint(45, 70)
        eta = random.randint(5, 30)
        
        buses.append({
            "id": bus_id,
            "active": state.get("active", False),
            "current_city": current_city,
            "next_city": next_city,
            "distance_to_next": distance_to_next,
            "progress": progress,
            "passengers": passengers,
            "capacity": capacity,
            "speed": speed,
            "eta": eta,
            "status": state.get("status", "Unknown"),
            "route": state.get("route", [])
        })
    
    return {"buses": buses}

@app.get("/api/stations")
async def get_stations():
    """Get all station states with enhanced passenger information"""
    if not state_manager:
        return {"stations": []}
    
    station_states = state_manager.get_station_states()
    stations = []
    
    # Load city names
    try:
        import pandas as pd
        df = pd.read_csv('data/cities.csv')
        cities = df['Origin'].unique().tolist()
    except:
        cities = ["Mumbai", "Pune", "Ahmedabad", "Hyderabad", "Bengaluru", "Jaipur", "Delhi", "Chennai"]
    
    for station_id, state in station_states.items():
        waiting_passengers = state.get("waiting_passengers", {})
        
        # Enhanced passenger destination breakdown
        if isinstance(waiting_passengers, dict):
            # Convert numeric destinations to city names
            passenger_destinations = {}
            total_waiting = 0
            
            for dest, count in waiting_passengers.items():
                try:
                    if str(dest).isdigit() and int(dest) < len(cities):
                        city_name = cities[int(dest)]
                    else:
                        city_name = str(dest)
                    passenger_destinations[city_name] = int(count)
                    total_waiting += int(count)
                except:
                    passenger_destinations[str(dest)] = int(count) if isinstance(count, (int, float)) else 0
                    total_waiting += int(count) if isinstance(count, (int, float)) else 0
        else:
            passenger_destinations = {}
            total_waiting = 0
        
        # Get station name
        try:
            station_name = cities[int(station_id)] if str(station_id).isdigit() and int(station_id) < len(cities) else f"Station {station_id}"
        except:
            station_name = f"Station {station_id}"
        
        # Generate additional realistic data
        import random
        random.seed(int(station_id) if str(station_id).isdigit() else hash(str(station_id)))
        buses_at_station = random.randint(0, 2)
        last_arrival = random.randint(1, 10) if random.random() > 0.3 else None
        
        stations.append({
            "id": station_id,
            "name": station_name,
            "waiting_passengers": passenger_destinations,
            "total_waiting": total_waiting,
            "buses_at_station": buses_at_station,
            "last_arrival": last_arrival,
            "next_arrivals": state.get("next_arrivals", {})
        })
    
    return {"stations": stations}

@app.get("/api/incidents")
async def get_incidents():
    """Get all active incidents"""
    if not system_time:
        return {"incidents": []}
    
    incidents = []
    current_time = system_time.get_current_time()
    
    for (city_a, city_b), incident in system_time.incidents.items():
        time_remaining = incident['duration'] - (current_time - incident['start_time'])
        if time_remaining > 0:
            incidents.append({
                "city_a": city_a,
                "city_b": city_b,
                "type": incident['type'],
                "start_time": incident['start_time'],
                "time_remaining": time_remaining
            })
    
    return {"incidents": incidents}

@app.post("/api/incidents")
async def create_incident(incident: IncidentCreate):
    """Create a new incident"""
    try:
        # Use simulation manager for demo incidents
        success = sim_manager.create_incident(incident.city_a, incident.city_b, incident.incident_type)
        
        # Also try to add to system_time if available
        if system_time:
            system_time.add_incident(incident.city_a, incident.city_b, incident.incident_type)
        
        if success:
            return {"message": f"Incident created between cities {incident.city_a} and {incident.city_b}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create incident")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating incident: {str(e)}")

@app.post("/api/simulation/control")
async def control_simulation(control: SimulationControl):
    """Control simulation (start/pause/stop/speed)"""
    global simulation_control
    
    if control.action == "start":
        simulation_control["running"] = True
        sim_manager.start_simulation(control.speed or 1.0)
    elif control.action == "pause":
        simulation_control["running"] = False
        sim_manager.pause_simulation()
    elif control.action == "stop":
        simulation_control["running"] = False
        sim_manager.stop_simulation()
    
    if control.speed:
        simulation_control["speed"] = control.speed
        sim_manager.set_speed(control.speed)
        if system_time:
            system_time.time_multiplier = 60 * control.speed
    
    return {"message": f"Simulation {control.action}", "status": simulation_control}

@app.get("/api/metrics")
async def get_metrics():
    """Get system KPIs and metrics"""
    if not db_manager:
        return {"metrics": {}}
    
    # Get recent data from database
    bus_states = db_manager.get_recent_bus_states(100)
    station_states = db_manager.get_recent_station_states(100)
    
    # Calculate basic metrics
    total_buses = len(set([state[1] for state in bus_states])) if bus_states else 0
    active_buses = len([state for state in bus_states if state[2]]) if bus_states else 0
    
    # Calculate average waiting passengers
    total_waiting = 0
    station_count = 0
    if station_states:
        for state in station_states[-20:]:  # Last 20 station updates
            try:
                waiting_data = json.loads(state[2]) if state[2] else {}
                if isinstance(waiting_data, dict):
                    total_waiting += sum(waiting_data.values())
                    station_count += 1
            except:
                continue
    
    avg_waiting = total_waiting / station_count if station_count > 0 else 0
    
    metrics = {
        "total_buses": total_buses,
        "active_buses": active_buses,
        "utilization_rate": (active_buses / total_buses * 100) if total_buses > 0 else 0,
        "average_waiting_passengers": avg_waiting,
        "total_incidents": len(system_time.incidents) if system_time else 0,
        "system_uptime": system_time.get_current_time() if system_time else 0
    }
    
    return {"metrics": metrics}

@app.get("/api/cities")
async def get_cities():
    """Get city information and connections"""
    try:
        # Read cities data
        df = pd.read_csv('data/cities.csv')
        cities = df['Origin'].unique().tolist()
        
        connections = []
        for _, row in df.iterrows():
            connections.append({
                "origin": row['Origin'],
                "destination": row['Destination'],
                "distance": row['Distance']
            })
        
        return {
            "cities": cities,
            "connections": connections
        }
    except Exception as e:
        return {"cities": [], "connections": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
