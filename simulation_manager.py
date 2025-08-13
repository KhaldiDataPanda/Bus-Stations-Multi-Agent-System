"""
Enhanced Simulation Manager for Real-time Dashboard
Provides realistic simulation data when main simulation is not running
"""
import asyncio
import random
import time
import json
from datetime import datetime
from utils import SystemTime, SystemState
from db_manager import DatabaseManager

class SimulationManager:
    def __init__(self):
        self.system_time = SystemTime()
        self.state_manager = SystemState()
        self.db_manager = DatabaseManager()
        self.running = False
        self.speed = 1.0
        
        # Demo cities (from CSV data)
        self.cities = ["Mumbai", "Pune", "Ahmedabad", "Hyderabad", "Bengaluru", "Jaipur", "Delhi", "Chennai"]
        self.n_cities = len(self.cities)
        
        # Initialize demo buses and stations
        self.demo_buses = {}
        self.demo_stations = {}
        self.incidents = {}
        
        self._initialize_demo_data()
    
    def _initialize_demo_data(self):
        """Initialize realistic demo data"""
        
        # Create demo buses
        for i in range(8):
            bus_id = f"BUS-{i+1}"
            current_city = random.randint(0, self.n_cities - 1)
            next_city = random.randint(0, self.n_cities - 1)
            if next_city == current_city:
                next_city = (current_city + 1) % self.n_cities
            
            self.demo_buses[bus_id] = {
                'bus_id': bus_id,
                'active': random.choice([True, True, True, False]),  # 75% active
                'current_city': current_city,
                'next_city': next_city,
                'progress': random.uniform(0.1, 0.9),
                'passengers': random.randint(15, 50),
                'capacity': 60,
                'speed': random.randint(45, 70),
                'eta': random.randint(5, 30),
                'status': random.choice(["Moving", "Boarding", "At Station"]),
                'route': [current_city, next_city],
                'last_update': time.time()
            }
        
        # Create demo stations
        for i in range(self.n_cities):
            waiting_passengers = {}
            total_waiting = 0
            
            # Generate passengers for different destinations
            for dest in range(self.n_cities):
                if dest != i:  # Can't wait for same city
                    count = random.randint(0, 15)
                    if count > 0:
                        waiting_passengers[str(dest)] = count
                        total_waiting += count
            
            self.demo_stations[str(i)] = {
                'station_id': str(i),
                'waiting_passengers': waiting_passengers,
                'total_waiting': total_waiting,
                'buses_at_station': random.randint(0, 2),
                'last_arrival': random.randint(1, 10) if random.random() > 0.3 else None,
                'next_arrivals': {},
                'timestamp': time.time()
            }
    
    def start_simulation(self, speed=1.0):
        """Start the demo simulation"""
        self.running = True
        self.speed = speed
        print(f"Demo simulation started at {speed}x speed")
    
    def pause_simulation(self):
        """Pause the simulation"""
        self.running = False
        print("Demo simulation paused")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        print("Demo simulation stopped")
    
    def set_speed(self, speed):
        """Set simulation speed"""
        self.speed = speed
        print(f"Demo simulation speed set to {speed}x")
    
    def update_simulation(self):
        """Update simulation state (called periodically)"""
        if not self.running:
            return
        
        current_time = time.time()
        
        # Update buses
        for bus_id, bus in self.demo_buses.items():
            if not bus['active']:
                continue
            
            # Simulate movement
            time_since_update = current_time - bus['last_update']
            progress_increment = (self.speed * time_since_update) / 60  # Move based on time
            
            bus['progress'] += progress_increment * 0.1  # Slow movement for visibility
            
            # Check if bus reached destination
            if bus['progress'] >= 1.0:
                # Bus reached destination
                old_current = bus['current_city']
                bus['current_city'] = bus['next_city']
                
                # Pick new destination
                available_cities = [i for i in range(self.n_cities) if i != bus['current_city']]
                bus['next_city'] = random.choice(available_cities)
                bus['progress'] = 0.0
                
                # Update passenger counts
                passengers_leaving = random.randint(5, bus['passengers'])
                bus['passengers'] -= passengers_leaving
                
                # Pick up new passengers
                station = self.demo_stations[str(bus['current_city'])]
                new_passengers = min(random.randint(0, 10), bus['capacity'] - bus['passengers'])
                bus['passengers'] += new_passengers
                
                # Update station waiting count
                station['total_waiting'] = max(0, station['total_waiting'] - new_passengers)
                
                # Update status and ETA
                bus['status'] = "At Station"
                bus['eta'] = random.randint(5, 30)
            
            else:
                bus['status'] = "Moving"
                # Update ETA based on progress
                remaining_progress = 1.0 - bus['progress']
                bus['eta'] = int(remaining_progress * 30)  # Rough ETA calculation
            
            bus['last_update'] = current_time
            
            # Update state manager
            self.state_manager.update_bus_state(bus_id, {
                'active': bus['active'],
                'current_city': bus['current_city'],
                'next_city': bus['next_city'],
                'distance_to_next': (1 - bus['progress']) * 100,  # Simulated distance
                'status': bus['status']
            })
        
        # Update stations - add new passengers randomly
        for station_id, station in self.demo_stations.items():
            # Random new passengers
            if random.random() < 0.3:  # 30% chance of new passengers
                new_passengers = random.randint(1, 5)
                dest_city = random.randint(0, self.n_cities - 1)
                if dest_city != int(station_id):
                    if str(dest_city) not in station['waiting_passengers']:
                        station['waiting_passengers'][str(dest_city)] = 0
                    station['waiting_passengers'][str(dest_city)] += new_passengers
                    station['total_waiting'] += new_passengers
            
            # Update state manager
            self.state_manager.update_station_state(station_id, {
                'waiting_passengers': station['waiting_passengers'],
                'next_arrivals': station['next_arrivals'],
                'timestamp': current_time
            })
    
    def create_incident(self, city_a, city_b, incident_type):
        """Create a traffic incident"""
        incident_key = f"{city_a}-{city_b}"
        
        duration_map = {
            'light_traffic': 6,
            'heavy_traffic': 12,
            'closed_road': 24
        }
        
        self.incidents[incident_key] = {
            'type': incident_type,
            'city_a': city_a,
            'city_b': city_b,
            'start_time': time.time(),
            'duration': duration_map.get(incident_type, 6) * 3600,  # Convert to seconds
            'active': True
        }
        
        # Affect buses on this route
        for bus_id, bus in self.demo_buses.items():
            if ((bus['current_city'] == city_a and bus['next_city'] == city_b) or 
                (bus['current_city'] == city_b and bus['next_city'] == city_a)):
                
                # Slow down or reroute bus
                if incident_type == 'closed_road':
                    # Find alternative route
                    available_cities = [i for i in range(self.n_cities) 
                                      if i != bus['current_city'] and i != city_b]
                    if available_cities:
                        bus['next_city'] = random.choice(available_cities)
                        bus['eta'] *= 1.5  # Longer route
                elif incident_type in ['light_traffic', 'heavy_traffic']:
                    # Increase ETA
                    multiplier = 1.3 if incident_type == 'light_traffic' else 1.7
                    bus['eta'] = int(bus['eta'] * multiplier)
        
        print(f"Incident created: {incident_type} between {self.cities[city_a]} and {self.cities[city_b]}")
        return True
    
    def get_active_incidents(self):
        """Get list of active incidents"""
        current_time = time.time()
        active_incidents = []
        
        for incident_key, incident in self.incidents.items():
            elapsed = current_time - incident['start_time']
            if elapsed < incident['duration']:
                time_remaining = (incident['duration'] - elapsed) / 3600  # Convert to hours
                active_incidents.append({
                    'city_a': incident['city_a'],
                    'city_b': incident['city_b'],
                    'type': incident['type'],
                    'start_time': incident['start_time'],
                    'time_remaining': time_remaining
                })
            else:
                incident['active'] = False
        
        # Clean up expired incidents
        self.incidents = {k: v for k, v in self.incidents.items() if v['active']}
        
        return active_incidents
    
    def get_system_status(self):
        """Get current system status"""
        return {
            'current_time': time.time(),
            'simulation_running': self.running,
            'simulation_speed': self.speed,
            'active_incidents': len(self.get_active_incidents()),
            'active_buses': len([b for b in self.demo_buses.values() if b['active']]),
            'total_buses': len(self.demo_buses)
        }

# Global simulation manager instance
sim_manager = SimulationManager()

async def simulation_loop():
    """Main simulation update loop"""
    while True:
        sim_manager.update_simulation()
        await asyncio.sleep(1)  # Update every second

# Start the simulation loop
asyncio.create_task(simulation_loop())
