"""
Bus Lines Manager
Handles creation and management of bus lines and stations
"""
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import json
with open('config.json') as f:
    config = json.load(f)

@dataclass
class Station:
    id: int
    name: str
    lat: float
    lon: float
    line_id: int
    is_terminal: bool = False  # True if it's at the beginning or end of a line
    
@dataclass
class BusLine:
    id: int
    name: str
    stations: List[Station]
    reserve_buses: int = config['buses']['reserve_buses_per_line']  # Number of reserve buses at terminals
    
    


class BusLinesManager:
    def __init__(self):
        self.lines: Dict[int, BusLine] = {}
        self.stations: Dict[int, Station] = {}
        self.next_line_id = 0
        self.next_station_id = 0
        self.logger = logging.getLogger('bus_lines_manager')
        
    def create_line(self, name: str, station_coordinates: List[Tuple[float, float]]) -> int:
        """
        Create a new bus line with stations
        Args:
            name: Name of the bus line
            station_coordinates: List of (lat, lon) coordinates for stations
        Returns:
            line_id: ID of the created line
        """
        if len(station_coordinates) < 2:
            raise ValueError("A line must have at least 2 stations")
        
        line_id = self.next_line_id
        self.next_line_id += 1
        
        # Create stations for this line
        stations = []
        for i, (lat, lon) in enumerate(station_coordinates):
            station_id = self.next_station_id
            self.next_station_id += 1
            
            # Check if this is a terminal station (first or last)
            is_terminal = (i == 0) or (i == len(station_coordinates) - 1)
            
            station = Station(
                id=station_id,
                name=f"{name}_Station_{i}",
                lat=lat,
                lon=lon,
                line_id=line_id,
                is_terminal=is_terminal )
            
            
            stations.append(station)
            self.stations[station_id] = station
        
        # Create the bus line
        bus_line = BusLine(
            id=line_id,
            name=name,
            stations=stations
        )
        
        self.lines[line_id] = bus_line
        
        self.logger.info(f"Created bus line {name} (ID: {line_id}) with {len(stations)} stations")
        return line_id
    


    def delete_line(self, line_id: int) -> bool:
        """
        Delete a bus line and all its stations
        Args: line_id: ID of the line to delete
        Returns:True if successfully deleted, False if line not found
        """
        if line_id not in self.lines:
            self.logger.warning(f"Line {line_id} not found for deletion")
            return False
        
        line = self.lines[line_id]
        
        # Remove all stations belonging to this line
        stations_to_remove = [station.id for station in line.stations]
        for station_id in stations_to_remove:
            if station_id in self.stations:
                del self.stations[station_id]
        
        # Remove the line
        del self.lines[line_id]
        
        self.logger.info(f"Deleted line '{line.name}' (ID: {line_id}) and {len(stations_to_remove)} stations")
        return True



    
    def get_line(self, line_id: int) -> Optional[BusLine]:
        """Get a bus line by ID"""
        return self.lines.get(line_id)
    
    def get_station(self, station_id: int) -> Optional[Station]:
        """Get a station by ID"""
        return self.stations.get(station_id)
    
    def get_all_lines(self) -> Dict[int, BusLine]:
        """Get all bus lines"""
        return self.lines.copy()
    
    def get_all_stations(self) -> Dict[int, Station]:
        """Get all stations"""
        return self.stations.copy()
    
    def get_terminal_stations(self) -> List[Station]:
        """Get all terminal stations (start/end of lines)"""
        return [station for station in self.stations.values() if station.is_terminal]
    
    def get_line_stations(self, line_id: int) -> List[Station]:
        """Get all stations for a specific line"""
        line = self.lines.get(line_id)
        return line.stations if line else []
    
    def get_stations_by_line(self) -> Dict[int, List[Station]]:
        """Get stations grouped by line"""
        result = {}
        for line_id, line in self.lines.items():
            result[line_id] = line.stations
        return result
    
    def find_nearest_station(self, lat: float, lon: float) -> Optional[Station]:
        """Find the nearest station to given coordinates"""
        if not self.stations:
            return None
        
        min_distance = float('inf')
        nearest_station = None
        
        for station in self.stations.values():
            # Simple Euclidean distance (in practice, would use haversine)
            distance = np.sqrt((station.lat - lat)**2 + (station.lon - lon)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        
        return nearest_station
    
    def get_next_station_on_line(self, current_station_id: int, direction: str = "forward") -> Optional[Station]:
        """
        Get the next station on the same line
        Args:
            current_station_id: Current station ID
            direction: "forward" or "backward"
        """
        current_station = self.stations.get(current_station_id)
        if not current_station:
            return None
        
        line = self.lines.get(current_station.line_id)
        if not line:
            return None
        
        # Find current station index in the line
        current_index = None
        for i, station in enumerate(line.stations):
            if station.id == current_station_id:
                current_index = i
                break
        
        if current_index is None:
            return None
        
        # Get next station based on direction
        if direction == "forward":
            next_index = current_index + 1
            if next_index < len(line.stations):
                return line.stations[next_index]
        elif direction == "backward":
            next_index = current_index - 1
            if next_index >= 0:
                return line.stations[next_index]
        
        return None
    
    def remove_line(self, line_id: int) -> bool:
        """Remove a bus line and its stations"""
        if line_id not in self.lines:
            return False
        
        line = self.lines[line_id]
        
        # Remove all stations from this line
        for station in line.stations:
            if station.id in self.stations:
                del self.stations[station.id]
        
        # Remove the line
        del self.lines[line_id]
        
        self.logger.info(f"Removed bus line {line.name} (ID: {line_id})")
        return True
    
    def save_to_file(self, filename: str):
        """Save lines configuration to JSON file"""
        data = {
            'lines': {},
            'stations': {},
            'next_line_id': self.next_line_id,
            'next_station_id': self.next_station_id
        }
        
        # Convert lines to dict
        for line_id, line in self.lines.items():
            data['lines'][str(line_id)] = {
                'id': line.id,
                'name': line.name,
                'stations': [station.id for station in line.stations],
                'reserve_buses': line.reserve_buses
            }
        
        # Convert stations to dict
        for station_id, station in self.stations.items():
            data['stations'][str(station_id)] = {
                'id': station.id,
                'name': station.name,
                'lat': station.lat,
                'lon': station.lon,
                'line_id': station.line_id,
                'is_terminal': station.is_terminal
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load lines configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.next_line_id = data.get('next_line_id', 0)
            self.next_station_id = data.get('next_station_id', 0)
            
            # Load stations first
            self.stations = {}
            for station_data in data.get('stations', {}).values():
                station = Station(
                    id=station_data['id'],
                    name=station_data['name'],
                    lat=station_data['lat'],
                    lon=station_data['lon'],
                    line_id=station_data['line_id'],
                    is_terminal=station_data['is_terminal']
                )
                self.stations[station.id] = station
            
            # Load lines
            self.lines = {}
            for line_data in data.get('lines', {}).values():
                stations = [self.stations[sid] for sid in line_data['stations'] 
                           if sid in self.stations]
                
                line = BusLine(
                    id=line_data['id'],
                    name=line_data['name'],
                    stations=stations,
                    reserve_buses=line_data.get('reserve_buses', 5)
                )
                self.lines[line.id] = line
            
            self.logger.info(f"Loaded {len(self.lines)} lines and {len(self.stations)} stations")
            
        except FileNotFoundError:
            self.logger.info("No existing lines configuration found")
        except Exception as e:
            self.logger.error(f"Error loading lines configuration: {e}")
    
    def get_line_demand(self, line_id: int, passenger_requests: List[Dict]) -> int:
        """
        Calculate demand for a specific line based on passenger requests
        Counts passengers wanting to go to cities/stations in this line
        Args:
            line_id: Line ID
            passenger_requests: List of passenger requests with destination info
        Returns:
            Number of passengers wanting to travel to stations on this line
        """
        if line_id not in self.lines:
            return 0
        
        line = self.lines[line_id]
        line_station_ids = {station.id for station in line.stations}
        
        demand = 0
        for request in passenger_requests:
            destination_station_id = request.get('destination_station')
            if destination_station_id in line_station_ids:
                demand += request.get('passenger_count', 1)
        
        return demand
    
    def should_release_reserve_bus(self, line_id: int, passenger_requests: List[Dict], 
                                 threshold: int = 70) -> bool:
        """
        Determine if a reserve bus should be released for a line
        Args:
            line_id: Line ID
            passenger_requests: Current passenger requests
            threshold: Passenger count threshold for releasing bus
        Returns:
            True if reserve bus should be released
        """
        demand = self.get_line_demand(line_id, passenger_requests)
        return demand >= threshold
    
