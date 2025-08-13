import heapq
from datetime import timedelta
import asyncio



def log_message(direction, sender, receiver, message,system_time ,expected_response=None,DEBUG_MESSAGING=True):
    if DEBUG_MESSAGING:
        base_msg = f"[{system_time.get_current_time():.2f}h] {direction} | From: {sender} | To: {receiver} | Message: {message}"
        if expected_response:
            base_msg += f" | Expecting: {expected_response}"
        print(base_msg)


def a_star(matrix, start, goal):
    """
    A* pathfinding algorithm to determine optimal route between cities
    Returns: path (list of city indices) and total distance
    """
    visited = set()
    came_from = {}
    g_score = {i: float('inf') for i in range(matrix.shape[0])}
    g_score[start] = 0
    open_set = [(0, start)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            total_dist = 0
            for i in range(len(path) - 1):
                total_dist += matrix[path[i]][path[i + 1]]
            return path, total_dist

        for neighbor in range(matrix.shape[0]):
            if neighbor in visited or matrix[current][neighbor] == 0:
                continue
            tentative_g_score = g_score[current] + matrix[current][neighbor]
            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                came_from[neighbor] = current
                heapq.heappush(open_set, (tentative_g_score, neighbor))

        visited.add(current)

    return None, None



class SystemTime:
    def __init__(self):
        self._current_time = 0  # Time in system hours
        self.time_multiplier = 60  # 1 real minute = 1 system hour
        self.incidents = {}  # {(city_a, city_b): {'type': type, 'start_time': time, 'duration': duration}}

    async def update_time(self):
        """Updates system time every real second"""
        while True:
            await asyncio.sleep(1)  # Sleep for 1 real second
            self._current_time += self.time_multiplier/60  # Add 1 hour in system time

    def get_current_time(self):
        return self._current_time

    def add_incident(self, city_a, city_b, incident_type):
        duration = {
            'light_traffic': 6,    # 6 system hours
            'heavy_traffic': 12,   # 12 system hours
            'closed_road': 24      # 24 system hours
        }
        self.incidents[(city_a, city_b)] = {
            'type': incident_type,
            'start_time': self._current_time,
            'duration': duration[incident_type]
        }
        print(f"System Time {self._current_time:.2f}h: New incident {incident_type} between cities {city_a} and {city_b}")

    def check_incident(self, city_a, city_b):
        """Returns active incident between cities if exists"""
        if (city_a, city_b) in self.incidents:
            incident = self.incidents[(city_a, city_b)]
            if self._current_time - incident['start_time'] < incident['duration']:
                return incident
            else:
                del self.incidents[(city_a, city_b)]
        return None
    
    
    
    
class SystemState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemState, cls).__new__(cls)
            cls._instance.bus_states = {}
            cls._instance.station_states = {}
        return cls._instance

    def update_bus_state(self, bus_id, state):
        if not isinstance(bus_id, (int, str)):
            print(f"[WARNING] Invalid bus_id type: {type(bus_id)}")
            return
        
        processed_state = {
            'active': bool(state.get('active', False)),
            'current_city': state.get('current_city'),
            'next_city': state.get('next_city'),
            'distance_to_next': float(state.get('distance_to_next', 0)),
            'status': state.get('status', 'Unknown'),

        }
        
        self.bus_states[str(bus_id)] = processed_state
        print(f"[DEBUG] Updated bus {bus_id} state in manager: {processed_state}")

    def update_station_state(self, station_id, state):
        if not isinstance(station_id, (int, str)):
            print(f"[WARNING] Invalid station_id type: {type(station_id)}")
            return
            
        processed_state = {
            'station_id': str(station_id),  # Add station_id to the state
            'waiting_passengers': {
                str(k): int(v) for k, v in state.get('waiting_passengers', {}).items()
            },
            'next_arrivals': {
                str(k): float(v) if isinstance(v, (int, float)) else v 
                for k, v in state.get('next_arrivals', {}).items()
            }
        }
        
        self.station_states[str(station_id)] = processed_state
        print(f"[DEBUG] Updated station {station_id} state in manager: {processed_state}")

    def get_bus_states(self):
        return self.bus_states

    def get_station_states(self):
        return self.station_states