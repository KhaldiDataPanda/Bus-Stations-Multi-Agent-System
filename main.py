"""
Traffic Routing System with A* Pathfinding
Main simulation file with A* routing for bus navigation
"""
from collections import defaultdict
import pandas as pd
import random
import json
import asyncio
import logging
import time
import numpy as np
import networkx as nx
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import heapq
import socket
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Get hostname for dynamic JID creation
HOSTNAME = socket.gethostname()
CONTROL_JID = f"control@{HOSTNAME}"

def get_bus_jid(bus_id: int) -> str:
    """Get the JID for a bus agent."""
    return f"bus_{bus_id}@{HOSTNAME}"

def get_station_jid(station_id: int) -> str:
    """Get the JID for a station agent."""
    return f"station_{station_id}@{HOSTNAME}"


# Import custom modules
from utils import log_message, a_star, SystemTime, SystemState
from db_manager import DatabaseManager
from bus_lines_manager import BusLinesManager, Station, BusLine
from plotting_utils import initialize_plotter, get_plotter

# Load configuration
import json
with open('config.json', 'r') as f:
    CONFIG = json.load(f)



Path(CONFIG['plotting']['plot_folder']).mkdir(parents=True, exist_ok=True)

# Initialize managers
db_manager = DatabaseManager()
bus_lines_manager = BusLinesManager()
bus_lines_manager.load_from_file('data/bus_lines.json')

# Load graph data from Blida map
from graph_loader import GraphLoader
from full_graph_manager import FullGraphManager

graph_loader = GraphLoader('data/Blida_map.graphml')
# Initialize full_graph_manager only if not already set (allows external initialization)
full_graph_manager = None
if full_graph_manager is None:
    full_graph_manager = FullGraphManager('data/Blida_map.graphml')

# Initialize plotting (after graph manager is created)
plotter = initialize_plotter('config.json', full_graph_manager)

# Global simulation state
simulation_running = False
simulation_lock = threading.Lock()

# Message and logging configuration
MESSAGE_FORMATS = {
    'system': "[SYSTEM TIME {:.2f}h] {}",
    'bus': "[BUS-{} | {:.2f}h] {}",
    'station': "[STATION-{} | {:.2f}h] {}",
    'control': "[CONTROL | {:.2f}h] {}",
    'incident': "[INCIDENT | {:.2f}h] Bus {} - {} between stations {} and {}"
}

LOGGING_FORMAT = {
    'bus': '[BUS-%(bus_id)s ] - %(levelname)s - %(message)s',
    'station': '[STATION-%(station_id)s ] - %(levelname)s - %(message)s',
    'control': '[CONTROL ] - %(levelname)s - %(message)s',
    'astar': '[A*-ROUTING ] - %(levelname)s - %(message)s'
}

DEBUG_MESSAGING = False  # Set to True for detailed message logging

# Configure logging with simplified formatters
bus_logger = logging.getLogger('bus')
station_logger = logging.getLogger('station')
control_logger = logging.getLogger('control')
astar_logger = logging.getLogger('astar')

# Note: Logging handlers are now configured in utils.py
# Individual logger levels can still be adjusted if needed
for logger in [bus_logger, station_logger, control_logger, astar_logger]:
    logger.setLevel(logging.INFO)

# System managers
system_time = SystemTime(time_multiplier=CONFIG['simulation']['time_speed_multiplier'])
state_manager = SystemState()

# Global simulation state
simulation_state = {
    'running': False,
    'passenger_requests': [],
    'active_incidents': {},
    'bus_performance_metrics': defaultdict(list),
    'buses_by_line': defaultdict(list),  # Track buses per line
    'schedule_tracker': {},  # Track bus scheduling
    'reserve_buses': {}  # Track reserve buses per line
}


############################################################################################################
############################################################################################################
#-----------------------------         ENHANCED BUS AGENT          --------------------------------------
############################################################################################################
############################################################################################################

class BusAgent(Agent):
    """Bus Agent with A* routing for optimal pathfinding"""
    
    async def setup(self):
        self.bus_id = int(str(self.jid).split('_')[1].split('@')[0])
        self.assigned_line_id = None
        self.current_station_id = None
        self.target_station_id = None
        self.passenger_count = 0
        self.trip_start_time = 0
        self.direction = "forward"  # "forward" or "backward"
        self.timeout_threshold = 0
        self.current_path = []
        self.is_initialized = False
        
        # Enhanced path tracking for full graph with A* routing
        self.current_graph_node = None  # Current position in the full graph
        self.current_path_nodes = []  # List of graph node IDs
        self.current_path_edges = []  # List of edge IDs
        self.station_reached_count = 0  # Track stations reached
        self.line_completion_count = 0  # Track full line completions
        self.current_trip_start_station = None  # Track start of current trip
        self.steps_taken = 0  # Track steps in current trip
        self.route_steps = 0  # Track steps in current route segment (between stations)
        
        # A* path tracking
        self.total_routes_completed = 0  # Track total routes completed (station to station)
        self.current_trip_visited_nodes = {}  # Track nodes visited in current trip only (station A to B)
        self.astar_path = []  # Complete A* path nodes
        self.astar_path_index = 0  # Current position in A* path
        self.astar_path_calculated = False  # Whether path has been calculated for current route
        
        if DEBUG_MESSAGING:
            print(MESSAGE_FORMATS['bus'].format(self.bus_id, system_time.get_current_time(), 
                                               "Bus Agent starting setup..."))

        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(self.InitBehavior())
        self.add_behaviour(self.MessageHandler())

    class InitBehavior(CyclicBehaviour):
        async def run(self):
            if not self.agent.is_initialized:
                msg = Message(to=CONTROL_JID)
                msg.set_metadata("performative", "subscribe")
                msg.body = f"REGISTER:{self.agent.bus_id}"

                log_message("SEND", f"Bus_{self.agent.bus_id}", "Control", msg.body, system_time)
                await self.send(msg)
                
                msg = await self.receive(timeout=5)
                if msg:
                    log_message("RECEIVE", "Control", f"Bus_{self.agent.bus_id}", msg.body, system_time)
                    if "INIT_CONFIRM" in msg.body:
                        self.agent.is_initialized = True
                        bus_logger.info(f"Bus {self.agent.bus_id + 1} received initialization confirmation")
                        
                        behaviour = self.agent.AStarBehaviour(self.agent.bus_id)
                        self.agent.add_behaviour(behaviour)
                        bus_logger.info(f"Bus {self.agent.bus_id + 1} initialized and ready for A* routing")
                        self.kill() # Stop this behavior after initialization
                else:
                    await asyncio.sleep(1)

    class AStarBehaviour(CyclicBehaviour):
        def __init__(self, bus_id):
            super().__init__()
            self.bus_id = bus_id
            self.waiting_for_assignment = True
            self.is_active = False

        async def run(self):
            await self.update_bus_state()
            
            if self.waiting_for_assignment:
                await self.wait_for_line_assignment()
                return
            
            if not self.is_active:
                await asyncio.sleep(1)
                return
            
            # Check if reached target station
            if self.agent.current_graph_node and self.agent.target_station_id is not None:
                if full_graph_manager.is_node_near_station(self.agent.current_graph_node, self.agent.target_station_id):
                    await self.handle_station_arrival()
                    return
            
            # Only make decisions if we have a different target than current
            if self.agent.target_station_id is not None:
                await self.make_astar_decision()
            else:
                # No valid target, request new target
                await self.request_next_target()
            
            await asyncio.sleep(1)

        async def update_bus_state(self):
            """Update bus state in database and system state"""
            current_station = bus_lines_manager.get_station(self.agent.current_station_id) if self.agent.current_station_id is not None else None
            target_station = bus_lines_manager.get_station(self.agent.target_station_id) if self.agent.target_station_id is not None else None
            
            # Get coordinates from current graph node if available, otherwise use station coordinates
            if self.agent.current_graph_node:
                current_coords = full_graph_manager.get_node_coordinates(self.agent.current_graph_node)
                current_coords = {'lat': current_coords[0], 'lon': current_coords[1]}
                
                # If bus is moving on an edge (has next node in path), interpolate position
                if hasattr(self.agent, 'astar_path') and self.agent.astar_path and self.agent.astar_path_index < len(self.agent.astar_path) - 1:
                    current_node = self.agent.astar_path[self.agent.astar_path_index]
                    next_node = self.agent.astar_path[self.agent.astar_path_index + 1]
                    
                    # Calculate progress along the edge (0.5 = middle of edge, can be enhanced with actual distance tracking)
                    progress_ratio = 0.5  # Default to middle of edge
                    
                    # Get interpolated position on the edge
                    bus_lat, bus_lon = full_graph_manager.get_bus_position_on_edge(current_node, next_node, progress_ratio)
                    current_coords = {'lat': bus_lat, 'lon': bus_lon}
                    
            elif current_station:
                current_coords = {'lat': current_station.lat, 'lon': current_station.lon}
            else:
                current_coords = {'lat': 0, 'lon': 0}
                
            target_coords = {'lat': target_station.lat, 'lon': target_station.lon} if target_station else {'lat': 0, 'lon': 0}
            
            # Calculate distance to target
            distance_to_next = 0.0
            if current_coords['lat'] != 0 and target_coords['lat'] != 0:
                lat1, lon1 = current_coords['lat'], current_coords['lon']
                lat2, lon2 = target_coords['lat'], target_coords['lon']
                distance_to_next = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111.32  # Approximate km per degree
            
            # Convert A* path to coordinate path for visualization
            path_coordinates = []
            if hasattr(self.agent, 'astar_path') and self.agent.astar_path:
                path_coordinates = full_graph_manager.get_path_coordinates(self.agent.astar_path)
            
            state = {
                'bus_id': self.bus_id,
                'active': self.is_active,
                'current_station_id': self.agent.current_station_id,
                'target_station_id': self.agent.target_station_id,
                'assigned_line_id': self.agent.assigned_line_id,
                'passenger_count': self.agent.passenger_count,
                'lat': current_coords['lat'],
                'lon': current_coords['lon'],
                'destination_lat': target_coords['lat'],
                'destination_lon': target_coords['lon'],
                'status': "Active" if self.is_active else "Waiting",
                'timestamp': time.time(),
                'direction': self.agent.direction,
                'path': path_coordinates,  # Use coordinate path instead of JSON string
                'current_location': current_station.name if current_station else "In Transit",
                'destination': target_station.name if target_station else "Unknown",
                'current_city': current_station.name if current_station else None,
                'next_city': target_station.name if target_station else None,
                'distance_to_next': distance_to_next,
                'current_graph_node': self.agent.current_graph_node,
                'steps_taken': self.agent.steps_taken
            }
            
            await db_manager.save_bus_state(self.bus_id, state)
            state_manager.update_bus_state(self.bus_id, state)

        async def wait_for_line_assignment(self):
            """Wait for line assignment from control agent"""
            await asyncio.sleep(1)

        async def make_astar_decision(self):
            """Make A*-based routing decision on the full graph"""
            if not self.agent.current_graph_node:
                # Initialize current graph node from current station
                if self.agent.current_station_id is not None:
                    if full_graph_manager is None:
                        astar_logger.error(f"Bus {self.bus_id + 1} ERROR: full_graph_manager is None!")
                        await self.request_next_target()
                        return
                    
                    station_nodes = full_graph_manager.get_station_nodes(self.agent.current_station_id)
                    if station_nodes:
                        self.agent.current_graph_node = station_nodes[0]
                        astar_logger.info(f"[INIT] Bus {self.bus_id + 1} initialized at graph node {self.agent.current_graph_node} (Station {self.agent.current_station_id + 1})")
                    else:
                        # Debug: Check if mapping exists
                        all_mappings = full_graph_manager.station_to_nodes
                        astar_logger.error(f"Bus {self.bus_id + 1} cannot find graph nodes for station {self.agent.current_station_id}. Available mappings: {list(all_mappings.keys())}")
                        await self.request_next_target()
                        return
                else:
                    astar_logger.error(f"Bus {self.bus_id + 1} has no current station")
                    await self.request_next_target()
                    return
            
            # Check if already at target station
            if full_graph_manager.is_node_near_station(self.agent.current_graph_node, self.agent.target_station_id):
                await self.handle_station_arrival()
                return
            
            # Get possible next nodes from current position
            possible_nodes = full_graph_manager.get_possible_next_nodes(self.agent.current_graph_node)
            if not possible_nodes:
                astar_logger.warning(f"Bus {self.bus_id + 1} has no possible moves from node {self.agent.current_graph_node}")
                await self.request_next_target()
                return
            
            # Get next node using A* pathfinding
            next_node = self.get_astar_next_node(possible_nodes)
            
            if not next_node:
                # Fallback to shortest distance decision if A* fails
                target_nodes = full_graph_manager.get_station_nodes(self.agent.target_station_id)
                if target_nodes:
                    # Choose the node that's closest to any target node
                    best_node = None
                    best_distance = float('inf')
                    
                    for node in possible_nodes:
                        for target_node in target_nodes:
                            weight = full_graph_manager.get_edge_weight(node, target_node)
                            if weight < best_distance:
                                best_distance = weight
                                best_node = node
                    
                    next_node = best_node or possible_nodes[0]
                    
                    edge_id = full_graph_manager.get_edge_id(self.agent.current_graph_node, next_node)
                    astar_logger.info(f"[FALLBACK-STEP] Bus {self.bus_id + 1} | Step {self.agent.steps_taken + 1} | "
                                     f"From Node: {self.agent.current_graph_node} | To Node: {next_node} | "
                                     f"Edge ID: {edge_id}")
                else:
                    next_node = possible_nodes[0]
            else:
                # Log successful A* decision
                edge_id = full_graph_manager.get_edge_id(self.agent.current_graph_node, next_node)
                current_station = full_graph_manager.get_nearest_station_to_node(self.agent.current_graph_node)
                target_station_name = bus_lines_manager.get_station(self.agent.target_station_id).name if self.agent.target_station_id else "Unknown"
                edge_distance = full_graph_manager.get_edge_weight(self.agent.current_graph_node, next_node)
                
                astar_logger.info(f"[A*-STEP] Bus {self.bus_id + 1} | Route Step {self.agent.route_steps + 1} | "
                                 f"From Node: {self.agent.current_graph_node} | To Node: {next_node} | "
                                 f"Edge ID: {edge_id} | Distance: {edge_distance:.1f}m | "
                                 f"Current Nearest Station: {current_station + 1 if current_station is not None else 'None'} | "
                                 f"Target Station: {target_station_name}")
            
            # Execute move if valid
            if next_node and next_node != self.agent.current_graph_node:
                await self.move_to_node(next_node)
                self.agent.steps_taken += 1
            else:
                astar_logger.warning(f"Bus {self.bus_id + 1} cannot move - invalid next node: {next_node}")
                await self.request_next_target()

        def get_astar_next_node(self, possible_nodes: List[str]) -> Optional[str]:
            """Get the next node from A* path for forced A* following"""
            if not self.agent.current_graph_node or self.agent.target_station_id is None:
                return None
            
            # Calculate A* path only once per route segment
            if not self.agent.astar_path_calculated:
                try:
                    # Get target station nodes
                    target_nodes = full_graph_manager.get_station_nodes(self.agent.target_station_id)
                    if not target_nodes:
                        return None
                    
                    # Find the best path from current node to any target station node
                    best_path = None
                    best_distance = float('inf')
                    
                    for target_node in target_nodes:
                        try:
                            # Use NetworkX A* directly from current node to target node
                            path = nx.astar_path(full_graph_manager.graph, 
                                               self.agent.current_graph_node, 
                                               target_node,
                                               heuristic=full_graph_manager._heuristic, 
                                               weight='weight')
                            distance = nx.astar_path_length(full_graph_manager.graph,
                                                          self.agent.current_graph_node,
                                                          target_node,
                                                          heuristic=full_graph_manager._heuristic,
                                                          weight='weight')
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_path = path
                                
                        except nx.NetworkXNoPath:
                            continue
                    
                    if best_path and len(best_path) >= 2:
                        self.agent.astar_path = best_path
                        self.agent.astar_path_index = 0  # Start at current position
                        self.agent.astar_path_calculated = True
                        
                        astar_logger.info(f"[A*-INIT] Bus {self.bus_id + 1} calculated A* path: {len(best_path)} nodes, {best_distance:.1f}m")
                    else:
                        astar_logger.warning(f"[A*-INIT] Bus {self.bus_id + 1} failed to calculate A* path")
                        return None
                        
                except Exception as e:
                    astar_logger.error(f"[A*-INIT] Bus {self.bus_id + 1} A* calculation error: {e}")
                    return None
            
            # If we have a valid A* path, follow it
            if self.agent.astar_path and self.agent.astar_path_index < len(self.agent.astar_path):
                # Find current position in A* path
                current_node = self.agent.current_graph_node
                
                # Check if current node matches the expected position in A* path
                if self.agent.astar_path_index < len(self.agent.astar_path) and current_node == self.agent.astar_path[self.agent.astar_path_index]:
                    # We're on track, get next node
                    if self.agent.astar_path_index + 1 < len(self.agent.astar_path):
                        next_astar_node = self.agent.astar_path[self.agent.astar_path_index + 1]
                        if next_astar_node in possible_nodes:
                            self.agent.astar_path_index += 1  # Advance index
                            return next_astar_node
                else:
                    # Current node doesn't match expected position, try to find it in the path
                    for i, path_node in enumerate(self.agent.astar_path):
                        if path_node == current_node:
                            self.agent.astar_path_index = i  # Update index to current position
                            if i + 1 < len(self.agent.astar_path):
                                next_astar_node = self.agent.astar_path[i + 1]
                                if next_astar_node in possible_nodes:
                                    self.agent.astar_path_index += 1  # Advance index
                                    return next_astar_node
                            break
                
                # If we can't find exact match, find the closest node in possible_nodes to the next A* target
                if self.agent.astar_path_index + 1 < len(self.agent.astar_path):
                    target_node = self.agent.astar_path[self.agent.astar_path_index + 1]
                    best_node = None
                    best_distance = float('inf')
                    
                    for node in possible_nodes:
                        try:
                            dist = full_graph_manager.get_edge_weight(node, target_node)
                            if dist < best_distance:
                                best_distance = dist
                                best_node = node
                        except:
                            continue
                    
                    if best_node:
                        return best_node
            
            # Fallback: find node that moves toward target station
            target_nodes = full_graph_manager.get_station_nodes(self.agent.target_station_id)
            if target_nodes:
                best_node = None
                best_distance = float('inf')
                
                for node in possible_nodes:
                    for target_node in target_nodes:
                        try:
                            dist = full_graph_manager.get_edge_weight(node, target_node)
                            if dist < best_distance:
                                best_distance = dist
                                best_node = node
                        except:
                            continue
                
                return best_node
            
            return None

        async def move_to_node(self, next_node: str):
            """Execute movement to next graph node"""
            if not self.agent.current_graph_node or not next_node:
                astar_logger.error(f"Bus {self.bus_id + 1} movement failed - invalid nodes: current={self.agent.current_graph_node}, next={next_node}")
                return
            
            # Prevent moving to the same node
            if self.agent.current_graph_node == next_node:
                astar_logger.warning(f"Bus {self.bus_id + 1} attempted to move to same node: {next_node}")
                return
            
            # Track node visits for current trip
            self.agent.current_trip_visited_nodes[next_node] = self.agent.current_trip_visited_nodes.get(next_node, 0) + 1
            
            # Get edge weight and calculate travel time
            edge_weight = full_graph_manager.get_edge_weight(self.agent.current_graph_node, next_node)
            base_speed = 60  # km/h
            
            # Calculate effective travel time
            passenger_slowdown = min(0.3, self.agent.passenger_count / 200.0)
            env_effect = 1.0
            
            # Handle incidents
            edge_id = full_graph_manager.get_edge_id(self.agent.current_graph_node, next_node)
            incident_type = None
            
            if edge_id in simulation_state['active_incidents']:
                incident = simulation_state['active_incidents'][edge_id]
                incident_type = incident['type']
                if incident_type == 'light_traffic':
                    env_effect = 0.8
                elif incident_type == 'heavy_traffic':
                    env_effect = 0.5
                elif incident_type == 'closed_road':
                    env_effect = 0.1
            
            effective_speed = base_speed * (1 - passenger_slowdown) * env_effect
            travel_time_hours = (edge_weight / 1000.0) / effective_speed if effective_speed > 0 else (edge_weight / 1000.0) / 10
            travel_time_real_seconds = travel_time_hours * 3600 / system_time.time_multiplier
            
            if incident_type:
                astar_logger.warning(f"[INCIDENT] Bus {self.bus_id + 1} affected by {incident_type} on Edge {edge_id}")
            
            # Check if node is being revisited
            visit_count = self.agent.current_trip_visited_nodes.get(next_node, 0)
            revisit_indicator = f" (Revisit #{visit_count})" if visit_count > 1 else ""
            
            astar_logger.info(f"[MOVEMENT] Bus {self.bus_id + 1} traveling Edge {edge_id} "
                             f"Nodes ({self.agent.current_graph_node} -> {next_node}). "
                             f"Distance: {edge_weight:.1f}m, ETA: {travel_time_hours:.2f}h{revisit_indicator}")
            
            # Update path tracking
            self.agent.current_path_nodes.append(self.agent.current_graph_node)
            self.agent.current_path_edges.append(edge_id)
            self.agent.route_steps += 1  # Increment route steps
            
            # Execute the actual movement
            await asyncio.sleep(max(0.1, travel_time_real_seconds))
            
            # Update position
            old_node = self.agent.current_graph_node
            self.agent.current_graph_node = next_node
            
            # Enhanced logging with A* path progress
            if self.agent.astar_path_calculated and self.agent.astar_path:
                path_progress = f"{self.agent.astar_path_index}/{len(self.agent.astar_path)}"
                astar_logger.info(f"[POSITION-UPDATE] Bus {self.bus_id + 1} moved from Node {old_node} to Node {next_node} | A* Progress: {path_progress}")
            else:
                astar_logger.info(f"[POSITION-UPDATE] Bus {self.bus_id + 1} moved from Node {old_node} to Node {next_node}")
            
            # Update nearest station based on new position
            nearest_station = full_graph_manager.get_nearest_station_to_node(self.agent.current_graph_node)
            if nearest_station is not None:
                self.agent.current_station_id = nearest_station

        async def handle_station_arrival(self):
            """Handle arrival at target station"""
            target_station = bus_lines_manager.get_station(self.agent.target_station_id)
            
            if target_station:
                # Increment total routes completed (station to station)
                self.agent.total_routes_completed += 1
                
                # Reset visited nodes for next trip
                self.agent.current_trip_visited_nodes = {}
                
                # Calculate total route distance
                total_distance = sum(full_graph_manager.get_edge_weight(
                    self.agent.current_path_nodes[i] if i < len(self.agent.current_path_nodes) else self.agent.current_graph_node, 
                    self.agent.current_path_nodes[i+1] if i+1 < len(self.agent.current_path_nodes) else self.agent.current_graph_node
                ) for i in range(len(self.agent.current_path_nodes) - 1)) if len(self.agent.current_path_nodes) > 1 else 0
                
                astar_logger.info(f"[ARRIVAL] ✅✅✅✅✅✅ ✅✅✅✅✅✅ Bus {self.bus_id + 1} reached {target_station.name} | "
                                 f"Route {self.agent.total_routes_completed} | Steps: {self.agent.route_steps} | "
                                 f"Distance: {total_distance:.1f}m | Time: {system_time.get_current_time():.2f}h")
                
                # Reset path tracking for next route
                self.agent.current_path_nodes = []
                self.agent.current_path_edges = []
                self.agent.route_steps = 0
                self.agent.astar_path_calculated = False
                
                # Request next target
                await self.request_next_target()
                
                # Handle passenger unloading/loading at station
                await self.handle_passengers_at_station()
                
                # Request new target
                await self.request_next_target()
                
                # Update metrics
                trip_duration = system_time.get_current_time() - self.agent.trip_start_time
                simulation_state['bus_performance_metrics']['trip_durations'].append(trip_duration)
                simulation_state['bus_performance_metrics']['passengers_served'].append(self.agent.passenger_count)
                
                # Reset path tracking for next trip
                self.agent.current_path = []
                self.agent.current_path_nodes = []
                self.agent.current_path_edges = []
                self.agent.current_trip_start_station = self.agent.current_station_id
                self.agent.steps_taken = 0
                self.agent.route_steps = 0  # Reset route steps counter

        async def handle_passengers_at_station(self):
            """Handle passenger loading/unloading at current station"""
            current_station = bus_lines_manager.get_station(self.agent.current_station_id)
            if current_station:
                # Unload some passengers
                unloading = random.randint(0, min(10, self.agent.passenger_count))
                self.agent.passenger_count -= unloading
                
                # Load new passengers
                loading = random.randint(0, 15)
                self.agent.passenger_count = min(60, self.agent.passenger_count + loading)  # Max 60 passengers
                
                if unloading > 0 or loading > 0:
                    bus_logger.info(f"[PASSENGERS] Bus {self.bus_id + 1} at {current_station.name}: "
                                  f"unloaded {unloading}, loaded {loading} passengers (total: {self.agent.passenger_count})")

        async def request_next_target(self):
            """Request next target station from control agent"""
            msg = Message(to=CONTROL_JID)
            msg.set_metadata("performative", "request")
            msg.body = f"NEXT_TARGET:{self.bus_id}:{self.agent.assigned_line_id}:{self.agent.current_station_id}:{self.agent.direction}"
            await self.send(msg)

    class MessageHandler(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if msg and isinstance(msg, Message):
                if "LINE_ASSIGNMENT" in msg.body:
                    await self.handle_line_assignment(msg)
                elif "NEXT_TARGET" in msg.body:
                    await self.handle_next_target(msg)
                elif "INCIDENT_UPDATE" in msg.body:
                    await self.handle_incident_update(msg)

        async def handle_line_assignment(self, msg):
            """Handle line assignment message"""
            try:
                _, assignment_data = msg.body.split(":", 1)
                assignment = json.loads(assignment_data)
                
                # Set assignment data directly on agent
                self.agent.assigned_line_id = assignment['line_id']
                self.agent.current_station_id = assignment['start_station_id']
                self.agent.target_station_id = assignment['target_station_id']
                self.agent.direction = assignment.get('direction', 'forward')
                self.agent.passenger_count = assignment.get('passenger_count', random.randint(15, 45))
                self.agent.trip_start_time = system_time.get_current_time()
                
                # Reset A* path tracking for new route
                self.agent.astar_path = []
                self.agent.astar_path_index = 0
                self.agent.astar_path_calculated = False
                
                # Calculate timeout threshold
                line = bus_lines_manager.get_line(self.agent.assigned_line_id)
                if line:
                    self.agent.timeout_threshold = len(line.stations) * 10
                    self.agent.astar_decision_count = 0
                
                # Initialize path tracking for this trip
                self.agent.current_path_nodes = []
                self.agent.current_trip_start_station = self.agent.current_station_id
                
                # Calculate A* path for reference
                if self.agent.current_station_id is not None and self.agent.target_station_id is not None:
                    astar_path, astar_edges, astar_distance = full_graph_manager.a_star_path_finding(
                        self.agent.current_station_id, self.agent.target_station_id
                    )
                    
                    # Enhanced logging at initialization
                    current_station = bus_lines_manager.get_station(self.agent.current_station_id)
                    target_station = bus_lines_manager.get_station(self.agent.target_station_id)
                    
                    bus_logger.info(f"[ROUTE-INIT] Bus {self.agent.bus_id + 1} | "
                                  f"From: {current_station.name if current_station else 'Unknown'} -> "
                                  f"To: {target_station.name if target_station else 'Unknown'} | "
                                  f"A* Steps: {len(astar_path) - 1 if astar_path else 0} | "
                                  f"A* Distance: {astar_distance:.1f}m")
                else:
                    bus_logger.warning(f"Bus {self.agent.bus_id + 1} assignment missing station IDs")
                
                # Reset steps counter for this route segment
                self.agent.route_steps = 0
                
                # Update the Bus Behaviour state
                for behaviour in self.agent.behaviours:
                    if hasattr(behaviour, 'waiting_for_assignment'):
                        behaviour.waiting_for_assignment = False
                        behaviour.is_active = True
                        bus_logger.debug(f"Updated behaviour state for bus {self.agent.bus_id}")
                        break
                
                # Send acknowledgment
                ack = Message(to=CONTROL_JID)
                ack.set_metadata("performative", "confirm")
                ack.body = f"LINE_ACCEPTED:{self.agent.bus_id}"
                await self.send(ack)
                
                current_station = bus_lines_manager.get_station(self.agent.current_station_id)
                target_station = bus_lines_manager.get_station(self.agent.target_station_id)
                
                bus_logger.info(f"Bus {self.agent.bus_id + 1} assigned to line {self.agent.assigned_line_id}: "
                              f"{current_station.name if current_station else 'Unknown'} -> "
                              f"{target_station.name if target_station else 'Unknown'}")
                              
                # Debug log the assignment - FORCE INFO LEVEL
                bus_logger.info(f"[ASSIGNMENT-DEBUG] Bus {self.agent.bus_id + 1} assignment details: line_id={self.agent.assigned_line_id}, current={self.agent.current_station_id + 1 if self.agent.current_station_id is not None else None}, target={self.agent.target_station_id + 1 if self.agent.target_station_id is not None else None}")
                
                # Add small delay to ensure assignment persists
                await asyncio.sleep(0.1)
                
                # Verify assignment is still set after delay
                bus_logger.info(f"[ASSIGNMENT-VERIFY] Bus {self.agent.bus_id} after delay: line_id={self.agent.assigned_line_id}")
                
                # Force an immediate state update to reflect the assignment with proper current_city
                for behaviour in self.agent.behaviours:
                    if hasattr(behaviour, 'update_bus_state'):
                        await behaviour.update_bus_state()
                        break
                        
            except Exception as e:
                bus_logger.error(f"Bus {self.agent.bus_id} error processing line assignment: {e}")

        async def handle_next_target(self, msg):
            """Handle next target assignment"""
            try:
                _, target_data = msg.body.split(":", 1)
                target_info = json.loads(target_data)
                
                self.agent.target_station_id = target_info['target_station_id']
                self.agent.direction = target_info.get('direction', self.agent.direction)
                self.agent.trip_start_time = system_time.get_current_time()
                self.agent.using_astar = True  # Reset A* for new segment
                self.agent.current_path = []
                self.agent.route_steps = 0  # Reset route steps for new segment
                
                # Reset A* path tracking for new route
                self.agent.astar_path = []
                self.agent.astar_path_index = 0
                self.agent.astar_path_calculated = False
                
                # Calculate A* path for the new route segment and initialize tracking
                if self.agent.current_station_id is not None and self.agent.target_station_id is not None:
                    astar_path, astar_edges, astar_distance = full_graph_manager.a_star_path_finding(
                        self.agent.current_station_id, self.agent.target_station_id
                    )
                    
                    # Enhanced logging for new target assignment
                    current_station = bus_lines_manager.get_station(self.agent.current_station_id)
                    target_station = bus_lines_manager.get_station(self.agent.target_station_id)
                    
                    bus_logger.info(f"[ROUTE-INIT] Bus {self.agent.bus_id + 1} | "
                                  f"From: {current_station.name if current_station else 'Unknown'} -> "
                                  f"To: {target_station.name if target_station else 'Unknown'} | "
                                  f"A* Steps: {len(astar_path) - 1 if astar_path else 0} | "
                                  f"A* Distance: {astar_distance:.1f}m")
                
                bus_logger.info(f"Bus {self.agent.bus_id + 1} received next target: Station {self.agent.target_station_id + 1}")
                
                # Update bus state immediately to reflect new target
                # Find the behavior instance to call update_bus_state
                for behavior in self.agent.behaviours:
                    if hasattr(behavior, 'update_bus_state'):
                        await behavior.update_bus_state()
                        break
                
                bus_logger.info(f"[TARGET-UPDATE] Bus {self.agent.bus_id + 1} state updated: current={self.agent.current_station_id + 1 if self.agent.current_station_id is not None else None}, target={self.agent.target_station_id + 1 if self.agent.target_station_id is not None else None}")
                
            except Exception as e:
                bus_logger.error(f"Error handling next target: {e}")

        async def handle_incident_update(self, msg):
            """Handle incident updates from control"""
            try:
                _, incident_data = msg.body.split(":", 1)
                incident_info = json.loads(incident_data)
                
                # Update local incident awareness
                bus_logger.info(f"Bus {self.agent.bus_id} received incident update: {incident_info}")
                
            except Exception as e:
                bus_logger.error(f"Error handling incident update: {e}")


############################################################################################################
############################################################################################################
#-----------------------------          STATION AGENT          ---------------------------------------------
############################################################################################################
############################################################################################################

class StationAgent(Agent):
    """Station Agent for managing passenger requests and bus arrivals"""
    
    async def setup(self):
        self.station_id = int(str(self.jid).split('_')[1].split('@')[0])
        self.is_initialized = False
        self.waiting_passengers = []
        self.registration_attempts = 0
        self.max_registration_attempts = 3
        
        if DEBUG_MESSAGING:
            print(MESSAGE_FORMATS['station'].format(self.station_id, system_time.get_current_time(), 
                                                   "Station Agent starting setup..."))
        
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(self.StationInitBehaviour())
        self.add_behaviour(self.StationBehaviour(self.station_id))
        station_logger.info(f"Station {self.station_id + 1} setup complete")

    class StationInitBehaviour(CyclicBehaviour):
        async def run(self):
            if not self.agent.is_initialized and self.agent.registration_attempts < self.agent.max_registration_attempts:
                try:
                    msg = Message(to=CONTROL_JID)
                    msg.set_metadata("performative", "subscribe")
                    msg.body = f"REGISTER:{self.agent.station_id}"
                    log_message("SEND", f"Station_{self.agent.station_id}", "Control", msg.body, system_time, "INIT_CONFIRM")
                    await self.send(msg)
                
                    response = await self.receive(timeout=10)
                    
                    if response and response.body == "INIT_CONFIRM":
                        self.agent.is_initialized = True
                        station_logger.info(f"Station {self.agent.station_id + 1} initialized")
                        self.kill() # Stop this behavior after initialization
                    else:
                        self.agent.registration_attempts += 1
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    station_logger.error(f"Station {self.agent.station_id} initialization error: {e}")
                    self.agent.registration_attempts += 1
                    await asyncio.sleep(2)
            elif self.agent.registration_attempts >= self.agent.max_registration_attempts:
                station_logger.error(f"Station {self.agent.station_id} failed to initialize after {self.agent.max_registration_attempts} attempts.")
                self.kill()
            else:
                await asyncio.sleep(5)

    class StationBehaviour(CyclicBehaviour):
        def __init__(self, station_id):
            super().__init__()
            self.station_id = station_id

        async def run(self):
            if not self.agent.is_initialized:
                await asyncio.sleep(1)
                return

            # Generate passenger requests
            await self.generate_passenger_requests()
            
            # Process incoming messages
            msg = await self.receive(timeout=1)
            if msg:
                await self.handle_message(msg)
            
            await asyncio.sleep(2)

        async def generate_passenger_requests(self):
            """Generate passenger requests with target distribution of 5 passengers per 30 minutes"""
            # Generate passengers with proper distribution
            # Target: 5 passengers per 30 minutes = 0.1667 passengers per minute
            # In simulation time with time_multiplier, this becomes more frequent
            
            # Calculate probability based on time multiplier
            # We check every 1 second (real time), so probability should be:
            # (5 passengers / 30 minutes) * (1 minute / 60 seconds) * time_multiplier
            passenger_rate_per_second = (5.0 / (30.0 * 60.0)) * system_time.time_multiplier
            
            # Use random generation based on Poisson process
            if random.random() < passenger_rate_per_second:
                # Generate 1-3 passengers at a time
                num_passengers = random.randint(1, 3)
                
                # Choose random destination from other stations
                all_stations = bus_lines_manager.get_all_stations()
                available_destinations = [s.id for s in all_stations.values() if s.id != self.station_id]
                
                if available_destinations:
                    destination = random.choice(available_destinations)
                    destination_station = bus_lines_manager.get_station(destination)
                    
                    passenger_request = {
                        'origin_station': self.station_id,
                        'destination_station': destination,
                        'passenger_count': num_passengers,
                        'timestamp': system_time.get_current_time(),
                        'wait_time': 0.0
                    }
                    
                    self.agent.waiting_passengers.append(passenger_request)
                    simulation_state['passenger_requests'].append(passenger_request)
                    
                    station_logger.info(f"Station {self.station_id + 1} generated {num_passengers} passenger requests to {destination_station.name}")
            
            # Update wait times for existing passengers
            current_time = system_time.get_current_time()
            for passenger in self.agent.waiting_passengers:
                passenger['wait_time'] = current_time - passenger['timestamp']

        async def handle_message(self, msg):
            """Handle incoming messages"""
            if "BUS_ARRIVED" in msg.body:
                await self.handle_bus_arrival(msg)
            elif "INCIDENT" in msg.body:
                await self.handle_incident_report(msg)

        async def handle_bus_arrival(self, msg):
            """Handle bus arrival at station"""
            try:
                parts = msg.body.split(":")
                bus_id = int(parts[1])
                
                # Load passengers onto bus (simplified)
                if self.agent.waiting_passengers:
                    loaded_passengers = self.agent.waiting_passengers[:5]  # Load up to 5 passenger groups
                    self.agent.waiting_passengers = self.agent.waiting_passengers[5:]
                    
                    station_logger.info(f"Station {self.station_id + 1} loaded {len(loaded_passengers)} passenger groups onto Bus {bus_id + 1}")
                
            except Exception as e:
                station_logger.error(f"Station {self.station_id} error handling bus arrival: {e}")

        async def handle_incident_report(self, msg):
            """Handle incident reports"""
            station_logger.info(f"Station {self.station_id} received incident report: {msg.body}")


############################################################################################################
############################################################################################################
#-----------------------------          CONTROL AGENT          -----------------------------------------
############################################################################################################
############################################################################################################

class ControlAgent(Agent):
    """Enhanced Control Agent with line-based bus management and A* routing"""
    
    async def setup(self):
        self.registered_buses = []
        self.registered_stations = []
        self.reserve_buses = {}  # {line_id: count}
        self.active_assignments = {}
        self.bus_schedule_tracker = {}  # Track scheduled launches
        self.incident_manager = IncidentManager()
        
        # Initialize reserve buses and schedule for each line
        for line_id, line in bus_lines_manager.get_all_lines().items():
            self.reserve_buses[line_id] = CONFIG['buses']['reserve_buses_per_line']
            self.bus_schedule_tracker[line_id] = {
                'buses_launched': 0,
                'last_launch_time': -1,  # Set to -1 so first bus launches immediately
                'launch_interval': CONFIG['buses']['launch_interval_hours']
            }
        
        if DEBUG_MESSAGING:
            print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), 
                                                   "Control Agent starting setup..."))

        template = Template()
        template.set_metadata("performative", "subscribe")
        self.add_behaviour(self.ControlBehaviour())
        self.add_behaviour(self.LineManagementBehaviour())
        self.add_behaviour(self.IncidentManagementBehaviour())
        
        control_logger.info("Control Agent setup complete")

    class ControlBehaviour(CyclicBehaviour):
        async def run(self):
            template = Template()
            template.set_metadata("performative", "subscribe")
            msg = await self.receive(timeout=1)

            if msg:
                if "REGISTER:" in msg.body:
                    await self.handle_registration(msg)
                elif "NEXT_TARGET:" in msg.body:
                    await self.handle_next_target_request(msg)

        async def handle_registration(self, msg):
            """Handle agent registration"""
            try:
                entity_id = int(msg.body.split(":")[1])
                sender_type = "bus" if "bus_" in str(msg.sender) else "station"
                
                if sender_type == "bus" and entity_id not in self.agent.registered_buses:
                    self.agent.registered_buses.append(entity_id)
                    control_logger.info(f"Registered Bus {entity_id + 1}")
                elif sender_type == "station" and entity_id not in self.agent.registered_stations:
                    self.agent.registered_stations.append(entity_id)
                    control_logger.info(f"Registered Station {entity_id + 1}")

                # Send confirmation
                response = Message(to=str(msg.sender))
                response.set_metadata("performative", "inform")
                response.body = "INIT_CONFIRM"
                await self.send(response)
                
            except Exception as e:
                control_logger.error(f"Registration error: {e}")

        async def handle_next_target_request(self, msg):
            """Handle next target requests from buses"""
            try:
                parts = msg.body.split(":")
                bus_id = int(parts[1])
                line_id = int(parts[2])
                current_station_id = int(parts[3])
                direction = parts[4]
                
                control_logger.info(f"[NEXT-TARGET] Bus {bus_id + 1} requesting next target from station {current_station_id + 1} going {direction}")
                
                # Determine next target
                line = bus_lines_manager.get_line(line_id)
                if not line:
                    control_logger.error(f"Line {line_id} not found for bus {bus_id}")
                    return
                
                current_station = bus_lines_manager.get_station(current_station_id)
                if not current_station:
                    control_logger.error(f"Current station {current_station_id} not found for bus {bus_id}")
                    return
                
                # Find current station index in line
                current_index = None
                for i, station in enumerate(line.stations):
                    if station.id == current_station_id:
                        current_index = i
                        break
                
                if current_index is None:
                    control_logger.error(f"Station {current_station_id} not found in line {line_id} for bus {bus_id}")
                    return
                
                # Determine next target based on direction
                next_target_id = None
                new_direction = direction
                
                control_logger.info(f"[DIRECTION-LOGIC] Bus {bus_id}: current_index={current_index}, direction={direction}, line_length={len(line.stations)}")
                
                if direction == "forward":
                    if current_index + 1 < len(line.stations):
                        # Normal forward movement
                        next_target_id = line.stations[current_index + 1].id
                        control_logger.info(f"[FORWARD] Bus {bus_id + 1} moving forward to station {next_target_id + 1}")
                    else:
                        # Reached end, turn around
                        new_direction = "backward"
                        if current_index - 1 >= 0:
                            next_target_id = line.stations[current_index - 1].id
                            control_logger.info(f"[TURN-AROUND] Bus {bus_id + 1} reached end, turning around to station {next_target_id + 1}")
                        else:
                            # Only one station in line - stay put and continue forward
                            next_target_id = current_station_id
                            new_direction = "forward"
                            control_logger.info(f"[SINGLE-STATION] Bus {bus_id} staying at single station")
                
                elif direction == "backward":
                    if current_index - 1 >= 0:
                        # Normal backward movement
                        next_target_id = line.stations[current_index - 1].id
                        control_logger.info(f"[BACKWARD] Bus {bus_id + 1} moving backward to station {next_target_id + 1}")
                    else:
                        # Reached start, turn around
                        new_direction = "forward"
                        if current_index + 1 < len(line.stations):
                            next_target_id = line.stations[current_index + 1].id
                            control_logger.info(f"[TURN-AROUND] Bus {bus_id + 1} reached start, turning around to station {next_target_id + 1}")
                        else:
                            # Only one station in line - stay put and continue backward
                            next_target_id = current_station_id
                            new_direction = "backward"
                            control_logger.info(f"[SINGLE-STATION] Bus {bus_id} staying at single station")
                
                # Validate and send the target assignment
                control_logger.info(f"[FINAL-CHECK] Bus {bus_id + 1}: next_target_id={next_target_id + 1 if next_target_id is not None else None}, current_station_id={current_station_id + 1}")
                
                if next_target_id is not None and next_target_id != current_station_id:
                    # Send next target assignment
                    response = Message(to=get_bus_jid(bus_id))
                    response.set_metadata("performative", "inform")
                    response.body = f"NEXT_TARGET:{json.dumps({
                        'target_station_id': next_target_id,
                        'direction': new_direction
                    })}"
                    await self.send(response)
                    
                    control_logger.info(f"✅ Assigned next target to Bus {bus_id + 1}: Station {next_target_id + 1} ({new_direction})")
                elif next_target_id is None:
                    control_logger.warning(f"⚠️ Could not determine valid next target for Bus {bus_id + 1} (next_target_id is None)")
                    # Emergency assignment - try to assign any valid adjacent station
                    await self.emergency_target_assignment(bus_id, line_id, current_station_id)
                else:
                    control_logger.warning(f"⚠️ Prevented same-station assignment for Bus {bus_id + 1} (would assign current station {current_station_id + 1})")
                    # Emergency assignment - try to assign any valid adjacent station
                    await self.emergency_target_assignment(bus_id, line_id, current_station_id)
                
            except Exception as e:
                control_logger.error(f"Next target assignment error: {e}", exc_info=True)

        async def emergency_target_assignment(self, bus_id: int, line_id: int, current_station_id: int):
            """Emergency assignment when normal logic fails"""
            try:
                line = bus_lines_manager.get_line(line_id)
                if not line or len(line.stations) < 2:
                    control_logger.error(f"Emergency assignment failed - invalid line {line_id}")
                    return
                
                # Find any valid adjacent station
                current_index = None
                for i, station in enumerate(line.stations):
                    if station.id == current_station_id:
                        current_index = i
                        break
                
                if current_index is None:
                    control_logger.error(f"Emergency assignment failed - station {current_station_id} not in line {line_id}")
                    return
                
                # Try forward first, then backward
                emergency_target = None
                emergency_direction = "forward"
                
                if current_index + 1 < len(line.stations):
                    emergency_target = line.stations[current_index + 1].id
                    emergency_direction = "forward"
                elif current_index - 1 >= 0:
                    emergency_target = line.stations[current_index - 1].id
                    emergency_direction = "backward"
                
                if emergency_target is not None and emergency_target != current_station_id:
                    response = Message(to=get_bus_jid(bus_id))
                    response.set_metadata("performative", "inform")
                    response.body = f"NEXT_TARGET:{json.dumps({
                        'target_station_id': emergency_target,
                        'direction': emergency_direction
                    })}"
                    await self.send(response)
                    
                    control_logger.info(f"🚨 EMERGENCY assignment for Bus {bus_id + 1}: Station {emergency_target + 1} ({emergency_direction})")
                else:
                    control_logger.error(f"🚨 EMERGENCY assignment failed - no valid target for Bus {bus_id + 1}")
                    
            except Exception as e:
                control_logger.error(f"Emergency assignment error: {e}", exc_info=True)

    class LineManagementBehaviour(CyclicBehaviour):
        """Manages bus scheduling and reserve bus releases"""
        
        async def run(self):
            await self.schedule_regular_buses()
            await self.check_reserve_bus_release()
            await asyncio.sleep(5)

        async def schedule_regular_buses(self):
            """Schedule regular buses (3 per line, 20-minute intervals)"""
            current_time = system_time.get_current_time()
            
            for line_id, line in bus_lines_manager.get_all_lines().items():
                schedule_info = self.agent.bus_schedule_tracker.get(line_id, {})
                
                # Check if it's time to launch next bus
                time_since_last = current_time - schedule_info.get('last_launch_time', 0)
                buses_launched = schedule_info.get('buses_launched', 0)
                launch_interval = schedule_info.get('launch_interval', CONFIG['buses']['launch_interval_hours'])
                
                # DEBUG: Log scheduling check
                available_buses = [b for b in self.agent.registered_buses 
                                 if b not in self.agent.active_assignments]
                
                max_buses = CONFIG['buses']['initial_buses_per_line']
                if len(available_buses) > 0 and buses_launched < max_buses:
                    control_logger.info(f"DEBUG - Line {line_id}: time_since_last={time_since_last:.3f}h, launch_interval={launch_interval:.3f}h, buses_launched={buses_launched}, available_buses={len(available_buses)}")
                
                # Launch buses per line with configured intervals
                if buses_launched < max_buses and time_since_last >= launch_interval:
                    if available_buses and len(line.stations) >= 2:
                        bus_id = available_buses[0]
                        
                        # Start from first station, going forward
                        start_station = line.stations[0]
                        target_station = line.stations[1] if len(line.stations) > 1 else line.stations[0]
                        
                        assignment = {
                            'line_id': line_id,
                            'start_station_id': start_station.id,
                            'target_station_id': target_station.id,
                            'direction': 'forward',
                            'passenger_count': random.randint(15, 45)
                        }
                        
                        await self.assign_line_to_bus(bus_id, assignment)
                        
                        # Update schedule tracker
                        self.agent.bus_schedule_tracker[line_id] = {
                            'buses_launched': buses_launched + 1,
                            'last_launch_time': current_time,
                            'launch_interval': launch_interval
                        }
                        
                        # Track bus assignment
                        simulation_state['buses_by_line'][line_id].append(bus_id)
                        
                        control_logger.info(f"Scheduled Bus {bus_id + 1} for line {line_id} (bus {buses_launched + 1}/3)")

        async def check_reserve_bus_release(self):
            """Check if reserve buses should be released based on demand"""
            for line_id, line in bus_lines_manager.get_all_lines().items():
                # Count passengers wanting to go to stations in this line
                line_station_ids = [s.id for s in line.stations]
                line_demand = 0
                
                for request in simulation_state['passenger_requests']:
                    if request['destination_station'] in line_station_ids:
                        line_demand += request['passenger_count']
                
                # Release reserve bus if demand > 70 passengers
                if line_demand >= 70 and self.agent.reserve_buses.get(line_id, 0) > 0:
                    self.agent.reserve_buses[line_id] -= 1
                    
                    # Find available bus
                    available_buses = [b for b in self.agent.registered_buses 
                                     if b not in self.agent.active_assignments]
                    
                    if available_buses and len(line.stations) >= 2:
                        bus_id = available_buses[0]
                        
                        # Choose a terminal station as start
                        terminal_stations = [s for s in line.stations if s.is_terminal]
                        start_station = random.choice(terminal_stations) if terminal_stations else line.stations[0]
                        
                        # Choose target (next station in line)
                        start_idx = line.stations.index(start_station)
                        if start_idx == 0:
                            target_station = line.stations[1] if len(line.stations) > 1 else start_station
                            direction = 'forward'
                        else:
                            target_station = line.stations[start_idx - 1] if start_idx > 0 else start_station
                            direction = 'backward'
                        
                        assignment = {
                            'line_id': line_id,
                            'start_station_id': start_station.id,
                            'target_station_id': target_station.id,
                            'direction': direction,
                            'passenger_count': random.randint(25, 55)
                        }
                        
                        await self.assign_line_to_bus(bus_id, assignment)
                        
                        control_logger.info(f"Released reserve bus {bus_id} for line {line_id} due to high demand ({line_demand} passengers)")

        async def assign_line_to_bus(self, bus_id, assignment):
            """Assign a line-based route to a bus"""
            try:
                msg = Message(to=get_bus_jid(bus_id))
                msg.set_metadata("performative", "inform")
                msg.body = f"LINE_ASSIGNMENT:{json.dumps(assignment)}"
                
                await self.send(msg)
                self.agent.active_assignments[bus_id] = assignment
                
                control_logger.info(f"Assigned line route to Bus {bus_id}: {assignment}")
                
            except Exception as e:
                control_logger.error(f"Line assignment error: {e}")

    class IncidentManagementBehaviour(CyclicBehaviour):
        async def run(self):
            """Manage random incidents"""
            await self.generate_random_incidents()
            await self.cleanup_expired_incidents()
            await asyncio.sleep(10)

        async def generate_random_incidents(self):
            """Generate random incidents on edges between stations"""
            if random.random() < 0.1:  # 10% chance per cycle
                # Choose random edge between stations
                all_stations = list(bus_lines_manager.get_all_stations().values())
                if len(all_stations) < 2:
                    return
                
                station_a = random.choice(all_stations)
                station_b = random.choice([s for s in all_stations if s.id != station_a.id])
                
                incident_types = ['light_traffic', 'heavy_traffic', 'closed_road']
                incident_weights = [0.7, 0.25, 0.05]  # Probabilities
                incident_type = random.choices(incident_types, weights=incident_weights)[0]
                
                duration_map = {
                    'light_traffic': random.uniform(1, 3),  # 1-3 hours
                    'heavy_traffic': random.uniform(2, 6),  # 2-6 hours
                    'closed_road': random.uniform(4, 12)    # 4-12 hours
                }
                
                incident = {
                    'type': incident_type,
                    'start_time': system_time.get_current_time(),
                    'duration': duration_map[incident_type],
                    'edge': (station_a.id, station_b.id)
                }
                
                simulation_state['active_incidents'][(station_a.id, station_b.id)] = incident
                
                control_logger.info(f"Generated incident: {incident_type} on Edge({station_a.name}->{station_b.name}) [ID: {station_a.id}->{station_b.id}]")

        async def cleanup_expired_incidents(self):
            """Remove expired incidents"""
            current_time = system_time.get_current_time()
            expired = []
            
            for edge, incident in simulation_state['active_incidents'].items():
                if current_time - incident['start_time'] > incident['duration']:
                    expired.append(edge)
            
            for edge in expired:
                del simulation_state['active_incidents'][edge]
                control_logger.info(f"Incident on edge {edge} expired")


############################################################################################################
############################################################################################################
#-----------------------------          INCIDENT MANAGER          --------------------------------------
############################################################################################################
############################################################################################################

class IncidentManager:
    """Manages traffic incidents across the network"""
    
    def __init__(self):
        self.active_incidents = {}
        self.incident_history = []
    
    def add_incident(self, station_a_id: int, station_b_id: int, incident_type: str, duration: float):
        """Add a new incident"""
        incident = {
            'type': incident_type,
            'start_time': system_time.get_current_time(),
            'duration': duration,
            'stations': (station_a_id, station_b_id)
        }
        
        self.active_incidents[(station_a_id, station_b_id)] = incident
        self.incident_history.append(incident)
    
    def get_incident(self, station_a_id: int, station_b_id: int) -> Optional[Dict]:
        """Get active incident between two stations"""
        return self.active_incidents.get((station_a_id, station_b_id))
    
    def cleanup_expired_incidents(self):
        """Remove expired incidents"""
        current_time = system_time.get_current_time()
        expired = []
        
        for edge, incident in self.active_incidents.items():
            if current_time - incident['start_time'] > incident['duration']:
                expired.append(edge)
        
        for edge in expired:
            del self.active_incidents[edge]


############################################################################################################
############################################################################################################
#-----------------------------          SIMULATION RUNNER          -------------------------------------
############################################################################################################
############################################################################################################

async def run_simulation(num_buses: int = 15, num_stations: int = 10):
    """Run the main simulation"""
    control_logger.info("Starting enhanced traffic routing simulation with A* and full graph routing")
    
    # Initialize the full graph manager and map stations to graph nodes
    full_graph_manager.map_stations_to_graph_nodes(bus_lines_manager)
    
    # Log graph statistics
    stats = full_graph_manager.get_graph_stats()
    control_logger.info(f"Graph initialized: {stats['total_nodes']} nodes, {stats['total_edges']} edges, "
                       f"{stats['mapped_stations']} mapped stations, avg degree: {stats['average_node_degree']:.2f}")
    
    # Start system time
    asyncio.create_task(system_time.update_time())
    
    # Create and start control agent with auto-registration
    control_agent = ControlAgent(CONTROL_JID, "password")
    try:
        await control_agent.start(auto_register=True)
        control_logger.info(f"✅ Control agent started successfully: {CONTROL_JID}")
    except Exception as e:
        control_logger.error(f"❌ Failed to start control agent: {e}")
        raise
    
    # Create and start station agents (one for each station in all lines)
    station_agents = []
    all_stations = bus_lines_manager.get_all_stations()
    for station_id in all_stations.keys():
        station = StationAgent(get_station_jid(station_id), "password")
        try:
            await station.start(auto_register=True)
            station_agents.append(station)
            control_logger.info(f"✅ Station agent {station_id} started successfully")
        except Exception as e:
            control_logger.error(f"❌ Failed to start station agent {station_id}: {e}")
            # Continue with other stations even if one fails
    
    # Create and start bus agents
    bus_agents = []
    for i in range(num_buses):
        bus = BusAgent(get_bus_jid(i), "password")
        try:
            await bus.start(auto_register=True)
            bus_agents.append(bus)
            control_logger.info(f"✅ Bus agent {i} started successfully")
        except Exception as e:
            control_logger.error(f"❌ Failed to start bus agent {i}: {e}")
            # Continue with other buses even if one fails
    
    simulation_state['running'] = True
    control_logger.info(f"Simulation started with {num_buses} buses and {len(all_stations)} stations")
    
    try:
        # Run simulation
        while simulation_state['running']:
            await asyncio.sleep(10)
            
            # Print periodic status
            active_buses = len([b for b in bus_agents if b.is_initialized])
            active_incidents = len(simulation_state['active_incidents'])
            lines_count = len(bus_lines_manager.get_all_lines())
            
            control_logger.info(f"Status: {active_buses} active buses, {active_incidents} active incidents, {lines_count} bus lines")
    
    except KeyboardInterrupt:
        control_logger.info("Simulation interrupted by user")
    
    finally:
        # Cleanup
        simulation_state['running'] = False
        
        # Stop all agents
        for agent in [control_agent] + station_agents + bus_agents:
            await agent.stop()
        
        control_logger.info("Simulation ended")


############################################################################################################
############################################################################################################
#-----------------------------          MAIN EXECUTION          ----------------------------------------
############################################################################################################
############################################################################################################

def main():
    """Main execution function"""
    # Import logging configuration for dashboard simulation
    from utils import setup_logging, get_logger
    
    # Setup logging for dashboard simulation
    setup_logging("dashboard")
    
    # Get logger for main
    main_logger = get_logger('MAIN')
    
    main_logger.info("🚌 Enhanced Traffic Routing System with A* and Line-Based Routing")
    main_logger.info("=" * 70)
    
    # Check for bus lines
    lines = bus_lines_manager.get_all_lines()
    stations = bus_lines_manager.get_all_stations()
    
    if not lines:
        main_logger.warning("⚠️  No bus lines found. Please create bus lines using the dashboard first.")
        main_logger.warning("   Run the dashboard to create lines before starting simulation.")
        return
    
    main_logger.info(f"✅ Found {len(lines)} bus lines")
    main_logger.info(f"✅ Found {len(stations)} stations")
    
    # Display line information
    for line_id, line in lines.items():
        main_logger.info(f"   📍 Line {line_id}: {line.name} ({len(line.stations)} stations)")
    
    try:
        # Calculate total buses needed (3 per line + some extras)
        total_buses = max(15, len(lines) * 3 + 5)
        asyncio.run(run_simulation(num_buses=total_buses, num_stations=len(stations)))
    except KeyboardInterrupt:
        logging.info("\n🛑 Simulation stopped by user")
    except Exception as e:
        logging.error(f"❌ Simulation error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
