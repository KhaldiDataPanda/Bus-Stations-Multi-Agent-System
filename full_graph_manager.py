"""
Enhanced Graph Manager for Full Road Network Integration
Handles the full road network from Blida_map.graphml and integrates with A* routing
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
import heapq
from graph_loader import GraphLoader
from bus_lines_manager import BusLinesManager, Station
import math

class FullGraphManager:
    def __init__(self, graphml_path: str = "data/Blida_map.graphml"):
        self.logger = logging.getLogger('full_graph_manager')
        self.graph_loader = GraphLoader(graphml_path)
        self.graph = nx.Graph()
        self.station_to_nodes = {}  # Maps station_id to closest graph nodes
        self.node_coordinates = {}  # Maps node_id to (lat, lon)
        self.edge_weights = {}  # Maps edge to weight
        
        self._build_networkx_graph()
        
    def _build_networkx_graph(self):
        """Build NetworkX graph from loaded GraphML data"""
        try:
            # Add all nodes with their coordinates
            for node_id, node_data in self.graph_loader.nodes.items():
                self.graph.add_node(node_id)
                self.node_coordinates[node_id] = (node_data['lat'], node_data['lon'])
            
            # Add all edges with weights based on distance
            for edge_id, edge_data in self.graph_loader.edges.items():
                source = edge_data['source']
                target = edge_data['target']
                
                if source in self.node_coordinates and target in self.node_coordinates:
                    # Calculate edge weight based on geographic distance
                    weight = self._calculate_distance(
                        self.node_coordinates[source][0], self.node_coordinates[source][1],
                        self.node_coordinates[target][0], self.node_coordinates[target][1]
                    )
                    
                    self.graph.add_edge(source, target, weight=weight, edge_id=edge_id)
                    self.edge_weights[edge_id] = weight
            
            self.logger.info(f"Built NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Error building NetworkX graph: {e}")
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two coordinates in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def map_stations_to_graph_nodes(self, bus_lines_manager: BusLinesManager):
        """Map each station to the closest graph nodes"""
        stations = bus_lines_manager.get_all_stations()
        
        for station_id, station in stations.items():
            closest_nodes = self.graph_loader.find_nearest_nodes(station.lat, station.lon, 3)
            if closest_nodes:
                self.station_to_nodes[station_id] = closest_nodes
                self.logger.info(f"Mapped station {station_id} ({station.name}) to graph nodes: {closest_nodes}")
            else:
                self.logger.warning(f"No nearby graph nodes found for station {station_id} ({station.name})")
    
    def get_station_nodes(self, station_id: int) -> List[str]:
        """Get the graph nodes associated with a station"""
        return self.station_to_nodes.get(station_id, [])
    
    def get_possible_next_nodes(self, current_node: str) -> List[str]:
        """Get all possible next nodes from current node"""
        if current_node not in self.graph:
            return []
        return list(self.graph.neighbors(current_node))
    
    def get_edge_id(self, source_node: str, target_node: str) -> str:
        """Get edge ID between two nodes"""
        if self.graph.has_edge(source_node, target_node):
            edge_data = self.graph[source_node][target_node]
            return edge_data.get('edge_id', f"{source_node}_{target_node}")
        return f"{source_node}_{target_node}"
    
    def get_edge_weight(self, source_node: str, target_node: str) -> float:
        """Get edge weight between two nodes"""
        if self.graph.has_edge(source_node, target_node):
            return self.graph[source_node][target_node].get('weight', 1000.0)
        return float('inf')
    
    def a_star_path_finding(self, start_station_id: int, end_station_id: int) -> Tuple[List[str], List[str], float]:
        """
        Find optimal path between two stations using A* on the full graph
        Returns: (path_nodes, path_edges, total_distance)
        """
        start_nodes = self.get_station_nodes(start_station_id)
        end_nodes = self.get_station_nodes(end_station_id)
        
        if not start_nodes or not end_nodes:
            self.logger.warning(f"Cannot find path: start_nodes={start_nodes}, end_nodes={end_nodes}")
            return [], [], float('inf')
        
        best_path = None
        best_edges = None
        best_distance = float('inf')
        
        # Try all combinations of start and end nodes to find the shortest path
        for start_node in start_nodes:
            for end_node in end_nodes:
                try:
                    path = nx.astar_path(self.graph, start_node, end_node, 
                                       heuristic=self._heuristic, weight='weight')
                    distance = nx.astar_path_length(self.graph, start_node, end_node, 
                                                  heuristic=self._heuristic, weight='weight')
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                        best_edges = [self.get_edge_id(path[i], path[i+1]) for i in range(len(path)-1)]
                        
                except nx.NetworkXNoPath:
                    continue
        
        if best_path:
            self.logger.info(f"A* path from station {start_station_id} to {end_station_id}: "
                           f"{len(best_path)} nodes, distance: {best_distance:.2f}m")
            return best_path, best_edges, best_distance
        else:
            self.logger.warning(f"No path found between stations {start_station_id} and {end_station_id}")
            return [], [], float('inf')
    
    def _heuristic(self, node1: str, node2: str) -> float:
        """Heuristic function for A* (straight-line distance)"""
        if node1 not in self.node_coordinates or node2 not in self.node_coordinates:
            return 0
        
        lat1, lon1 = self.node_coordinates[node1]
        lat2, lon2 = self.node_coordinates[node2]
        return self._calculate_distance(lat1, lon1, lat2, lon2)
    
    def get_node_coordinates(self, node_id: str) -> Tuple[float, float]:
        """Get coordinates for a node"""
        return self.node_coordinates.get(node_id, (0, 0))
    
    def is_node_near_station(self, node_id: str, station_id: int, max_distance: float = 200.0) -> bool:
        """Check if a node is near a specific station"""
        station_nodes = self.get_station_nodes(station_id)
        
        if node_id in station_nodes:
            return True
        
        # Check distance to all station nodes
        if node_id in self.node_coordinates:
            node_lat, node_lon = self.node_coordinates[node_id]
            for station_node in station_nodes:
                if station_node in self.node_coordinates:
                    station_lat, station_lon = self.node_coordinates[station_node]
                    distance = self._calculate_distance(node_lat, node_lon, station_lat, station_lon)
                    if distance <= max_distance:
                        return True
        
        return False
    
    def get_nearest_station_to_node(self, node_id: str) -> Optional[int]:
        """Find the nearest station to a given node"""
        if node_id not in self.node_coordinates:
            return None
        
        node_lat, node_lon = self.node_coordinates[node_id]
        min_distance = float('inf')
        nearest_station = None
        
        for station_id, station_nodes in self.station_to_nodes.items():
            for station_node in station_nodes:
                if station_node in self.node_coordinates:
                    station_lat, station_lon = self.node_coordinates[station_node]
                    distance = self._calculate_distance(node_lat, node_lon, station_lat, station_lon)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_station = station_id
        
        return nearest_station
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'mapped_stations': len(self.station_to_nodes),
            'average_node_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
    
    def get_bus_position_on_edge(self, start_node: str, end_node: str, progress_ratio: float = 0.5) -> Tuple[float, float]:
        """
        Calculate bus position along an edge based on progress ratio
        Args:
            start_node: Starting node of the edge
            end_node: Ending node of the edge  
            progress_ratio: Progress along edge (0.0 = start, 1.0 = end, 0.5 = middle)
        Returns:
            Tuple of (latitude, longitude) for the bus position
        """
        if start_node not in self.node_coordinates or end_node not in self.node_coordinates:
            # Fallback to start node if coordinates not available
            return self.get_node_coordinates(start_node)
        
        start_lat, start_lon = self.node_coordinates[start_node]
        end_lat, end_lon = self.node_coordinates[end_node]
        
        # Clamp progress ratio between 0 and 1
        progress_ratio = max(0.0, min(1.0, progress_ratio))
        
        # Linear interpolation between start and end coordinates
        bus_lat = start_lat + (end_lat - start_lat) * progress_ratio
        bus_lon = start_lon + (end_lon - start_lon) * progress_ratio
        
        return bus_lat, bus_lon
    
    def get_path_coordinates(self, path_nodes: List[str]) -> List[Tuple[float, float]]:
        """
        Convert a list of node IDs to coordinate pairs for path visualization
        Args:
            path_nodes: List of node IDs representing the path
        Returns:
            List of (lat, lon) tuples for the path
        """
        coordinates = []
        for node_id in path_nodes:
            if node_id in self.node_coordinates:
                coordinates.append(self.node_coordinates[node_id])
        return coordinates
