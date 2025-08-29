"""
Graph Loader for Blida Map
Loads data from Blida_map.graphml and provides node/coordinate information
"""
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class GraphLoader:
    def __init__(self, graphml_path: str = "data/Blida_map.graphml"):
        self.graphml_path = graphml_path
        self.nodes = {}
        self.edges = {}
        self.coordinate_bounds = {}
        self.load_graph()
    
    def load_graph(self):
        """Load the GraphML file and extract node coordinates and edges"""
        try:
            tree = ET.parse(self.graphml_path)
            root = tree.getroot()
            
            # Define namespaces, handling the default namespace for GraphML
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            default_ns_match = re.match(r'\{([^}]+)\}', root.tag)
            if default_ns_match:
                ns['default'] = default_ns_match.group(1)
                find_prefix = 'default:'
            else:
                find_prefix = ''

            # Extract key definitions to understand data attributes
            keys = {}
            for key in root.findall(f'.//{find_prefix}key', ns):
                key_id = key.get('id')
                key_name = key.get('attr.name', '')
                keys[key_id] = key_name
            
            # Extract nodes with coordinates
            graph = root.find(f'.//{find_prefix}graph', ns)
            if graph is not None:
                for node in graph.findall(f'{find_prefix}node', ns):
                    node_id = node.get('id')
                    node_data = {}
                    
                    for data in node.findall(f'{find_prefix}data', ns):
                        key_id = data.get('key')
                        if key_id in keys:
                            node_data[keys[key_id]] = data.text
                    
                    # Extract coordinates (usually stored as 'x', 'y' or 'lat', 'lon')
                    lat = None
                    lon = None
                    
                    # Try different coordinate attribute names
                    for coord_key in ['y', 'lat', 'latitude']:
                        if coord_key in node_data:
                            try:
                                lat = float(node_data[coord_key])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    for coord_key in ['x', 'lon', 'lng', 'longitude']:
                        if coord_key in node_data:
                            try:
                                lon = float(node_data[coord_key])
                                break
                            except (ValueError, TypeError):
                                continue
                    
                    if lat is not None and lon is not None:
                        self.nodes[node_id] = {
                            'lat': lat,
                            'lon': lon,
                            'data': node_data
                        }
                
                # Extract edges
                for edge in graph.findall(f'{find_prefix}edge', ns):
                    # Generate a unique edge identifier from source and target
                    source = edge.get('source')
                    target = edge.get('target')
                    edge_id = f"{source}_{target}"
                    
                    edge_data = {}
                    for data in edge.findall(f'{find_prefix}data', ns):
                        key_id = data.get('key')
                        if key_id in keys:
                            edge_data[keys[key_id]] = data.text
                    
                    self.edges[edge_id] = {
                        'source': source,
                        'target': target,
                        'data': edge_data
                    }
            
            # Calculate coordinate bounds
            if self.nodes:
                lats = [node['lat'] for node in self.nodes.values()]
                lons = [node['lon'] for node in self.nodes.values()]
                self.coordinate_bounds = {
                    'lat_min': min(lats),
                    'lat_max': max(lats),
                    'lon_min': min(lons),
                    'lon_max': max(lons),
                    'center_lat': sum(lats) / len(lats),
                    'center_lon': sum(lons) / len(lons)}
            
            logger.info(f"Loaded graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error loading GraphML file: {e}")
            self.coordinate_bounds = None
            logger.warning("Using default Blida coordinates as fallback")
    

    
    def get_map_center(self) -> Tuple[float, float]:
        """Get the center coordinates of the map"""
        if self.coordinate_bounds:
            return (self.coordinate_bounds['center_lat'], self.coordinate_bounds['center_lon'])
        else:
            return (36.47, 2.83)  # Default Blida center
    
    def get_map_bounds(self) -> Dict:
        """Get the coordinate bounds of the map"""
        return self.coordinate_bounds
    
    def find_nearest_nodes(self, lat: float, lon: float, count: int = 5) -> List[str]:
        """Find the nearest nodes to given coordinates"""
        if not self.nodes:
            return []
        
        def distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points"""
            import math
            R = 6371  # Earth's radius in km
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        # Calculate distances to all nodes
        distances = []
        for node_id, node_data in self.nodes.items():
            dist = distance(lat, lon, node_data['lat'], node_data['lon'])
            distances.append((dist, node_id))
        
        # Sort by distance and return top count
        distances.sort(key=lambda x: x[0])
        return [node_id for _, node_id in distances[:count]]
    

    
    def get_node_coordinates(self, node_id: str) -> Tuple[float, float]:
        """Get coordinates for a specific node"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            return (node['lat'], node['lon'])
        return None






import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from logger_setup import logger

def generate_map(city_name, output_file, image_file):
    """
    Generates a map from OpenStreetMap data for a given city and saves it as a GraphML file.
    Also, it plots the map and saves it as a PNG image.
    """
    logger.info(f"Generating map for {city_name}...")
    # Download the map data
    G = ox.graph_from_place(city_name, network_type='drive', simplify=True)

    # Save the graph to a file
    ox.save_graphml(G, filepath=output_file)
    logger.info(f"Graph for {city_name} saved to {output_file}")


    # Convert the MultiDiGraph to an undirected Graph for simpler plotting (optional)
    # This avoids plotting both directions of each edge

    plt.figure(figsize=(12, 8))
    G_simple = nx.Graph(G)  # This merges multi-edges and ignores direction
    pos = {node: (data['x'], data['y']) for node, data in G_simple.nodes(data=True)}
    nx.draw_networkx_edges(G_simple, pos, edge_color='gray', width=1)
    nx.draw_networkx_nodes(G_simple, pos, node_size=10, node_color='blue', alpha=0.7)
    plt.savefig(image_file, dpi=300, format='png')
    logger.info(f"Map image for {city_name} saved to {image_file}")


def main():
    # Define the city and output file path
    city_name = "Blida, Algeria"
    output_file = "data/Blida_map.graphml"
    image_file = "data/Blida_map.png"
    
    # Generate the map
    generate_map(city_name, output_file, image_file)

if __name__ == "__main__":
    main()
