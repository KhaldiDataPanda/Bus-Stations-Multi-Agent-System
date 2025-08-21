"""
Plotting utilities for bus routing simulation
Handles path visualization for A* routing
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger('plotting')

class SimulationPlotter:
    def __init__(self, config_path: str = 'config.json', graph_manager=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.plot_folder = Path(self.config['plotting']['plot_folder'])
        self.plot_folder.mkdir(parents=True, exist_ok=True)
        
        # Store graph manager for coordinate access
        self.graph_manager = graph_manager
        
        # Path tracking
        self.bus_step_counts = defaultdict(int)
        self.bus_paths = defaultdict(list)  # Store actual paths taken by buses
        self.bus_astar_paths = defaultdict(list)  # Store A* paths for buses
        
    def track_bus_step(self, bus_id: int, current_node: str, target_station_id: int, 
                      astar_path: List[str] = None):
        """Track a single step taken by a bus"""
        self.bus_step_counts[bus_id] += 1
        self.bus_paths[bus_id].append(current_node)
        
        # Store A* path for this route if provided
        if astar_path and f"{bus_id}_route_{target_station_id}" not in self.bus_astar_paths:
            self.bus_astar_paths[f"{bus_id}_route_{target_station_id}"] = astar_path.copy()
        
        # Check if we should plot for this bus
        if self.bus_step_counts[bus_id] % self.config['plotting']['plot_every_n_steps'] == 0:
            self.plot_bus_path(bus_id, target_station_id)
            # Also plot network graph if enabled
            if self.config['plotting'].get('save_network_graph_plots', False):
                # Try to determine current station from graph manager
                current_station_id = self._get_current_station_from_node(current_node)
                self.plot_network_graph(bus_id, target_station_id, current_station_id)
    
    def plot_bus_path(self, bus_id: int, target_station_id: int):
        """Plot the current path of a specific bus"""
        if not self.config['plotting']['save_individual_bus_plots']:
            return
        
        try:
            # Create bus-specific folder
            bus_folder = self.plot_folder / f"bus{bus_id}"
            bus_folder.mkdir(parents=True, exist_ok=True)
            
            # Get bus path and A* path
            bus_path = self.bus_paths[bus_id].copy()
            astar_key = f"{bus_id}_route_{target_station_id}"
            astar_path = self.bus_astar_paths.get(astar_key, [])
            
            if len(bus_path) < 2:
                return  # Need at least 2 nodes to plot a path
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # For now, create a simple path plot
            # In a real implementation, you'd want to use the actual graph coordinates
            self._plot_path_comparison(bus_path, astar_path, bus_id, target_station_id)
            
            # Save plot
            step_count = self.bus_step_counts[bus_id]
            filename = f"bus{bus_id}_step{step_count}_route_to_station{target_station_id}.png"
            plt.savefig(bus_folder / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved path plot for Bus {bus_id} at step {step_count}")
            
        except Exception as e:
            logger.error(f"Error plotting bus {bus_id} path: {e}")
    
    def _plot_path_comparison(self, bus_path: List[str], astar_path: List[str], 
                             bus_id: int, target_station_id: int):
        """Plot comparison between bus actual path and A* optimal path"""
        
        # Create a visualization using actual coordinates if available
        if self.graph_manager and hasattr(self.graph_manager, 'get_node_coordinates'):
            self._plot_path_with_coordinates(bus_path, astar_path, bus_id, target_station_id)
        else:
            self._plot_path_simple(bus_path, astar_path, bus_id, target_station_id)
    
    def _plot_path_with_coordinates(self, bus_path: List[str], astar_path: List[str], 
                                   bus_id: int, target_station_id: int):
        """Plot paths using actual geographic coordinates"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot A* path
            if astar_path:
                coords = []
                for node in astar_path:
                    coord = self.graph_manager.get_node_coordinates(node)
                    if coord:
                        coords.append(coord)
                
                if coords:
                    lats, lons = zip(*coords)
                    ax1.plot(lons, lats, 'b-o', linewidth=2, markersize=4, label='A* Optimal Path')
                    ax1.set_title(f'A* Optimal Path - Bus {bus_id} to Station {target_station_id}')
                    ax1.set_xlabel('Longitude')
                    ax1.set_ylabel('Latitude')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
            
            # Plot actual bus path
            if bus_path:
                coords = []
                for node in bus_path:
                    coord = self.graph_manager.get_node_coordinates(node)
                    if coord:
                        coords.append(coord)
                
                if coords:
                    lats, lons = zip(*coords)
                    ax2.plot(lons, lats, 'r-o', linewidth=2, markersize=4, label='Actual Bus Path')
                    ax2.set_title(f'Actual Path Taken - Bus {bus_id} (Step {self.bus_step_counts[bus_id]})')
                    ax2.set_xlabel('Longitude')
                    ax2.set_ylabel('Latitude')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error plotting with coordinates: {e}")
            # Fallback to simple plotting
            self._plot_path_simple(bus_path, astar_path, bus_id, target_station_id)
    
    def _plot_path_simple(self, bus_path: List[str], astar_path: List[str], 
                         bus_id: int, target_station_id: int):
        """Simple path visualization using hash-based positions"""
        
        # Create a simple visualization
        # Since we don't have actual coordinates easily accessible here,
        # we'll create a simplified representation
        
        plt.subplot(2, 1, 1)
        # Plot A* path
        if astar_path:
            x_astar = range(len(astar_path))
            y_astar = [hash(node) % 1000 for node in astar_path]  # Simple hash-based y position
            plt.plot(x_astar, y_astar, 'b-o', linewidth=2, markersize=4, label='A* Optimal Path')
        
        plt.title(f'A* Optimal Path - Bus {bus_id} to Station {target_station_id}')
        plt.xlabel('Step')
        plt.ylabel('Node Hash Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # Plot actual bus path
        x_bus = range(len(bus_path))
        y_bus = [hash(node) % 1000 for node in bus_path]  # Simple hash-based y position
        plt.plot(x_bus, y_bus, 'r-o', linewidth=2, markersize=4, label='Actual Bus Path')
        
        plt.title(f'Actual Path Taken - Bus {bus_id} (Step {self.bus_step_counts[bus_id]})')
        plt.xlabel('Step')
        plt.ylabel('Node Hash Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def plot_network_graph(self, bus_id: int, target_station_id: int, 
                          current_station_id: int = None, save_plot: bool = True):
        """
        Plot the entire network graph with:
        - Stations colored specifically
        - Target station with different color  
        - Bus path edges vs A* path edges in different colors
        - Large figure with small points for big graph
        """
        try:
            if not self.graph_manager:
                logger.error("Graph manager not available for network plotting")
                return
            
            # Get the NetworkX graph
            if hasattr(self.graph_manager, 'graph'):
                G = self.graph_manager.graph
            else:
                logger.error("NetworkX graph not accessible")
                return
            
            # Create large figure for big graph
            plt.figure(figsize=(24, 18))
            
            # Get node positions using coordinates if available
            pos = self._get_node_positions(G)
            
            # Get station information
            station_nodes = self._get_station_nodes()
            
            # Get paths
            bus_path = self.bus_paths[bus_id].copy()
            astar_key = f"{bus_id}_route_{target_station_id}"
            astar_path = self.bus_astar_paths.get(astar_key, [])
            
            # Draw the base graph with small nodes
            nx.draw_networkx_nodes(G, pos, 
                                 node_color='lightgray', 
                                 node_size=5,  # Very small nodes
                                 alpha=0.6)
            
            # Draw all edges in light gray
            nx.draw_networkx_edges(G, pos, 
                                 edge_color='lightgray', 
                                 width=0.5, 
                                 alpha=0.3)
            
            # Highlight station nodes
            if station_nodes:
                nx.draw_networkx_nodes(G, pos, 
                                     nodelist=[node for node in station_nodes if node in G.nodes()],
                                     node_color='blue', 
                                     node_size=50,  # Larger for stations
                                     alpha=0.8,
                                     label='Stations')
            
            # Highlight target station
            target_station_nodes = self._get_target_station_nodes(target_station_id)
            if target_station_nodes:
                nx.draw_networkx_nodes(G, pos, 
                                     nodelist=[node for node in target_station_nodes if node in G.nodes()],
                                     node_color='green', 
                                     node_size=100,  # Largest for target
                                     alpha=0.9,
                                     label=f'Target Station {target_station_id}')
            
            # Highlight current station if provided
            if current_station_id is not None:
                current_station_nodes = self._get_target_station_nodes(current_station_id)
                if current_station_nodes:
                    nx.draw_networkx_nodes(G, pos, 
                                         nodelist=[node for node in current_station_nodes if node in G.nodes()],
                                         node_color='orange', 
                                         node_size=80,
                                         alpha=0.9,
                                         label=f'Current Station {current_station_id}')
            
            # Draw A* path edges in blue
            if len(astar_path) > 1:
                astar_edges = [(astar_path[i], astar_path[i+1]) for i in range(len(astar_path)-1)]
                astar_edges = [(u, v) for u, v in astar_edges if G.has_edge(u, v)]
                if astar_edges:
                    nx.draw_networkx_edges(G, pos, 
                                         edgelist=astar_edges,
                                         edge_color='blue', 
                                         width=2.5, 
                                         alpha=0.8,
                                         label='A* Optimal Path')
            
            # Draw actual bus path edges in red
            if len(bus_path) > 1:
                bus_edges = [(bus_path[i], bus_path[i+1]) for i in range(len(bus_path)-1)]
                bus_edges = [(u, v) for u, v in bus_edges if G.has_edge(u, v)]
                if bus_edges:
                    nx.draw_networkx_edges(G, pos, 
                                         edgelist=bus_edges,
                                         edge_color='red', 
                                         width=2.5, 
                                         alpha=0.8,
                                         label='Bus Actual Path')
            
            # Create custom legend
            legend_elements = [
                mpatches.Patch(color='lightgray', label='Graph Nodes'),
                mpatches.Patch(color='blue', label='Stations'),
                mpatches.Patch(color='green', label=f'Target Station {target_station_id}'),
                mpatches.Patch(color='blue', label='A* Optimal Path'),
                mpatches.Patch(color='red', label='Bus Actual Path')
            ]
            
            if current_station_id is not None:
                legend_elements.insert(-2, mpatches.Patch(color='orange', label=f'Current Station {current_station_id}'))
            
            plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
            
            plt.title(f'Network Graph - Bus {bus_id} Route to Station {target_station_id}\n'
                     f'A* Path (Step {self.bus_step_counts[bus_id]})', 
                     fontsize=16, fontweight='bold')
            
            plt.axis('off')  # Remove axes for cleaner look
            
            if save_plot:
                # Save plot
                bus_folder = self.plot_folder / f"bus{bus_id}"
                bus_folder.mkdir(parents=True, exist_ok=True)
                
                step_count = self.bus_step_counts[bus_id]
                filename = f"bus{bus_id}_step{step_count}_network_graph_station{target_station_id}.png"
                plt.savefig(bus_folder / filename, dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                logger.info(f"Saved network graph for Bus {bus_id} at step {step_count}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting network graph for bus {bus_id}: {e}")
    
    def _get_node_positions(self, G):
        """Get node positions using coordinates if available, otherwise use spring layout"""
        pos = {}
        
        if self.graph_manager and hasattr(self.graph_manager, 'node_coordinates'):
            # Use actual coordinates
            for node in G.nodes():
                if node in self.graph_manager.node_coordinates:
                    coord = self.graph_manager.node_coordinates[node]
                    pos[node] = (coord[1], coord[0])  # (lon, lat) for matplotlib
        
        # If no coordinates available or incomplete, use spring layout
        if len(pos) < len(G.nodes()) * 0.8:  # If less than 80% have coordinates
            logger.info("Using spring layout for network graph positioning")
            pos = nx.spring_layout(G, k=1, iterations=50)
        
        return pos
    
    def _get_station_nodes(self):
        """Get all station nodes from the graph manager"""
        station_nodes = []
        if self.graph_manager and hasattr(self.graph_manager, 'station_to_nodes'):
            for station_data in self.graph_manager.station_to_nodes.values():
                if isinstance(station_data, list):
                    station_nodes.extend(station_data)
                else:
                    station_nodes.append(station_data)
        return station_nodes
    
    def _get_target_station_nodes(self, station_id: int):
        """Get nodes for a specific station"""
        if self.graph_manager and hasattr(self.graph_manager, 'station_to_nodes'):
            return self.graph_manager.station_to_nodes.get(station_id, [])
        return []
    
    def _get_current_station_from_node(self, current_node: str):
        """Try to determine which station a node belongs to (if any)"""
        if self.graph_manager and hasattr(self.graph_manager, 'station_to_nodes'):
            for station_id, nodes in self.graph_manager.station_to_nodes.items():
                if current_node in nodes:
                    return station_id
        return None

# Global plotter instance
plotter = None

def initialize_plotter(config_path: str = 'config.json', graph_manager=None):
    """Initialize the global plotter instance"""
    global plotter
    plotter = SimulationPlotter(config_path, graph_manager)
    return plotter

def get_plotter() -> Optional[SimulationPlotter]:
    """Get the global plotter instance"""
    return plotter
