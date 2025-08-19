"""
Enhanced Dashboard for Traffic Routing with RL
Features map-based line creation and real-time visualization
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from datetime import datetime, timedelta
import numpy as np
from bus_lines_manager import BusLinesManager, Station, BusLine
from graph_loader import GraphLoader
import logging

# Configure logging for dashboard UI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DASHBOARD-UI] - %(message)s')
logger = logging.getLogger(__name__)

# Initialize graph loader
graph_loader = GraphLoader('data/Blida_map.graphml')

# Page configuration
st.set_page_config(
    page_title="Traffic Routing Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'bus_lines_manager' not in st.session_state:
    st.session_state.bus_lines_manager = BusLinesManager()
    st.session_state.bus_lines_manager.load_from_file('data/bus_lines.json')

if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []

if 'current_line_name' not in st.session_state:
    st.session_state.current_line_name = ""

# API base URL
API_BASE_URL = "http://localhost:8000"

def get_api_data(endpoint: str):
    """Get data from API endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def start_simulation():
    """Start the main simulation"""
    try:
        # First check if API server is reachable
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_response.status_code != 200:
            st.error("⚠️ API Server is not running. Please start the API server first by running: `python api_server.py`")
            return
        
        response = requests.post(f"{API_BASE_URL}/start_simulation", timeout=10)
        if response.status_code == 200:
            st.session_state.simulation_running = True
            st.success("✅ Simulation started successfully!")
            st.info("🚌 Buses will launch automatically on your created lines")
        elif response.status_code == 400:
            error_data = response.json()
            st.error(f"❌ {error_data.get('detail', 'Error starting simulation')}")
        else:
            st.error(f"❌ Failed to start simulation (Status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to simulation server. Please ensure:")
        st.write("1. The API server is running (`python api_server.py`)")
        st.write("2. The API server is accessible at http://localhost:8000")
        st.write("3. Use the launcher: `python launcher.py` for automatic setup")
    except requests.exceptions.Timeout:
        st.error("⏱️ Connection timeout. The simulation server might be busy.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

def stop_simulation():
    """Stop the simulation"""
    try:
        response = requests.post(f"{API_BASE_URL}/stop_simulation", timeout=10)
        if response.status_code == 200:
            st.session_state.simulation_running = False
            st.success("✅ Simulation stopped")
        else:
            st.error("❌ Failed to stop simulation")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Could not connect to API server to stop simulation")
        st.session_state.simulation_running = False
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error stopping simulation: {e}")

def create_base_map():
    """Create base map for line creation"""
    # Use actual Blida coordinates from the graph
    center_lat, center_lon = graph_loader.get_map_center()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add existing stations to map
    for station in st.session_state.bus_lines_manager.get_all_stations().values():
        color = 'red' if station.is_terminal else 'blue'
        folium.CircleMarker(
            location=[station.lat, station.lon],
            radius=8,
            popup=f"{station.name} (Line {station.line_id})",
            color=color,
            fill=True,
            fillColor=color
        ).add_to(m)
    
    # Add existing lines (only for line creation/editing context)
    for line in st.session_state.bus_lines_manager.get_all_lines().values():
        if len(line.stations) >= 2:
            coords = [[station.lat, station.lon] for station in line.stations]
            folium.PolyLine(
                coords,
                color='gray',
                weight=2,
                opacity=0.5,
                popup=f"Line: {line.name}",
                dash_array='10, 10'  # Dashed line to show it's just reference
            ).add_to(m)
    
    # Add selected points for new line creation
    for i, point in enumerate(st.session_state.selected_points):
        folium.CircleMarker(
            location=[point['lat'], point['lng']],
            radius=6,
            popup=f"Station {i+1}",
            color='orange',
            fill=True,
            fillColor='orange'
        ).add_to(m)
    
    return m

def create_simulation_map():
    """Create map showing real-time simulation"""
    # Get bus positions from API
    bus_data = get_api_data("buses")
    incident_data = get_api_data("incidents")
    
    # Use actual Blida coordinates from the graph
    center_lat, center_lon = graph_loader.get_map_center()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add stations only (no static lines)
    for station in st.session_state.bus_lines_manager.get_all_stations().values():
        color = 'red' if station.is_terminal else 'blue'
        folium.CircleMarker(
            location=[station.lat, station.lon],
            radius=8,
            popup=f"{station.name}",
            color=color,
            fill=True,
            fillColor=color
        ).add_to(m)
    
    # Add buses with dynamic routing paths only
    if bus_data:
        for bus in bus_data:
            # Determine bus icon color based on routing type
            icon_color = 'green' if bus.get('using_rl', True) else 'blue'
            routing_type = "RL Agent" if bus.get('using_rl', True) else "A* Algorithm"
            
            folium.Marker(
                location=[bus['lat'], bus['lon']],
                popup=f"Bus {bus['id'] + 1}<br>Passengers: {bus['passenger_count']}<br>Status: {bus['status']}<br>Routing: {routing_type}",
                icon=folium.Icon(color=icon_color, icon='bus', prefix='fa')
            ).add_to(m)
            
            # Add dynamic routing path if available (only when bus is moving)
            if 'path' in bus and bus['path'] and len(bus['path']) > 1:
                path_color = 'red' if bus.get('using_rl', True) else 'blue'
                path_style = '5, 5' if bus.get('using_rl', True) else '10, 5'  # Different dash patterns
                
                folium.PolyLine(
                    bus['path'],
                    color=path_color,
                    weight=3,
                    opacity=0.8,
                    dash_array=path_style,
                    popup=f"Bus {bus['id'] + 1} - {routing_type} Route"
                ).add_to(m)
    
    # Add incidents
    if incident_data:
        for incident in incident_data:
            folium.CircleMarker(
                location=[incident['lat'], incident['lon']],
                radius=15,
                popup=f"Incident: {incident['type']}<br>Duration: {incident['duration']}h",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)
    
    return m

def line_creation_page():
    """Page for creating bus lines"""
    st.title("🗺️ Bus Line Creation")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Create New Line")
        
        # Line name input
        line_name = st.text_input("Line Name", value=st.session_state.current_line_name)
        st.session_state.current_line_name = line_name
        
        # Method selection
        creation_method = st.radio(
            "Creation Method:",
            ["Map Clicking", "Coordinate Input"],
            help="Choose how to create your line: click on map or enter coordinates manually"
        )
        
        if creation_method == "Map Clicking":
            # Instructions for map clicking
            st.info("Click on the map to add stations to your line. You need at least 2 stations.")
            
            # Show selected points
            if st.session_state.selected_points:
                st.subheader("Selected Stations")
                for i, point in enumerate(st.session_state.selected_points):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"Station {i+1}: ({point['lat']:.4f}, {point['lng']:.4f})")
                    with col_b:
                        if st.button("❌", key=f"remove_{i}", help="Remove this station"):
                            st.session_state.selected_points.pop(i)
                            st.rerun()
            
            # Clear points button
            if st.button("Clear All Points"):
                st.session_state.selected_points = []
                st.rerun()
        
        else:  # Coordinate Input
            st.info("Enter coordinates manually. Use Latitude, Longitude format (e.g., 36.4735, 2.8311)")
            
            # Initialize coordinate list in session state if not exists
            if 'coordinate_inputs' not in st.session_state:
                st.session_state.coordinate_inputs = ["", ""]
            
            st.subheader("Station Coordinates")
            
            # Display current coordinate inputs
            for i in range(len(st.session_state.coordinate_inputs)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    coord_input = st.text_input(
                        f"Station {i+1} (lat, lon):", 
                        value=st.session_state.coordinate_inputs[i],
                        key=f"coord_{i}",
                        placeholder="36.4735, 2.8311"
                    )
                    st.session_state.coordinate_inputs[i] = coord_input
                with col_b:
                    if len(st.session_state.coordinate_inputs) > 2:
                        if st.button("❌", key=f"remove_coord_{i}", help="Remove this station"):
                            st.session_state.coordinate_inputs.pop(i)
                            st.rerun()
            
            # Add more coordinate inputs
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("➕ Add Station"):
                    st.session_state.coordinate_inputs.append("")
                    st.rerun()
            
            with col_b:
                if st.button("Clear All"):
                    st.session_state.coordinate_inputs = ["", ""]
                    st.rerun()
            
            # Parse and validate coordinates
            valid_coordinates = []
            for i, coord_str in enumerate(st.session_state.coordinate_inputs):
                if coord_str.strip():
                    try:
                        parts = coord_str.split(',')
                        if len(parts) == 2:
                            lat = float(parts[0].strip())
                            lon = float(parts[1].strip())
                            valid_coordinates.append({'lat': lat, 'lng': lon})
                        else:
                            st.error(f"Station {i+1}: Invalid format. Use 'latitude, longitude'")
                    except ValueError:
                        st.error(f"Station {i+1}: Invalid numbers. Use decimal format (e.g., 36.4735, 2.8311)")
            
            # Update selected points for compatibility with map method
            if creation_method == "Coordinate Input":
                st.session_state.selected_points = valid_coordinates
        
        # Create line button
        create_disabled = (
            len(st.session_state.selected_points) < 2 or 
            not line_name or 
            (creation_method == "Coordinate Input" and len(valid_coordinates) < 2)
        )
        
        if st.button("Create Line", disabled=create_disabled):
            try:
                coordinates = [(point['lat'], point['lng']) for point in st.session_state.selected_points]
                line_id = st.session_state.bus_lines_manager.create_line(line_name, coordinates)
                st.session_state.bus_lines_manager.save_to_file('data/bus_lines.json')
                st.success(f"Line '{line_name}' created successfully!")
                st.session_state.selected_points = []
                st.session_state.current_line_name = ""
                if 'coordinate_inputs' in st.session_state:
                    st.session_state.coordinate_inputs = ["", ""]
                st.rerun()
            except Exception as e:
                st.error(f"Error creating line: {e}")
        
        # Show existing lines with delete and edit functionality
        st.subheader("Existing Lines")
        lines = st.session_state.bus_lines_manager.get_all_lines()
        
        if lines:
            for line_id, line in lines.items():
                with st.expander(f"🚌 {line.name} ({len(line.stations)} stations)", expanded=False):
                    # Show stations
                    st.write("**Stations:**")
                    for i, station in enumerate(line.stations):
                        icon = "🔴" if station.is_terminal else "🔵"
                        st.write(f"{icon} {station.name} {'(Terminal)' if station.is_terminal else ''}")
                        st.write(f"   📍 Coordinates: {station.lat:.4f}, {station.lon:.4f}")
                    
                    # Line management buttons
                    col_del, col_edit, col_info = st.columns(3)
                    
                    with col_del:
                        if st.button(f"🗑️ Delete Line", key=f"delete_line_{line_id}", type="secondary"):
                            if st.session_state.get(f'confirm_delete_{line_id}', False):
                                # Actually delete the line
                                st.session_state.bus_lines_manager.delete_line(line_id)
                                st.session_state.bus_lines_manager.save_to_file('data/bus_lines.json')
                                st.success(f"Line '{line.name}' deleted successfully!")
                                if f'confirm_delete_{line_id}' in st.session_state:
                                    del st.session_state[f'confirm_delete_{line_id}']
                                st.rerun()
                            else:
                                # Show confirmation
                                st.session_state[f'confirm_delete_{line_id}'] = True
                                st.warning(f"⚠️ Click 'Delete Line' again to confirm deletion of '{line.name}'")
                                st.rerun()
                    
                    with col_edit:
                        if st.button(f"✏️ Edit Line", key=f"edit_line_{line_id}", type="primary"):
                            # Enable edit mode for this line
                            st.session_state[f'edit_mode_{line_id}'] = True
                            st.session_state.current_line_name = line.name
                            # Populate coordinates for editing
                            coords = []
                            for station in line.stations:
                                coords.append(f"{station.lat}, {station.lon}")
                            st.session_state.coordinate_inputs = coords
                            st.rerun()
                    
                    with col_info:
                        st.metric("Reserve Buses", line.reserve_buses)
                    
                    # Edit mode interface
                    if st.session_state.get(f'edit_mode_{line_id}', False):
                        st.divider()
                        st.write("**Edit Mode - Modify stations:**")
                        
                        # Edit line name
                        new_name = st.text_input("New Line Name:", value=line.name, key=f"edit_name_{line_id}")
                        
                        # Edit coordinates
                        edited_coords = []
                        for i, station in enumerate(line.stations):
                            coord_str = st.text_input(
                                f"Station {i+1} coordinates:", 
                                value=f"{station.lat}, {station.lon}",
                                key=f"edit_coord_{line_id}_{i}"
                            )
                            edited_coords.append(coord_str)
                        
                        # Add new station
                        if st.button(f"➕ Add Station", key=f"add_station_{line_id}"):
                            st.session_state[f'new_stations_{line_id}'] = st.session_state.get(f'new_stations_{line_id}', 0) + 1
                            st.rerun()
                        
                        # Show inputs for new stations
                        new_station_count = st.session_state.get(f'new_stations_{line_id}', 0)
                        for i in range(new_station_count):
                            new_coord = st.text_input(
                                f"New Station {i+1} coordinates:", 
                                placeholder="36.4735, 2.8311",
                                key=f"new_coord_{line_id}_{i}"
                            )
                            edited_coords.append(new_coord)
                        
                        # Save/Cancel buttons
                        col_save, col_cancel = st.columns(2)
                        
                        with col_save:
                            if st.button(f"💾 Save Changes", key=f"save_edit_{line_id}", type="primary"):
                                try:
                                    # Parse coordinates
                                    valid_coordinates = []
                                    for coord_str in edited_coords:
                                        if coord_str.strip():
                                            parts = coord_str.split(',')
                                            if len(parts) == 2:
                                                lat = float(parts[0].strip())
                                                lon = float(parts[1].strip())
                                                valid_coordinates.append((lat, lon))
                                    
                                    if len(valid_coordinates) >= 2:
                                        # Delete old line and create new one
                                        st.session_state.bus_lines_manager.delete_line(line_id)
                                        new_line_id = st.session_state.bus_lines_manager.create_line(new_name, valid_coordinates)
                                        st.session_state.bus_lines_manager.save_to_file('data/bus_lines.json')
                                        
                                        # Clear edit state
                                        st.session_state[f'edit_mode_{line_id}'] = False
                                        if f'new_stations_{line_id}' in st.session_state:
                                            del st.session_state[f'new_stations_{line_id}']
                                        
                                        st.success(f"Line '{new_name}' updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Need at least 2 valid coordinates")
                                        
                                except Exception as e:
                                    st.error(f"Error updating line: {e}")
                        
                        with col_cancel:
                            if st.button(f"❌ Cancel Edit", key=f"cancel_edit_{line_id}"):
                                st.session_state[f'edit_mode_{line_id}'] = False
                                if f'new_stations_{line_id}' in st.session_state:
                                    del st.session_state[f'new_stations_{line_id}']
                                st.rerun()
        else:
            st.info("No lines created yet. Create your first line above!")
            st.write("**🔹 Quick Start:**")
            st.write("1. Enter a line name")
            st.write("2. Click on the map to add stations (minimum 2)")
            st.write("3. Click 'Create Line' to save")
            st.write("4. Go to Simulation Control to start the simulation")
    
    with col1:
        st.subheader("Map")
        
        # Create map
        m = create_base_map()
        
        # Display map and capture clicks
        map_data = st_folium(
            m,
            width=700,
            height=500,
            returned_objects=["last_object_clicked"]
        )
        
        # Handle map clicks
        if map_data['last_object_clicked']:
            clicked_data = map_data['last_object_clicked']
            if 'lat' in clicked_data and 'lng' in clicked_data:
                new_point = {
                    'lat': clicked_data['lat'],
                    'lng': clicked_data['lng']
                }
                if new_point not in st.session_state.selected_points:
                    st.session_state.selected_points.append(new_point)
                    st.rerun()

def simulation_control_page():
    """Main simulation control page"""
    st.title("🚌 Traffic Routing Simulation")
    
    # Check if lines exist
    lines = st.session_state.bus_lines_manager.get_all_lines()
    if not lines:
        st.warning("No bus lines created yet. Please create bus lines first.")
        return
    
    # Check API server connection
    try:
        api_status = requests.get(f"{API_BASE_URL}/health", timeout=2)
        api_connected = api_status.status_code == 200
    except:
        api_connected = False
    
    if not api_connected:
        st.error("⚠️ **API Server not connected!**")
        st.write("To start the simulation, you need to run the API server:")
        st.code("python api_server.py", language="bash")
        st.write("Or use the launcher for automatic setup:")
        st.code("python launcher.py", language="bash")
        st.write("Then select option 4 for the full system.")
        return
    
    # Simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.simulation_running:
            if st.button("🚀 Start Simulation", type="primary"):
                start_simulation()
        else:
            if st.button("⏹️ Stop Simulation", type="secondary"):
                stop_simulation()
    
    with col2:
        st.metric("Simulation Status", 
                 "Running" if st.session_state.simulation_running else "Stopped",
                 "🟢" if st.session_state.simulation_running else "🔴")
    
    with col3:
        st.metric("Bus Lines", len(lines))
    
    # Real-time map
    st.subheader("🗺️ Real-time Bus Tracking")
    
    # Add map legend
    with st.expander("🗺️ Map Legend", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🚌 Buses:**")
            st.write("🟢 Green Bus = RL Agent Routing")
            st.write("🔵 Blue Bus = A* Algorithm Routing")
            st.write("")
            st.write("**📍 Stations:**")
            st.write("🔴 Red Circle = Terminal Station")
            st.write("🔵 Blue Circle = Regular Station")
        with col2:
            st.write("**🛤️ Routes:**")
            st.write("━━━ Red Dashed = RL Agent Path")
            st.write("━━━ Blue Dashed = A* Algorithm Path")
            st.write("")
            st.write("**⚠️ Incidents:**")
            st.write("🔴 Red Circle = Traffic Incident")
        
        st.info("💡 **Note**: Only active routing paths are shown - no static line routes")
    
    if st.session_state.simulation_running:
        simulation_map = create_simulation_map()
        st_folium(simulation_map, width=1000, height=600, returned_objects=[])
        
        # Auto refresh every 10 seconds
        time.sleep(10)
        st.rerun()
    else:
        simulation_map = create_simulation_map()
        st_folium(simulation_map, width=1000, height=600, returned_objects=[])

def system_metrics_page():
    """System performance metrics with 20s refresh"""
    st.title("📊 System Performance Metrics")
    
    # Get metrics data
    metrics_data = get_api_data("metrics")
    
    if metrics_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Buses",
                metrics_data.get('active_buses', 0),
                delta=metrics_data.get('buses_delta', 0)
            )
        
        with col2:
            st.metric(
                "Total Passengers",
                metrics_data.get('total_passengers', 0),
                delta=metrics_data.get('passengers_delta', 0)
            )
        
        with col3:
            st.metric(
                "Average Travel Time",
                f"{metrics_data.get('avg_travel_time', 0):.1f}h",
                delta=f"{metrics_data.get('travel_time_delta', 0):.1f}h"
            )
        
        with col4:
            st.metric(
                "Active Incidents",
                metrics_data.get('active_incidents', 0),
                delta=metrics_data.get('incidents_delta', 0)
            )
        
        # Charts
        st.subheader("Performance Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bus utilization over time
            if 'bus_utilization_history' in metrics_data:
                df_util = pd.DataFrame(metrics_data['bus_utilization_history'])
                fig = px.line(df_util, x='time', y='utilization',
                            title='Bus Utilization Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Bus utilization data not available")
        
        with col2:
            # Passenger flow
            if 'passenger_flow_history' in metrics_data:
                df_flow = pd.DataFrame(metrics_data['passenger_flow_history'])
                fig = px.line(df_flow, x='time', y='passengers',
                            title='Passenger Flow Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Passenger flow data not available")
    else:
        st.warning("No system metrics available. Make sure the simulation is running.")
    
    # Auto refresh after 20 seconds
    time.sleep(20)
    st.rerun()

def rl_metrics_page():
    """RL Agent performance metrics with 20s refresh"""
    st.title("🤖 RL Agent Performance")
    
    rl_metrics = get_api_data("rl_metrics")
    
    if rl_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Reward",
                f"{rl_metrics.get('avg_reward', 0):.2f}",
                delta=f"{rl_metrics.get('reward_delta', 0):.2f}"
            )
        
        with col2:
            st.metric(
                "Success Rate",
                f"{rl_metrics.get('success_rate', 0):.1f}%",
                delta=f"{rl_metrics.get('success_delta', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Training Episodes",
                rl_metrics.get('total_episodes', 0),
                delta=rl_metrics.get('episodes_delta', 0)
            )
        
        # RL performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'reward_history' in rl_metrics:
                df_rewards = pd.DataFrame(rl_metrics['reward_history'])
                fig = px.line(df_rewards, x='episode', y='reward',
                            title='RL Agent Reward Over Episodes')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Reward history not available")
        
        with col2:
            if 'loss_history' in rl_metrics:
                df_loss = pd.DataFrame(rl_metrics['loss_history'])
                fig = px.line(df_loss, x='step', y='loss',
                            title='Training Loss Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Loss history not available")
        
        # Additional RL insights
        st.subheader("RL Agent Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Performance")
            if 'recent_episodes' in rl_metrics:
                df_recent = pd.DataFrame(rl_metrics['recent_episodes'])
                st.dataframe(df_recent, use_container_width=True)
            else:
                st.info("Recent episode data not available")
        
        with col2:
            st.subheader("Model Statistics")
            st.write(f"Model Parameters: {rl_metrics.get('model_params', 'N/A')}")
            st.write(f"Training Steps: {rl_metrics.get('training_steps', 'N/A')}")
            st.write(f"Exploration Rate: {rl_metrics.get('exploration_rate', 'N/A'):.3f}")
    else:
        st.warning("No RL metrics available. Make sure the simulation is running.")
    
    # Auto refresh after 20 seconds
    time.sleep(20)
    st.rerun()

def traffic_metrics_page():
    """Traffic and incident metrics with 20s refresh"""
    st.title("🚦 Traffic & Incident Metrics")
    
    traffic_data = get_api_data("traffic_metrics")
    incident_data = get_api_data("incidents")
    
    if traffic_data or incident_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Incidents",
                len(incident_data) if incident_data else 0
            )
        
        with col2:
            if traffic_data:
                st.metric(
                    "Average Speed",
                    f"{traffic_data.get('avg_speed', 0):.1f} km/h"
                )
        
        with col3:
            if traffic_data:
                st.metric(
                    "Network Efficiency",
                    f"{traffic_data.get('efficiency', 0):.1f}%"
                )
        
        with col4:
            if traffic_data:
                st.metric(
                    "Congestion Level",
                    traffic_data.get('congestion_level', 'Low')
                )
        
        # Incident breakdown
        if incident_data:
            st.subheader("Active Incidents")
            incident_df = pd.DataFrame(incident_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'type' in incident_df.columns:
                    incident_counts = incident_df['type'].value_counts()
                    fig = px.pie(values=incident_counts.values, names=incident_counts.index,
                               title='Incident Types Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'duration' in incident_df.columns:
                    fig = px.histogram(incident_df, x='duration',
                                     title='Incident Duration Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Incident table
            st.subheader("Incident Details")
            st.dataframe(incident_df, use_container_width=True)
        else:
            st.info("No active incidents")
    else:
        st.warning("No traffic data available. Make sure the simulation is running.")
    
    # Auto refresh after 20 seconds
    time.sleep(20)
    st.rerun()

def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🚌 Navigation")
        
        page = st.radio(
            "Select Page",
            ["Line Creation", "Simulation Control", "System Metrics", "RL Performance", "Traffic Analysis"],
            index=1 if st.session_state.get('simulation_running', False) else 0
        )
        
        st.divider()
        
        # System info
        st.subheader("System Info")
        lines_count = len(st.session_state.bus_lines_manager.get_all_lines())
        stations_count = len(st.session_state.bus_lines_manager.get_all_stations())
        
        st.metric("Bus Lines", lines_count)
        st.metric("Stations", stations_count)
        
        if st.session_state.simulation_running:
            st.success("Simulation Running")
        else:
            st.error("Simulation Stopped")
    
    # Route to appropriate page
    if page == "Line Creation":
        line_creation_page()
    elif page == "Simulation Control":
        simulation_control_page()
    elif page == "System Metrics":
        system_metrics_page()
    elif page == "RL Performance":
        rl_metrics_page()
    elif page == "Traffic Analysis":
        traffic_metrics_page()

if __name__ == "__main__":
    main()
