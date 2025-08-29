"""
Enhanced Dashboard for Traffic Routing with A*
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
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from simulation.bus_lines_manager import BusLinesManager, Station, BusLine
from simulation.graph_loader import GraphLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Init 

graph_loader = GraphLoader('data/Blida_map.graphml')


st.set_page_config(
    page_title="Traffic Routing Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded" )


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
            print(f"API {endpoint} error: {response.status_code}")
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




def create_base_map(show_nodes=False):
    """Create base map for line creation"""

    center_lat, center_lon = graph_loader.get_map_center()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='cartodbpositron')


    if show_nodes:    # Add graph nodes to map
        for node, data in graph_loader.nodes.items():
            folium.CircleMarker(
                location=(data['lat'], data['lon']),  
                radius=0.05,
                color='blue',
                fill=False,
            ).add_to(m)


    
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
            popup=f"Station {i}",
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
        zoom_start=15,
        tiles='cartodbpositron')
    
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

    

    cmap = plt.cm.tab20  # tab20 provides 20 distinct colors
    bus_colors = [mcolors.to_hex(cmap(i)) for i in range(20)]
    
    # Add buses with X markers and diverse colors
    if bus_data:
        for i, bus in enumerate(bus_data):
            # Skip buses with invalid coordinates
            if (bus.get('lat') is None or bus.get('lon') is None or 
                bus.get('lat') == 0 and bus.get('lon') == 0):
                continue
                
            # Assign unique color to each bus
            bus_color = bus_colors[i % len(bus_colors)]
            
            # Create custom X marker HTML
            x_html = f'''
            <div style="text-align: center; color: {bus_color}; font-size: 20px; font-weight: bold; 
                        text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white;">
                ✕
            </div>
            '''
            
            # Robust bus label
            raw_id = bus.get('id')
            try:
                bus_label = str(int(raw_id) + 1)
            except Exception:
                bus_label = str(raw_id)
            
            # Additional safety check for coordinates
            try:
                lat = float(bus.get('lat', 0))
                lon = float(bus.get('lon', 0))
            except (TypeError, ValueError):
                # If coordinates can't be converted to float, skip this bus
                continue
            
            # Add bus as X marker
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>Bus {bus_label}</b><br>"
                      f"Passengers: {bus.get('passenger_count', 0)}<br>"
                      f"Status: {bus.get('status', 'Unknown')}<br>"
                      f"Current: {bus.get('current_location', 'Unknown')}<br>"
                      f"Destination: {bus.get('destination', 'Unknown')}<br>"
                      f"Routing: A* Algorithm",
                icon=folium.DivIcon(
                    html=x_html,
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(m)
            
            # Add dynamic routing path with bus color (only current route segment)
            if 'path' in bus and bus['path'] and len(bus['path']) > 1:
                # Convert path coordinates to proper format if needed
                path_coords = []
                for coord in bus['path']:
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        # Handle both [lat, lon] and (lat, lon) formats
                        lat, lon = coord[0], coord[1]
                        if lat != 0 or lon != 0:  # Skip invalid coordinates
                            path_coords.append([lat, lon])
                
                if len(path_coords) > 1:
                    folium.PolyLine(
                        path_coords,
                        color=bus_color,
                        weight=4,
                        opacity=0.8,
                        popup=f"Bus {bus_label} Route"
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


############################################################################################################
############################################################################################################
#-----------------------------           Bus Creation Page         -----------------------------------------
############################################################################################################
############################################################################################################


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
            help="Choose how to create your line: click on map or enter coordinates manually")
        
        if creation_method == "Map Clicking":
            # Instructions for map clicking
            st.info("Click on the map to add stations to your line. You need at least 2 stations.")
            
            # Show selected points
            if st.session_state.selected_points:
                st.subheader("Selected Stations")
                for i, point in enumerate(st.session_state.selected_points):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"Station {i}: ({point['lat']:.4f}, {point['lng']:.4f})")
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
                        f"Station {i} (lat, lon):", 
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
                            st.error(f"Station {i}: Invalid format. Use 'latitude, longitude'")
                    except ValueError:
                        st.error(f"Station {i}: Invalid numbers. Use decimal format (e.g., 36.4735, 2.8311)")
            
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
                                f"Station {i} coordinates:", 
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
                                f"New Station {i} coordinates:", 
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
    





    #########################################################################################
    with col1:
        st.subheader("Map")

        st.radio('Show Graph Nodes:', ['Yes','No'], index=1, key='show_nodes')

        if st.session_state.show_nodes == 'Yes':
            show_nodes = True
        else:
            show_nodes = False
        
        # Create map
        m = create_base_map(show_nodes=show_nodes)
        
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



############################################################################################################
############################################################################################################
#-----------------------------  Traffic Routing Simulation Page    -----------------------------------------
############################################################################################################
############################################################################################################



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
            st.write("✕ X Markers = Active buses (each has unique color)")
            st.write("✕ Color-coded routes show bus paths")
            st.write("")
            st.write("**📍 Stations:**")
            st.write("🔴 Red Circle = Terminal Station")
            st.write("🔵 Blue Circle = Regular Station")
        with col2:
            st.write("**🛤️ Routes:**")
            st.write("━━━ Colored Lines = Current bus routes")
            st.write("(Each bus has its own color)")
            st.write("")
            st.write("**⚠️ Incidents:**")
            st.write("🔴 Red Circle = Traffic Incident")
        
        st.info("💡 **Note**: Each bus appears as a colored X marker with its route highlighted in the same color. Only active routes are shown.")
    
    if st.session_state.simulation_running:
        simulation_map = create_simulation_map()
        st_folium(simulation_map, width=1000, height=600, returned_objects=[])
        
        # Debug information for bus tracking
        bus_data = get_api_data("buses")
        if bus_data:
            st.subheader("🐛 Debug: Active Buses")
            with st.expander("View bus data", expanded=False):
                for i, bus in enumerate(bus_data):
                    raw_id = bus.get('id')
                    try:
                        bus_label = str(int(raw_id) + 1)
                    except Exception:
                        bus_label = str(raw_id)
                    st.write(f"**Bus {bus_label}:**")
                    st.write(f"- Position: ({bus.get('lat', 0):.6f}, {bus.get('lon', 0):.6f})")
                    st.write(f"- Status: {bus.get('status', 'Unknown')}")
                    st.write(f"- Passengers: {bus.get('passenger_count', 0)}")
                    st.write(f"- Current Location: {bus.get('current_location', 'Unknown')}")
                    st.write(f"- Destination: {bus.get('destination', 'Unknown')}")
                    if 'path' in bus and bus['path']:
                        st.write(f"- Path points: {len(bus['path'])}")
                    st.write("---")
        else:
            st.warning("No bus data received from API. Check if the simulation is running and buses are active.")
        
        # Auto refresh every 10 seconds
        time.sleep(10)
        st.rerun()
    else:
        simulation_map = create_simulation_map()
        st_folium(simulation_map, width=1000, height=600, returned_objects=[])




############################################################################################################
############################################################################################################
#-----------------------------                Other Pages          -----------------------------------------
############################################################################################################
############################################################################################################



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
            util_hist = metrics_data.get('bus_utilization_history', []) or []
            if isinstance(util_hist, list) and len(util_hist) > 0:
                df_util = pd.DataFrame(util_hist)
                if {'time', 'utilization'}.issubset(df_util.columns):
                    fig = px.line(df_util, x='time', y='utilization',
                                title='Bus Utilization Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Bus utilization data not available")
            else:
                st.info("Bus utilization data not available")
        
        with col2:
            # Passenger flow
            flow_hist = metrics_data.get('passenger_flow_history', []) or []
            if isinstance(flow_hist, list) and len(flow_hist) > 0:
                df_flow = pd.DataFrame(flow_hist)
                if {'time', 'passengers'}.issubset(df_flow.columns):
                    fig = px.line(df_flow, x='time', y='passengers',
                                title='Passenger Flow Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Passenger flow data not available")
            else:
                st.info("Passenger flow data not available")
    else:
        st.warning("No system metrics available. Make sure the simulation is running.")
    
    # Auto refresh after 20 seconds
    time.sleep(20)
    st.rerun()


def astar_metrics_page():
    """A* Algorithm performance metrics with 20s refresh"""
    st.title("🎯 A* Algorithm Performance")
    
    # Use metrics combined endpoint
    metrics = get_api_data("metrics")
    buses = get_api_data("buses")
    
    if metrics is None and buses is None:
        st.warning("Performance metrics not available")
        time.sleep(20)
        st.rerun()
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Steps",
            f"{(metrics or {}).get('avg_steps', 0):.1f}",
            delta=f"{(metrics or {}).get('steps_delta', 0):.1f}"
        )
    with col2:
        st.metric(
            "Success Rate",
            f"{(metrics or {}).get('success_rate', 100):.1f}%",
            delta=f"{(metrics or {}).get('success_delta', 0):.1f}%"
        )
    with col3:
        total_routes = (metrics or {}).get('total_routes')
        if total_routes is None and isinstance(buses, list):
            total_routes = len(buses)
        st.metric(
            "Total Routes",
            total_routes or 0,
            delta=(metrics or {}).get('routes_delta', 0)
        )
    
    # Optional detail table of current routes
    if isinstance(buses, list) and len(buses) > 0:
        st.subheader("Current Bus Routes (Sample)")
        table_rows = []
        for b in buses[:20]:
            table_rows.append({
                'bus_id': b.get('id'),
                'status': b.get('status'),
                'passengers': b.get('passenger_count'),
                'current': b.get('current_location'),
                'destination': b.get('destination'),
                'steps_taken': b.get('steps_taken', 0)
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    
    # Auto refresh after 20 seconds
    time.sleep(20)
    st.rerun()


def traffic_metrics_page():
    """Traffic and incident metrics with 20s refresh"""
    st.title("🚦 Traffic & Incident Metrics")
    
    traffic_data = get_api_data("traffic_metrics")
    incident_data = get_api_data("incidents")
    buses = get_api_data("buses")
    
    # Derive simple traffic indicators from current buses if available
    derived_avg_speed = None

    if isinstance(buses, list) and len(buses) > 0:
        
        active_buses = len(buses)
        # Placeholder derivations; real logic can use distances and timestamps
        derived_network_util = min(100.0, active_buses * 5.0)
    
    if traffic_data or incident_data or buses:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Incidents",
                len(incident_data) if incident_data else 0
            )
        
        with col2:
            st.metric(
                "Average Speed",
                f"{(traffic_data or {}).get('avg_speed', derived_avg_speed or 0):.1f} km/h"
            )
        
        with col3:
            st.metric(
                "Network Efficiency",
                f"{(traffic_data or {}).get('efficiency', 0):.1f}%"
            )
        
        with col4:
            st.metric(
                "Congestion Level",
                (traffic_data or {}).get('congestion_level', 'Low')
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



############################################################################################################
############################################################################################################
#---------------------------------              Main Page          -----------------------------------------
############################################################################################################
############################################################################################################


def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🚌 Navigation")
        
        page = st.radio(
            "Select Page",
            ["Line Creation", "Simulation Control", "System Metrics", "A* Performance", "Traffic Analysis"],
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
    elif page == "A* Performance":
        astar_metrics_page()
    elif page == "Traffic Analysis":
        traffic_metrics_page()

if __name__ == "__main__":
    main()
