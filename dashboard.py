import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import io
import numpy as np
import random
from datetime import datetime
import folium
from streamlit_folium import st_folium
import osmnx as ox

# Configure page
st.set_page_config(
    page_title="🚌 Traffic Routing System",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api"

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-running { color: #28a745; font-weight: bold; }
    .status-stopped { color: #dc3545; font-weight: bold; }
    .status-waiting { color: #ffc107; font-weight: bold; }
    .big-button {
        font-size: 1.2rem !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
    }
    .simulation-control {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #007bff;
    }
    .bus-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .station-info {
        background-color: #d1ecf1;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #17a2b8;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {
        'buses': [],
        'stations': [],
        'cities': [],
        'connections': []
    }
if 'stations' not in st.session_state:
    st.session_state.stations = {}


def fetch_data(endpoint):
    """Fetch data from API with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def send_control_command(action, speed=None):
    """Send control command to API"""
    try:
        payload = {"action": action}
        if speed:
            payload["speed"] = speed
        
        response = requests.post(f"{API_BASE_URL}/simulation/control", 
                               json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

def create_incident(city_a, city_b, incident_type):
    """Create incident via API"""
    try:
        payload = {
            "city_a": int(city_a),
            "city_b": int(city_b), 
            "incident_type": incident_type
        }
        response = requests.post(f"{API_BASE_URL}/incidents", 
                               json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_mock_bus_data():
    """Generate realistic mock data for demonstration"""
    cities = ["Mumbai", "Pune", "Ahmedabad", "Hyderabad", "Bengaluru", "Jaipur", "Delhi", "Chennai"]
    
    buses = []
    for i in range(8):
        current_city = random.randint(0, len(cities)-1)
        next_city = random.randint(0, len(cities)-1)
        if next_city == current_city:
            next_city = (current_city + 1) % len(cities)
        
        progress = random.uniform(0, 1)
        passengers = random.randint(15, 50)
        
        buses.append({
            "id": f"BUS-{i+1}",
            "active": random.choice([True, True, True, False]),  # 75% active
            "current_city": current_city,
            "next_city": next_city,
            "progress": progress,  # 0 to 1, how far along the route
            "passengers": passengers,
            "capacity": 60,
            "speed": random.randint(45, 70),
            "route": [current_city, next_city],
            "eta": random.randint(5, 30),
            "status": random.choice(["Moving", "Boarding", "At Station"])
        })
    
    stations = []
    for i, city in enumerate(cities):
        waiting_passengers = {}
        total_waiting = 0
        
        # Generate passengers waiting for different destinations
        for dest in range(len(cities)):
            if dest != i:  # Can't wait for same city
                count = random.randint(0, 15)
                if count > 0:
                    waiting_passengers[cities[dest]] = count
                    total_waiting += count
        
        stations.append({
            "id": i,
            "name": city,
            "waiting_passengers": waiting_passengers,
            "total_waiting": total_waiting,
            "buses_at_station": random.randint(0, 2),
            "last_arrival": random.randint(1, 10) if random.random() > 0.3 else None
        })
    
    return buses, stations, cities

def create_enhanced_network_plot(buses, stations, cities):
    """Create an enhanced network visualization with real-time bus positions"""
    
    fig = go.Figure()
    
    # City positions (arranged in a circle for better visualization)
    n_cities = len(cities)
    angles = np.linspace(0, 2*np.pi, n_cities, endpoint=False)
    city_x = np.cos(angles) * 3
    city_y = np.sin(angles) * 3
    
    # Add connections between cities (simplified - show main routes)
    connections = [
        (0, 1), (1, 2), (2, 5), (5, 6), (6, 0),  # Outer ring
        (0, 3), (1, 4), (2, 7), (3, 4), (4, 7)   # Cross connections
    ]
    
    for start, end in connections:
        if start < len(cities) and end < len(cities):
            fig.add_trace(go.Scatter(
                x=[city_x[start], city_x[end]], 
                y=[city_y[start], city_y[end]],
                mode='lines',
                line=dict(color='lightgray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add city stations
    station_colors = []
    station_sizes = []
    station_texts = []
    
    for i, station in enumerate(stations):
        total_waiting = station.get('total_waiting', 0)
        
        # Color based on passenger load
        if total_waiting > 20:
            color = 'red'
        elif total_waiting > 10:
            color = 'orange'
        elif total_waiting > 0:
            color = 'yellow'
        else:
            color = 'lightgreen'
        
        station_colors.append(color)
        station_sizes.append(max(15, min(40, 15 + total_waiting)))
        station_texts.append(f"{cities[i]}<br>Waiting: {total_waiting}")
    
    fig.add_trace(go.Scatter(
        x=city_x[:len(stations)],
        y=city_y[:len(stations)],
        mode='markers+text',
        marker=dict(
            size=station_sizes,
            color=station_colors,
            line=dict(width=2, color='darkblue')
        ),
        text=[cities[i] for i in range(len(stations))],
        textposition="bottom center",
        textfont=dict(size=12, color='darkblue'),
        name="Stations",
        hovertemplate="<b>%{text}</b><br>" +
                     "Click for details<br>" +
                     "<extra></extra>",
        customdata=station_texts
    ))
    
    # Add buses with their progress along routes
    bus_x, bus_y, bus_texts, bus_colors, bus_sizes = [], [], [], [], []
    
    for bus in buses:
        if not bus.get('active', False):
            continue
            
        current_city = bus.get('current_city', 0)
        next_city = bus.get('next_city', 0)
        progress = bus.get('progress', 0)
        passengers = bus.get('passengers', 0)
        capacity = bus.get('capacity', 60)
        
        if current_city < len(city_x) and next_city < len(city_x):
            # Calculate bus position based on progress
            bus_pos_x = city_x[current_city] + (city_x[next_city] - city_x[current_city]) * progress
            bus_pos_y = city_y[current_city] + (city_y[next_city] - city_y[current_city]) * progress
            
            bus_x.append(bus_pos_x)
            bus_y.append(bus_pos_y)
            
            # Color based on passenger load
            load_percent = (passengers / capacity) * 100
            if load_percent > 80:
                color = 'darkred'
            elif load_percent > 60:
                color = 'orange'
            elif load_percent > 30:
                color = 'blue'
            else:
                color = 'lightblue'
            
            bus_colors.append(color)
            bus_sizes.append(12)
            
            bus_texts.append(
                f"{bus['id']}<br>"
                f"Passengers: {passengers}/{capacity}<br>"
                f"Progress: {progress*100:.1f}%<br>"
                f"From: {cities[current_city]}<br>"
                f"To: {cities[next_city]}<br>"
                f"ETA: {bus.get('eta', 'N/A')} min"
            )
    
    if bus_x:  # Only add if there are active buses
        fig.add_trace(go.Scatter(
            x=bus_x, y=bus_y,
            mode='markers',
            marker=dict(
                size=bus_sizes,
                color=bus_colors,
                symbol='square',
                line=dict(width=1, color='black')
            ),
            name="Buses",
            hovertemplate="<b>🚌 %{customdata}</b><extra></extra>",
            customdata=bus_texts
        ))
    
    # Add route progress lines for active buses
    for bus in buses:
        if not bus.get('active', False):
            continue
            
        current_city = bus.get('current_city', 0)
        next_city = bus.get('next_city', 0)
        progress = bus.get('progress', 0)
        
        if current_city < len(city_x) and next_city < len(city_x):
            # Completed portion (green)
            bus_pos_x = city_x[current_city] + (city_x[next_city] - city_x[current_city]) * progress
            bus_pos_y = city_y[current_city] + (city_y[next_city] - city_y[current_city]) * progress
            
            fig.add_trace(go.Scatter(
                x=[city_x[current_city], bus_pos_x],
                y=[city_y[current_city], bus_pos_y],
                mode='lines',
                line=dict(color='green', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Remaining portion (light gray)
            fig.add_trace(go.Scatter(
                x=[bus_pos_x, city_x[next_city]],
                y=[bus_pos_y, city_y[next_city]],
                mode='lines',
                line=dict(color='lightgray', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title={
            'text': "🚌 Real-Time Bus Network",
            'x': 0.5,
            'font': {'size': 20}
        },
        showlegend=True,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def main():
    st.title("🚌 OSM Traffic Routing Dashboard")

    # Load the graph (do this once)
    @st.cache_resource
    def load_graph():
        try:
            return ox.load_graphml('pune_map.graphml')
        except FileNotFoundError:
            st.error("Error: 'pune_map.graphml' not found. Please make sure the file is in the correct directory.")
            return None
        except Exception as e:
            st.error(f"An error occurred while loading the map: {e}")
            return None
    
    G = load_graph()
    
    if G is None:
        return

    # Get map center
    map_center = [G.nodes[list(G.nodes)[0]]['y'], G.nodes[list(G.nodes)[0]]['x']]
    
    st.sidebar.header("🚉 Place Bus Stations")
    
    # Create a Folium map
    m = folium.Map(location=map_center, zoom_start=12)

    # Add existing stations to the map
    for name, data in st.session_state.stations.items():
        folium.Marker(
            location=[data['lat'], data['lon']], 
            popup=f"Station: {name}",
            icon=folium.Icon(color='blue', icon='bus')
        ).add_to(m)

    # Render map and get click events
    with st.sidebar:
        map_data = st_folium(m, width=400, height=400)

        if map_data and map_data['last_clicked']:
            lat = map_data['last_clicked']['lat']
            lon = map_data['last_clicked']['lng']
            
            station_name = st.text_input("Enter station name for the clicked point:")
            if st.button("Add Station"):
                if station_name and station_name not in st.session_state.stations:
                    # Find the nearest graph node to the clicked point
                    node_id = ox.distance.nearest_nodes(G, X=lon, Y=lat)
                    st.session_state.stations[station_name] = {
                        'lat': lat,
                        'lon': lon,
                        'node_id': node_id
                    }
                    st.success(f"Added station '{station_name}'")
                    st.rerun() # Rerun to update the map and list
                else:
                    st.warning("Please provide a unique station name.")

        st.write("### Current Stations:")
        st.json(st.session_state.stations)

    # Main content
    st.markdown('<div class="simulation-control">', unsafe_allow_html=True)
    st.subheader("🎛️ Simulation Control Center")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ START SIMULATION", key="start_sim", help="Start the bus simulation"):
            st.session_state.simulation_running = True
            if send_control_command("start"):
                st.success("✅ Simulation Started!")
            else:
                st.warning("⚠️ Starting with demo data")
    
    with col2:
        if st.button("⏸️ PAUSE", key="pause_sim", help="Pause the simulation"):
            st.session_state.simulation_running = False
            if send_control_command("pause"):
                st.success("⏸️ Simulation Paused")
    
    with col3:
        if st.button("🛑 STOP", key="stop_sim", help="Stop the simulation"):
            st.session_state.simulation_running = False
            if send_control_command("stop"):
                st.success("🛑 Simulation Stopped")
    
    with col4:
        status = "🟢 RUNNING" if st.session_state.simulation_running else "🔴 STOPPED"
        st.markdown(f"**Status:** {status}")
    
    # Speed control
    speed = st.slider("🚀 Simulation Speed", 0.1, 5.0, 1.0, 0.1, help="Adjust simulation speed multiplier")
    if st.button("Set Speed"):
        if send_control_command("start", speed):
            st.success(f"Speed set to {speed}x")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get data (real or mock)
    buses_data = fetch_data("buses")
    stations_data = fetch_data("stations")
    cities_data = fetch_data("cities")
    
    # Use mock data if API is not responding or no real data
    if not buses_data or not buses_data.get("buses"):
        st.info("📡 Using demo data - Start the actual simulation with `python main.py` for real data")
        buses, stations, cities = generate_mock_bus_data()
    else:
        buses = buses_data.get("buses", [])
        stations = stations_data.get("stations", []) if stations_data else []
        cities = cities_data.get("cities", []) if cities_data else []
    
    # Store in session state
    st.session_state.simulation_data = {
        'buses': buses,
        'stations': stations,
        'cities': cities
    }
    
    # Main visualization
    st.header("🗺️ Live Network Visualization")
    
    if buses and cities:
        network_fig = create_enhanced_network_plot(buses, stations, cities)
        selected_point = st.plotly_chart(network_fig, use_container_width=True, key="network_plot")
    else:
        st.error("❌ No simulation data available. Please start the simulation.")
    
    # Two-column layout for detailed information
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🚌 Active Buses")
        
        if buses:
            for bus in buses:
                if bus.get('active', False):
                    passengers = bus.get('passengers', 0)
                    capacity = bus.get('capacity', 60)
                    load_percent = (passengers / capacity) * 100
                    progress_percent = bus.get('progress', 0) * 100
                    
                    status_color = "🟢" if bus.get('status') == "Moving" else "🟡"
                    
                    st.markdown(f"""
                    <div class="bus-info">
                        <strong>{status_color} {bus['id']}</strong><br>
                        📍 From: {cities[bus.get('current_city', 0)] if bus.get('current_city', 0) < len(cities) else 'Unknown'}<br>
                        🎯 To: {cities[bus.get('next_city', 0)] if bus.get('next_city', 0) < len(cities) else 'Unknown'}<br>
                        👥 Passengers: {passengers}/{capacity} ({load_percent:.0f}%)<br>
                        🛣️ Progress: {progress_percent:.1f}%<br>
                        ⏰ ETA: {bus.get('eta', 'N/A')} min<br>
                        🚀 Speed: {bus.get('speed', 'N/A')} km/h
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No active buses")
    
    with col2:
        st.header("🚏 Station Status")
        
        if stations:
            for station in stations:
                total_waiting = station.get('total_waiting', 0)
                waiting_passengers = station.get('waiting_passengers', {})
                
                status_emoji = "🔴" if total_waiting > 20 else "🟡" if total_waiting > 10 else "🟢"
                
                st.markdown(f"""
                <div class="station-info">
                    <strong>{status_emoji} {station.get('name', f'Station {station.get("id", "?")}')}</strong><br>
                    👥 Total Waiting: {total_waiting}<br>
                """, unsafe_allow_html=True)
                
                if waiting_passengers:
                    st.markdown("**Destinations:**")
                    for dest, count in waiting_passengers.items():
                        st.markdown(f"  → {dest}: {count} passengers")
                
                buses_here = station.get('buses_at_station', 0)
                if buses_here > 0:
                    st.markdown(f"🚌 Buses at station: {buses_here}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No station data available")
    
    # Incident Management
    st.header("🚨 Incident Management")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if cities:
            city_from = st.selectbox("From City", cities, key="incident_from")
    
    with col2:
        if cities:
            city_to = st.selectbox("To City", cities, key="incident_to")
    
    with col3:
        incident_type = st.selectbox("Incident Type", 
                                   ["light_traffic", "heavy_traffic", "closed_road"])
    
    with col4:
        if st.button("🚨 CREATE INCIDENT", key="create_incident"):
            if cities:
                city_from_idx = cities.index(city_from)
                city_to_idx = cities.index(city_to)
                if create_incident(city_from_idx, city_to_idx, incident_type):
                    st.success(f"✅ {incident_type.replace('_', ' ').title()} incident created!")
                else:
                    st.error("❌ Failed to create incident")
    
    # Active Incidents
    incidents_data = fetch_data("incidents")
    if incidents_data and incidents_data.get("incidents"):
        st.subheader("🚨 Active Incidents")
        for incident in incidents_data["incidents"]:
            st.markdown(f"""
            **🔴 {incident['type'].replace('_', ' ').title()}**
            - Route: City {incident['city_a']} ↔ City {incident['city_b']}
            - Time Remaining: {incident['time_remaining']:.1f}h
            """)
    
    # KPI Dashboard
    st.header("📊 Key Performance Indicators")
    
    # Calculate real KPIs from current data
    if buses:
        active_buses = len([b for b in buses if b.get('active', False)])
        total_buses = len(buses)
        avg_passengers = sum(b.get('passengers', 0) for b in buses) / len(buses) if buses else 0
        avg_load = sum(b.get('passengers', 0) / b.get('capacity', 60) for b in buses) / len(buses) * 100 if buses else 0
    else:
        active_buses = total_buses = avg_passengers = avg_load = 0
    
    if stations:
        total_waiting = sum(s.get('total_waiting', 0) for s in stations)
        avg_waiting_per_station = total_waiting / len(stations) if stations else 0
    else:
        total_waiting = avg_waiting_per_station = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Buses", f"{active_buses}/{total_buses}", 
                 delta=f"{(active_buses/total_buses*100):.0f}%" if total_buses > 0 else "0%")
    
    with col2:
        st.metric("Avg Passengers/Bus", f"{avg_passengers:.1f}", 
                 delta=f"{avg_load:.0f}% capacity")
    
    with col3:
        st.metric("Total Waiting", total_waiting, 
                 delta=f"{avg_waiting_per_station:.1f} avg/station")
    
    with col4:
        efficiency = max(0, 100 - avg_waiting_per_station * 2)
        st.metric("System Efficiency", f"{efficiency:.0f}%", 
                 delta="Good" if efficiency > 70 else "Fair" if efficiency > 50 else "Poor")
    
    # Auto-refresh
    if st.session_state.simulation_running:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
