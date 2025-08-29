# Traffic Routing System with Multi-Agent Simulation

A comprehensive traffic routing system that simulates realistic bus transportation using multi-agent architecture, real-world geographic data, and advanced pathfinding algorithms.

## System Overview

This system provides a complete solution for modeling and visualizing urban transportation networks with realistic bus routing simulation. It combines multi-agent systems for distributed decision-making, geographic information systems for real-world accuracy, and web-based interfaces for real-time monitoring and control.

## Core Features

### Multi-Agent System Architecture
- **SPADE Framework Integration**: Implemented using SPADE (Smart Python Agent Development Environment) library for robust agent communication
- **XMPP Server Communication**: Utilizes OpenFire XMPP server for real-time message passing between agents
- **Agent Types**:
  - **Bus Agents**: Individual buses with autonomous routing capabilities
  - **Station Agents**: Bus stops that manage passenger queues and arrival predictions
  - **Control Agent**: Central coordinator managing system state and agent registration
- **Dynamic JID Management**: Automatic agent registration with hostname-based JID creation
- **Auto-Registration**: Seamless agent startup with automatic XMPP server registration

### Realistic Geographic Simulation
- **OpenStreetMap Integration**: Uses real-world geographic data from Blida, Algeria via GraphML format
- **Interactive Visualization**: Real-time map display using Folium for interactive geographic visualization
- **Coordinate-Based Routing**: Precise positioning using latitude/longitude coordinates
- **Real-World Constraints**: Accounts for actual road networks and geographic limitations

![Bus Simulation Dashboard](/.figs/BusSim1.png)

### Advanced Pathfinding Algorithm
- **Custom A* Implementation**: Optimized A* algorithm for efficient route calculation
- **Matrix-Based Distance Calculation**: Uses adjacency matrices for rapid pathfinding
- **Multi-Criteria Optimization**: Considers distance, travel time, and route efficiency
- **Dynamic Route Recalculation**: Real-time path updates based on current conditions

### Data Persistence and Management
- **SQLite Database**: Lightweight database for storing simulation states and historical data
- **Real-Time State Tracking**: Continuous monitoring of bus positions, passenger loads, and system metrics
- **Data Tables**:
  - Bus states (position, status, route, passengers)
  - Station states (waiting passengers, arrival predictions)
  - System metrics and performance data

### Web Platform and API
- **FastAPI Backend**: RESTful API server providing real-time data endpoints
- **Real-Time Data Sharing**: Live updates of bus positions, passenger counts, and system status
- **Health Monitoring**: API endpoints for system health checks and status monitoring
- **CORS Support**: Cross-origin resource sharing for web dashboard integration

### Interactive Dashboard
- **Streamlit Interface**: Modern web-based dashboard for system control and monitoring
- **Real-Time Visualization**: Live updates of bus movements and system metrics
- **Custom Bus Line Creation**: Two methods for creating bus routes:
  1. **Coordinate Input**: Direct latitude/longitude coordinate specification
  2. **Interactive Map Selection**: Point-and-click route creation on the map interface

![Interactive Map Interface](/.figs/BusSim2.png)

![Real-time Bus Tracking](/.figs/BusSim3.png)

## Technical Architecture

### System Components

```
├── main.py                 # Core simulation engine with agent implementations
├── launcher.py            # System launcher for coordinated startup
├── config.json           # Configuration parameters
├── data/
│   ├── db_manager.py     # SQLite database management
│   ├── Blida_map.graphml # OpenStreetMap geographic data
│   ├── bus_lines.json    # Bus line definitions
│   └── traffic_system.db # SQLite database file
├── simulation/
│   ├── utils.py          # A* algorithm and utility functions
│   ├── graph_loader.py   # GraphML data processing
│   ├── bus_lines_manager.py # Bus line and station management
│   └── full_graph_manager.py # Complete graph operations
└── web_platform/
    ├── api_server.py     # FastAPI server implementation
    └── dashboard.py      # Streamlit dashboard interface
```

### Agent Communication Protocol
- **Message Types**: Structured communication using SPADE message templates
- **Performatives**: Standard message performatives for different interaction types
- **Timeout Handling**: Robust message timeout and retry mechanisms
- **State Synchronization**: Distributed state management across all agents

### Configuration Management
- **Simulation Parameters**: Configurable time multipliers and bus deployment settings
- **Bus Fleet Management**: Adjustable initial and reserve bus counts per line
- **Launch Intervals**: Customizable bus deployment timing

## Installation and Setup

### Prerequisites
- Python 3.8+
- OpenFire XMPP Server (for agent communication)
- Required Python packages (see requirements below)

### Dependencies
```bash
pip install spade
pip install fastapi
pip install streamlit
pip install folium
pip install streamlit-folium
pip install networkx
pip install pandas
pip install numpy
pip install plotly
pip install requests
```

### XMPP Server Setup
1. Install and configure OpenFire XMPP server
2. Create agent accounts with password authentication
3. Ensure server is accessible on the local network

## Usage

### System Startup
1. **Launch Complete System**:
   ```bash
   python launcher.py
   ```
   This starts both the API server and Streamlit dashboard automatically.

2. **Manual Component Startup**:
   ```bash
   # Start API server
   python web_platform/api_server.py
   
   # Start dashboard (in separate terminal)
   streamlit run web_platform/dashboard.py --server.port 8501
   ```

### Creating Bus Lines
1. **Via Dashboard Interface**:
   - Access the dashboard at `http://localhost:8501`
   - Use the "Bus Line Creation" section
   - Select points on the interactive map or input coordinates directly

2. **Via Coordinate Input**:
   - Enter station coordinates in the coordinate input form
   - Specify line name and station details
   - Save the line configuration

### Running Simulation
1. Ensure bus lines are created
2. Start the simulation from the dashboard interface
3. Monitor real-time bus movements and system metrics
4. Access API endpoints at `http://localhost:8000` for data integration

## API Endpoints

### Core Endpoints
- `GET /health` - System health check
- `POST /start_simulation` - Start the simulation
- `GET /bus_data` - Current bus positions and status
- `GET /station_data` - Station states and passenger information
- `GET /metrics` - System performance metrics

### Data Formats
- **Bus Data**: Position, route, passenger count, status
- **Station Data**: Waiting passengers, arrival predictions
- **Metrics**: System utilization, performance indicators

## System Configuration

The system uses `config.json` for parameter configuration:
- **Time Speed Multiplier**: Simulation time acceleration factor
- **Bus Fleet Settings**: Initial and reserve bus counts per line
- **Launch Intervals**: Time between bus deployments

## Database Schema

### Bus States Table
- Bus ID, position, route, passenger count, status, timestamp

### Station States Table
- Station ID, waiting passengers, arrival predictions, status

### System Metrics
- Performance data, utilization statistics, historical trends

## Performance Characteristics

- **Real-Time Processing**: Sub-second response times for route calculations
- **Scalable Architecture**: Supports multiple bus lines and large fleets
- **Efficient Pathfinding**: Optimized A* implementation for rapid route computation
- **Concurrent Operations**: Multi-threaded processing for simultaneous agent operations
