# Traffic Routing System with Multi-Agent Simulation

A traffic routing system that simulates bus transportation using multi-agent architecture, OpenStreetMap data, A* pathfinding, and LLM-based speed adjustment based on weather conditions and passenger load.

## System Overview

This system models urban bus transportation using SPADE agents, OpenStreetMap data from Blida, Algeria, and integrates weather data scraping with LLM-based speed calculations for bus movement simulation.

## Core Features

### Multi-Agent System Architecture
- **SPADE Framework**: Uses SPADE library for agent communication via XMPP
- **OpenFire XMPP Server**: Message passing between agents
- **Agent Types**:
  - **Bus Agents**: Handle individual bus movement and routing
  - **Station Agents**: Manage passenger queues and arrival times
  - **Control Agent**: Coordinates system state and agent registration
- **JID Management**: Hostname-based XMPP JID creation and registration

### Geographic Data Processing
- **OpenStreetMap Data**: GraphML format data for Blida, Algeria road network
- **Folium Visualization**: Map rendering with real-time bus positions
- **Coordinate System**: WGS84 latitude/longitude positioning
- **Graph Structure**: NetworkX graph representation of road network

![Bus Simulation Dashboard](/.figs/BusSim1.png)

### Pathfinding Algorithm
- **A* Implementation**: Standard A* algorithm for shortest path calculation
- **Adjacency Matrix**: Distance calculations using graph adjacency matrix
- **Heuristic Function**: Euclidean distance for A* heuristic
- **Path Recalculation**: Route updates when incidents occur

### Data Storage
- **SQLite Database**: Stores bus states, station states, and metrics
- **State Tracking**: Bus positions, passenger counts, timestamps
- **Database Tables**:
  - Bus states: ID, position, route, passengers, status
  - Station states: waiting passengers, predictions
  - Metrics: performance data

### Web Platform
- **FastAPI Backend**: REST API for data access
- **Endpoints**: Bus data, station data, health checks
- **CORS Support**: Cross-origin requests enabled

### Dashboard
- **Streamlit Interface**: Web dashboard for monitoring and control
- **Map Interface**: Interactive bus line creation via point selection
- **Coordinate Input**: Manual station coordinate entry

### Weather Integration and Speed Adjustment
- **Web Scraping**: BeautifulSoup scraping of timeanddate.com for Blida weather data
- **Data Extraction**: Temperature, humidity, wind, weather conditions
- **LLM Integration**: Ollama-hosted language models (Gemma 3 1B, Qwen 3 1.7B)
- **Speed Calculation**: LangChain prompt engineering for speed reduction percentage
- **Implementation**: Called during bus movement in main.py agent behavior
- **Speed Reduction Logic**:
  - Input: passenger count, weather data
  - Output: speed reduction percentage (0-50%)
  - Base speed: 60 km/h
  - Factors: passenger load categories, weather severity, temperature/humidity

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
├── RAG/
│   ├── MyLLM.py          # LLM-powered speed reduction estimation
│   └── scraper.py        # Weather data web scraping module
└── web_platform/
    ├── api_server.py     # FastAPI server implementation
    └── dashboard.py      # Streamlit dashboard interface
```

### Agent Communication
- **Message Types**: SPADE message templates with performatives
- **Timeout Handling**: Message timeout and retry mechanisms
- **State Sync**: Distributed state management across agents

### Configuration
- **config.json**: Simulation parameters, bus counts, time multipliers
- **Bus Fleet**: Initial/reserve bus counts per line, launch intervals

## Installation and Setup

### Prerequisites
- Python 3.8+
- OpenFire XMPP Server
- Ollama for LLM models

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
pip install beautifulsoup4
pip install langchain-core
pip install langchain-ollama
```

### AI Model Setup
1. **Install Ollama**: Download from ollama.ai
2. **Download Models**:
   ```bash
   ollama pull gemma3:1b
   ollama pull qwen3:1.7b
   ```

### XMPP Server Setup
1. Install OpenFire XMPP server
2. Create agent accounts
3. Configure server access

## Usage

### System Startup
```bash
python launcher.py  # Starts API server and Streamlit dashboard
```

### Manual Startup
```bash
python web_platform/api_server.py
streamlit run web_platform/dashboard.py --server.port 8501
```

### Bus Line Creation
1. Access dashboard at `http://localhost:8501`
2. Use map interface to select stations or input coordinates
3. Save line configuration
4. Start simulation

## API Endpoints

- `GET /health` - System status
- `POST /start_simulation` - Start simulation
- `GET /bus_data` - Bus positions and status
- `GET /station_data` - Station information
- `GET /metrics` - Performance metrics

## Weather Integration Technical Details

### Web Scraping Implementation
- **Target**: timeanddate.com for Blida weather
- **Parser**: BeautifulSoup HTML parsing
- **Data Fields**: temperature, weather_type, wind, humidity
- **Update**: Real-time during simulation

### LLM Speed Calculation
- **Models**: Gemma 3 1B (fast), Qwen 3 1.7B (detailed)
- **Framework**: LangChain with structured prompts
- **Input**: passenger_count, weather_data
- **Output**: speed_reduction_percentage (integer 0-50)
- **Integration**: Called in main.py bus agent movement logic

### Speed Reduction Rules
```
Low load (<20 passengers):
  - Normal weather: 0% reduction
  - Adverse weather: 5-15% reduction

Medium load (25-40 passengers):
  - Normal weather: 1-4% reduction
  - High temp/humidity: +1.5-2% additional
  - Adverse weather: 7-20% reduction

High load (45-75 passengers):
  - Normal weather: 7-15% reduction
  - High temp/humidity: +1.5-2% additional
  - Severe weather: up to 30% reduction
```

## Configuration

### config.json Parameters
- `time_speed_multiplier`: Simulation time acceleration
- `initial_buses_per_line`: Bus count per route
- `reserve_buses_per_line`: Backup bus count
- `launch_interval_hours`: Time between bus deployments

### Weather System
- Default LLM models: Gemma 3 1B, Qwen 3 1.7B
- Max speed reduction: 50%
- Geographic target: Blida, Algeria
- Weather source: timeanddate.com

## Database Schema

- **Bus States**: ID, position, route, passengers, status, timestamp
- **Station States**: ID, waiting passengers, predictions
- **Metrics**: Performance data, utilization stats

## Implementation Notes

### Weather Integration
- Weather scraping: `RAG/scraper.py` - BeautifulSoup parsing
- LLM processing: `RAG/MyLLM.py` - LangChain + Ollama integration
- Speed calculation: Called in `main.py` line 527 during bus movement
- Capped at 50% reduction for simulation stability

### Performance
- Gemma 3 1B: ~100-500ms inference
- Qwen 3 1.7B: ~500-1500ms inference
- Weather data cached to reduce API calls
- SQLite for local data persistence

