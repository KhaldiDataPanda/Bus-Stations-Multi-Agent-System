# Traffic Routing Dashboard Configuration
# This file contains configuration settings for the dashboard system

# API Configuration
API_HOST = "localhost"
API_PORT = 8000
DASHBOARD_PORT = 8501

# System Settings
DEFAULT_SIMULATION_SPEED = 1.0
AUTO_REFRESH_INTERVAL = 5  # seconds
MAX_EXPORT_RECORDS = 1000

# Visualization Settings
CHART_HEIGHT = 400
NETWORK_GRAPH_HEIGHT = 500
MAX_BUSES_DISPLAYED = 20
MAX_STATIONS_DISPLAYED = 15

# Performance Thresholds
EFFICIENCY_GOOD_THRESHOLD = 70
EFFICIENCY_FAIR_THRESHOLD = 50
HIGH_UTILIZATION_THRESHOLD = 80
HIGH_WAITING_THRESHOLD = 20

# Export Settings
EXPORT_DIRECTORY = "data/exports"
INCLUDE_TIMESTAMP_IN_FILENAME = True

# Database Settings
DATABASE_PATH = "data/traffic_system.db"
STATE_BACKUP_INTERVAL = 60  # seconds

# Incident Types and Durations (in system hours)
INCIDENT_DURATIONS = {
    "light_traffic": 6,
    "heavy_traffic": 12,
    "closed_road": 24
}

# City Network Configuration
# These will be loaded from cities.csv at runtime
CITY_DATA_FILE = "data/cities.csv"
