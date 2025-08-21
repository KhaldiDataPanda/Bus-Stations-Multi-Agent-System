import sqlite3
import json
import asyncio
from pathlib import Path

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.db_path = Path("data/traffic_system.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()
            self._initialized = True
    
    def _init_database(self):
        """Initialize database tables and ensure required columns exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bus states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bus_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bus_id TEXT NOT NULL,
                active BOOLEAN,
                current_city INTEGER,
                next_city INTEGER,
                distance_to_next REAL,
                status TEXT,
                route TEXT,
                timestamp REAL
            )
        ''')
        
        # Station states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS station_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                waiting_passengers TEXT,
                next_arrivals TEXT,
                timestamp REAL,
                status TEXT
            )
        ''')
        
        # Lightweight schema migration for bus_states: add columns if missing
        cursor.execute("PRAGMA table_info(bus_states)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        columns_to_add = {
            'lat': 'REAL',
            'lon': 'REAL',
            'passenger_count': 'INTEGER',
            'current_location': 'TEXT',
            'destination': 'TEXT',
            'path': 'TEXT',
            'current_station_id': 'INTEGER',
            'target_station_id': 'INTEGER',
            'assigned_line_id': 'INTEGER',
            'steps_taken': 'INTEGER'
        }
        for col_name, col_type in columns_to_add.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE bus_states ADD COLUMN {col_name} {col_type}")
        
        conn.commit()
        conn.close()
    
    async def save_bus_state(self, bus_id, state):
        """Save bus state asynchronously"""
        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO bus_states (
                    bus_id, active, current_city, next_city, distance_to_next, status, route, timestamp,
                    lat, lon, passenger_count, current_location, destination, path,
                    current_station_id, target_station_id, assigned_line_id, steps_taken
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(bus_id),
                state.get('active', False),
                state.get('current_city'),
                state.get('next_city'),
                state.get('distance_to_next', 0),
                state.get('status', 'Unknown'),
                state.get('route', '[]'),
                state.get('timestamp', 0),
                state.get('lat', 0.0),
                state.get('lon', 0.0),
                state.get('passenger_count', 0),
                state.get('current_location', ''),
                state.get('destination', ''),
                json.dumps(state.get('path', [])),
                state.get('current_station_id'),
                state.get('target_station_id'),
                state.get('assigned_line_id'),
                state.get('steps_taken', 0)
            ))
            conn.commit()
            conn.close()
        
        # Run database operation in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def save_station_state(self, station_id, state):
        """Save station state asynchronously"""
        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO station_states (station_id, waiting_passengers, 
                                          next_arrivals, timestamp, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(station_id),
                json.dumps(state.get('waiting_passengers', {})),
                json.dumps(state.get('next_arrivals', {})),
                state.get('timestamp', 0),
                state.get('status', 'active')
            ))
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    def get_recent_bus_states(self, limit=100):
        """Get recent bus states (raw rows)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM bus_states 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_latest_bus_states_map(self):
        """Get latest state per bus_id as a list of dicts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT bs.*
            FROM bus_states bs
            JOIN (
                SELECT bus_id, MAX(id) AS max_id
                FROM bus_states
                GROUP BY bus_id
            ) latest ON bs.bus_id = latest.bus_id AND bs.id = latest.max_id
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        states = []
        for row in rows:
            d = dict(row)
            # Ensure path is list
            try:
                if isinstance(d.get('path'), str):
                    d['path'] = json.loads(d['path'])
            except Exception:
                d['path'] = []
            states.append(d)
        return states
    
    def get_recent_station_states(self, limit=100):
        """Get recent station states"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM station_states 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        results = cursor.fetchall()
        conn.close()
        return results
