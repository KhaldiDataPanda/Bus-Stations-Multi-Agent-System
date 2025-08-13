import pandas as pd
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from db_manager import DatabaseManager
from utils import SystemState

class MetricsExporter:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.state_manager = SystemState()
        self.export_dir = Path("data/exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def calculate_kpis(self, time_window_hours=24):
        """Calculate system KPIs over a time window"""
        # Get recent data
        bus_states = self.db_manager.get_recent_bus_states(1000)
        station_states = self.db_manager.get_recent_station_states(1000)
        
        if not bus_states:
            return {}
        
        # Convert to DataFrames for easier analysis
        bus_df = pd.DataFrame(bus_states, columns=[
            'id', 'bus_id', 'active', 'current_city', 'next_city', 
            'distance_to_next', 'status', 'route', 'timestamp'
        ])
        
        station_df = pd.DataFrame(station_states, columns=[
            'id', 'station_id', 'waiting_passengers', 'next_arrivals', 'timestamp', 'status'
        ])
        
        # Calculate KPIs
        kpis = {
            'timestamp': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            
            # Bus metrics
            'total_buses': bus_df['bus_id'].nunique(),
            'average_active_buses': bus_df.groupby('timestamp')['active'].sum().mean(),
            'bus_utilization_rate': (bus_df['active'].sum() / len(bus_df)) * 100,
            
            # Route metrics
            'average_route_completion_time': self._calculate_route_completion_time(bus_df),
            'total_distance_covered': bus_df['distance_to_next'].sum(),
            
            # Passenger metrics
            'average_passenger_wait_time': self._calculate_avg_wait_time(station_df),
            'total_passengers_served': self._calculate_passengers_served(station_df),
            'passenger_throughput': self._calculate_passenger_throughput(station_df),
            
            # System performance
            'incidents_per_hour': self._calculate_incident_rate(),
            'on_time_performance': self._calculate_on_time_performance(bus_df),
            'system_efficiency_score': self._calculate_efficiency_score(bus_df, station_df)
        }
        
        return kpis
    
    def _calculate_route_completion_time(self, bus_df):
        """Calculate average route completion time"""
        # Group by bus and calculate time differences
        completion_times = []
        for bus_id, group in bus_df.groupby('bus_id'):
            if len(group) > 1:
                time_diff = group['timestamp'].max() - group['timestamp'].min()
                completion_times.append(time_diff)
        
        return sum(completion_times) / len(completion_times) if completion_times else 0
    
    def _calculate_avg_wait_time(self, station_df):
        """Calculate average passenger wait time"""
        wait_times = []
        for _, row in station_df.iterrows():
            try:
                waiting_data = json.loads(row['waiting_passengers'])
                if isinstance(waiting_data, dict):
                    # Approximate wait time based on passenger count
                    total_waiting = sum(waiting_data.values())
                    # Assume each passenger waits proportionally to queue length
                    avg_wait = total_waiting * 0.5  # Simple approximation
                    wait_times.append(avg_wait)
            except:
                continue
        
        return sum(wait_times) / len(wait_times) if wait_times else 0
    
    def _calculate_passengers_served(self, station_df):
        """Calculate total passengers served"""
        total_served = 0
        for _, row in station_df.iterrows():
            try:
                waiting_data = json.loads(row['waiting_passengers'])
                if isinstance(waiting_data, dict):
                    total_served += sum(waiting_data.values())
            except:
                continue
        return total_served
    
    def _calculate_passenger_throughput(self, station_df):
        """Calculate passengers per hour"""
        if len(station_df) == 0:
            return 0
        
        time_span = station_df['timestamp'].max() - station_df['timestamp'].min()
        total_passengers = self._calculate_passengers_served(station_df)
        
        if time_span > 0:
            return total_passengers / (time_span / 3600)  # per hour
        return 0
    
    def _calculate_incident_rate(self):
        """Calculate incidents per hour"""
        # This would require incident tracking in the database
        # For now, return a placeholder
        return 0.5
    
    def _calculate_on_time_performance(self, bus_df):
        """Calculate percentage of on-time arrivals"""
        # Simplified: assume buses are on-time if they're active
        active_count = bus_df['active'].sum()
        total_count = len(bus_df)
        
        if total_count > 0:
            return (active_count / total_count) * 100
        return 0
    
    def _calculate_efficiency_score(self, bus_df, station_df):
        """Calculate overall system efficiency score (0-100)"""
        # Composite score based on multiple factors
        bus_utilization = bus_df['active'].mean() * 100 if len(bus_df) > 0 else 0
        
        # Passenger service efficiency
        avg_waiting = self._calculate_avg_wait_time(station_df)
        passenger_efficiency = max(0, 100 - avg_waiting)  # Lower wait = higher efficiency
        
        # Combined score
        efficiency_score = (bus_utilization * 0.6 + passenger_efficiency * 0.4)
        return min(100, max(0, efficiency_score))
    
    def export_to_csv(self, filename=None):
        """Export current metrics to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.csv"
        
        kpis = self.calculate_kpis()
        
        # Convert to DataFrame for CSV export
        df = pd.DataFrame([kpis])
        filepath = self.export_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_detailed_states(self, filename=None):
        """Export detailed bus and station states"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_states_{timestamp}.csv"
        
        # Get current states
        bus_states = self.state_manager.get_bus_states()
        station_states = self.state_manager.get_station_states()
        
        # Prepare bus data
        bus_data = []
        for bus_id, state in bus_states.items():
            bus_data.append({
                'type': 'bus',
                'id': bus_id,
                'active': state.get('active'),
                'current_city': state.get('current_city'),
                'next_city': state.get('next_city'),
                'distance_to_next': state.get('distance_to_next'),
                'status': state.get('status'),
                'timestamp': datetime.now().isoformat()
            })
        
        # Prepare station data
        station_data = []
        for station_id, state in station_states.items():
            waiting_passengers = state.get('waiting_passengers', {})
            total_waiting = sum(waiting_passengers.values()) if isinstance(waiting_passengers, dict) else 0
            
            station_data.append({
                'type': 'station',
                'id': station_id,
                'total_waiting_passengers': total_waiting,
                'waiting_details': json.dumps(waiting_passengers),
                'timestamp': datetime.now().isoformat()
            })
        
        # Combine and export
        all_data = bus_data + station_data
        df = pd.DataFrame(all_data)
        filepath = self.export_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def get_performance_summary(self):
        """Get a quick performance summary"""
        kpis = self.calculate_kpis()
        
        summary = {
            "System Health": "🟢 Good" if kpis.get('system_efficiency_score', 0) > 70 else 
                           "🟡 Fair" if kpis.get('system_efficiency_score', 0) > 50 else "🔴 Poor",
            "Bus Utilization": f"{kpis.get('bus_utilization_rate', 0):.1f}%",
            "Avg Wait Time": f"{kpis.get('average_passenger_wait_time', 0):.1f} min",
            "On-Time Performance": f"{kpis.get('on_time_performance', 0):.1f}%",
            "Passengers/Hour": f"{kpis.get('passenger_throughput', 0):.1f}",
            "Efficiency Score": f"{kpis.get('system_efficiency_score', 0):.1f}/100"
        }
        
        return summary

# Utility function for dashboard
def get_exportable_metrics():
    """Get metrics in a format suitable for dashboard display"""
    exporter = MetricsExporter()
    return exporter.get_performance_summary()
