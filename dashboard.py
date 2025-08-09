import streamlit as st
import pandas as pd
from pathlib import Path
import ast
import time
import os


def run_dashboard():
    st.set_page_config(layout="wide")
    st.title("Bus Network Management System")
    print('got in 0')
    while True:
        col1, col2 = st.columns(2)
        print('got in 0.0')
        with col1:
            print('got in 0.0.0')
            st.header("Bus Status")
            


            if Path('data/state/bus_states.csv').exists():
                print('got in 1')
                df = pd.read_csv('data/state/bus_states.csv')
                print(type(df))
                # Get only the latest state for each bus
                latest_states = df.sort_values('timestamp').groupby('bus_id').last()
                st.dataframe(latest_states.drop('timestamp', axis=1), use_container_width=True)
            else:
                st.info("Waiting for bus data.")
        
        with col2:
            st.header("Station Status")
            try:
                if Path('data/state/station_states.csv').exists():
                    print('got in 2')
                    df = pd.read_csv('data/state/station_states.csv')
                    
                    # Get only the latest state for each station
                    latest_states = df.sort_values('timestamp').groupby('station_id').last()
                    
                    for station_id, row in latest_states.iterrows():
                        st.subheader(f"Station {station_id}")
                        # Handle potential string representations of dictionaries
                        waiting = ast.literal_eval(row['waiting_passengers']) if isinstance(row['waiting_passengers'], str) else row['waiting_passengers']
                        arrivals = ast.literal_eval(row['next_arrivals']) if isinstance(row['next_arrivals'], str) else row['next_arrivals']
                        
                        if waiting and arrivals:  # Only create dataframe if there's data
                            data = pd.DataFrame({
                                'Line': list(waiting.keys()),
                                'Waiting Passengers': list(waiting.values()),
                                'Next Bus Arrival': [arrivals.get(str(k), "Unknown") for k in waiting.keys()]
                            })
                            st.dataframe(data, use_container_width=True)
                        else:
                            st.info("No passengers waiting at this station")
                else:
                    st.info("Waiting for station data...")
            except Exception as e:
                st.error(f"Error reading station data: {e}")
                st.error("Data structure: " + str(df.columns.tolist()) if 'df' in locals() else "No DataFrame available")
        
        time.sleep(1)
        st.experimental_rerun()

if __name__ == "__main__":
    run_dashboard()