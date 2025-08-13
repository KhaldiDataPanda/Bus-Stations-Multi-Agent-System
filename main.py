from collections import defaultdict
import pandas as pd
import random
import json
import asyncio
import logging
import time
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
from utils import log_message,a_star, SystemTime ,SystemState
from db_manager import DatabaseManager
from pathlib import Path


Path("data/state").mkdir(parents=True, exist_ok=True)


db_manager = DatabaseManager()





MESSAGE_FORMATS = {
    'system': "[SYSTEM TIME {:.2f}h] {}",
    'bus': "[BUS-{} | {:.2f}h] {}",
    'station': "[STATION-{} | {:.2f}h] {}",
    'control': "[CONTROL | {:.2f}h] {}",
    'incident': "[INCIDENT | {:.2f}h] Bus {} - {} between cities {} and {}" }

LOGGING_FORMAT = {
    'bus': '[BUS-%(bus_id)s ] - %(levelname)s - %(message)s',
    'station': '[STATION-%(station_id)s ] - %(levelname)s - %(message)s',
    'control': '[CONTROL ] - %(levelname)s - %(message)s',
    'messaging': '[MESSAGING ] - %(levelname)s - %(message)s' }

DEBUG_MESSAGING = True





############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


bus_logger = logging.getLogger('bus')
station_logger = logging.getLogger('station')
control_logger = logging.getLogger('control')

messaging_logger = logging.getLogger('messaging')
messaging_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(LOGGING_FORMAT['messaging']))
messaging_logger.addHandler(handler)



#-----------------------------------------------------------------------------------------------------------
#------------------------------------------   DATA  --------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------



graph_df = pd.read_csv('data/cities.csv')
graph_df.drop_duplicates(subset=['Origin', 'Destination'], inplace=True)
df = graph_df.pivot(index='Origin', columns='Destination', values='Distance').fillna(0)
df = df.loc[df.index.intersection(df.columns), df.index.intersection(df.columns)]
distance_matrix = df.to_numpy().astype(int)
city_list = df.index.tolist()






lines = [[0, 4],    
          #[2, 11],    
          #[13, 1],    
          #[3, 8],    
          #[4, 6],    
          #[1, 5],    
          [7, 2]]    




system_time = SystemTime()
state_manager = SystemState()



############################################################################################################
############################################################################################################
#-----------------------------              BUS AGENT          ---------------------------------------------
############################################################################################################
############################################################################################################



class BusAgent(Agent):
    
    async def setup(self):
        self.bus_id = int(str(self.jid).split('_')[1].split('@')[0])
        self.route = None
        self.is_initialized = False

        if DEBUG_MESSAGING:
            print(MESSAGE_FORMATS['bus'].format(self.bus_id, system_time.get_current_time(), "Agent starting setup..."))
            print('\n')

        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(self.InitBehavior())
        self.add_behaviour(self.MessageHandler())  


    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------


    class InitBehavior(CyclicBehaviour):

        async def run(self):

            if not self.agent.is_initialized:
                msg = Message(to="control@laptop-ko0jtu4m")
                msg.set_metadata("performative", "subscribe")
                msg.body = f"REGISTER:{self.agent.bus_id}"

                log_message("SEND", f"Bus_{self.agent.bus_id}", "Control", msg.body,system_time)

                await self.send(msg)
                msg = await self.receive(timeout=5)

                if msg:
                    log_message("RECEIVE", "Control", f"Bus_{self.agent.bus_id}", msg.body,system_time)
                    if "INIT_CONFIRM" in msg.body:
                        self.agent.is_initialized = True
                        bus_logger.info(f"Bus {self.agent.bus_id} received initialization confirmation")
                        behaviour = self.agent.BusBehaviour(
                            route_id=self.agent.name,
                            line_id=random.choice(lines),
                            start_time=0,
                            bus_id=self.agent.bus_id)
                        
                        self.agent.add_behaviour(behaviour)
                        bus_logger.info(f"Bus {self.agent.bus_id} initialized and ready")
                else:
                    bus_logger.warning(f"Bus {self.agent.bus_id} didn't receive initialization confirmation")
            else:
                await asyncio.sleep(1)
                
                
    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------


    class BusBehaviour(CyclicBehaviour):

        def __init__(self, route_id, line_id, start_time, bus_id):
            super().__init__()
            self.bus_id = bus_id
            self.route_id = route_id
            self.line_id = line_id
            self.start_time = start_time
            self.passengers = random.randint(10, 50)
            self.speed = max(0, 60000 - (3 * (self.passengers // 10)))
            self.current_station_idx = 0
            self.route = None
            self.is_active = True
            self.current_time = start_time
            self.waiting_for_route = True

        #--------------------------------------------

        async def run(self):

            state = {
                'bus_id': self.bus_id,
                'active': self.is_active,
                'current_city': self.route[self.current_station_idx] if self.route else None,
                'next_city': self.route[self.current_station_idx + 1] if self.route and self.current_station_idx < len(self.route) - 1 else None,
                'distance_to_next': distance_matrix[self.route[self.current_station_idx]][self.route[self.current_station_idx + 1]] if self.route and self.current_station_idx < len(self.route) - 1 else 0,
                'status': "Active" if self.is_active else "Waiting",
                'timestamp': time.time(),
                'route': json.dumps(self.route) if self.route else "[]"  } # Convert route to JSON string
            
            # Save state to database asynchronously
            await db_manager.save_bus_state(self.bus_id, state)


            state = {
                'active': self.is_active,
                'current_city': self.route[self.current_station_idx] if self.route else None,
                'next_city': self.route[self.current_station_idx + 1] if self.route and self.current_station_idx < len(self.route) - 1 else None,
                'distance_to_next': distance_matrix[self.route[self.current_station_idx]][self.route[self.current_station_idx + 1]] if self.route and self.current_station_idx < len(self.route) - 1 else 0,
                'status': "Active" if self.is_active else "Waiting"}
            

            state_manager.update_bus_state(self.bus_id, state)
            print(f"[DEBUG] Updated bus {self.bus_id} state: {state}")  # Add debug print
            await asyncio.sleep(1)  # Use async sleep instead of blocking sleep

            if self.waiting_for_route:

                template = Template()
                template.set_metadata("performative", "inform")
                msg = await self.receive(timeout=20)  
                print(MESSAGE_FORMATS['bus'].format(self.route_id, system_time.get_current_time(), f"received {type(msg)} message"))
                
                if msg and "ROUTE_ASSIGNMENT" in msg.body:
                    
                    try:
                        _, route_data_str = msg.body.split(":", 1)
                        self.route = json.loads(route_data_str)  # Use JSON instead of eval
                        self.waiting_for_route = False
                        
                        ack = Message(to="control@laptop-ko0jtu4m")
                        ack.set_metadata("performative", "confirm")
                        ack.body = f"ROUTE_ACCEPTED:{self.bus_id}"
                        await self.send(ack)
                        bus_logger.info(f"Bus {self.route_id} received and confirmed route: {self.route}")

                    except (ValueError, SyntaxError) as e:
                        bus_logger.error(f"Bus {self.route_id} received invalid route data: {msg.body}. Error: {e}")
                        return
                return  # Wait for next cycle if no route received
            

            if not self.route:
                return  # Skip processing if route is not set
            
            if self.is_active and self.current_station_idx < len(self.route) - 1:
                current_city = self.route[self.current_station_idx]
                next_city = self.route[self.current_station_idx + 1]
                distance = distance_matrix[current_city][next_city]
                incident_type, delay_factor = await self.handle_incident(current_city, next_city, distance)
                base_travel_time = (distance / self.speed) * 60  # in minutes
                actual_travel_time = base_travel_time * delay_factor
                
                if incident_type != 'closed_road':
                    # Normal travel or traffic delay
                    print(MESSAGE_FORMATS['bus'].format(self.route_id, system_time.get_current_time(), f"traveling from city {current_city} to {next_city}. ETA: {actual_travel_time:.2f} minutes"))
                    await asyncio.sleep(actual_travel_time / system_time.time_multiplier)  # Convert to real seconds
                    self.current_station_idx += 1
                else: # Closed road - request new route
                    print(MESSAGE_FORMATS['bus'].format(self.route_id, system_time.get_current_time(), f"requesting new route due to closed road"))
                    msg = Message(to="control@laptop-ko0jtu4m")
                    msg.set_metadata("performative", "inform")
                    msg.body = f"ROUTE_RECALCULATION:{self.route_id}:{current_city}:{next_city}"
                    await self.send(msg)


        #--------------------------------------------


        async def handle_incident(self, current_city, next_city, distance):
            """
            Incident probability based on distance:
            - Closed road: 1% * (distance/100)
            - Heavy traffic: 7% * (distance/100)
            - Light traffic: 50% * (distance/100)
            """
            probability = distance / 100
            incident_type = None
            delay_factor = 0
            if random.random() < 0.01 * probability:  # Closed road
                incident_type = 'closed_road'
                print(MESSAGE_FORMATS['incident'].format(system_time.get_current_time(), self.route_id, incident_type, current_city, next_city))       
            
            elif random.random() < 0.07 * probability:  # Heavy traffic
                incident_type = 'heavy_traffic'
                delay_factor = 1.7
                print(MESSAGE_FORMATS['incident'].format(system_time.get_current_time(), self.route_id, incident_type, current_city, next_city))       
            
            elif random.random() < 0.5 * probability:  # Light traffic
                incident_type = 'light_traffic'
                delay_factor = 1.2
                print(MESSAGE_FORMATS['incident'].format(system_time.get_current_time(), self.route_id, incident_type, current_city, next_city))
            
            if incident_type: # Notify station and control of incident
                msg = Message(to=f"station_{next_city}@laptop-ko0jtu4m")
                msg.set_metadata("performative", "inform")
                msg.body = f"INCIDENT:{self.route_id}:{incident_type}:{current_city}:{next_city}"
                await self.send(msg)  # Remove reliable parameter

                bus_logger.info(f"Bus {self.route_id} sent INCIDENT message to station_{next_city}")
                system_time.add_incident(current_city, next_city, incident_type)
                return incident_type, delay_factor
            
            return None, 1.0


    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------



    class MessageHandler(CyclicBehaviour):

        async def run(self):
            msg = await self.receive(timeout=1)  # Shorter timeout
            if msg and isinstance(msg, Message):  # Check for valid message
                if "ROUTE_ASSIGNMENT" in msg.body:
                    await self.handle_route_assignment(msg)
                elif "ROUTE_UPDATE" in msg.body:
                    await self.handle_route_update(msg)

        #--------------------------------------------

        async def handle_route_assignment(self, msg):
            try:
                _, route_data_str = msg.body.split(":", 1)
                route_data = json.loads(route_data_str)  # Use JSON instead of eval

                ack = Message(to="control@laptop-ko0jtu4m") # Send acknowledgment
                ack.set_metadata("performative", "confirm")
                ack.body = f"ROUTE_ACCEPTED:{self.agent.bus_id}"
                await self.send(ack)
                
                self.agent.route = route_data # Update agent state
                self.agent.is_initialized = True
                
            except Exception as e:
                bus_logger.error(f"Error handling route assignment: {e}")

        async def handle_route_update(self, msg):
            try:
                _, route_data_str = msg.body.split(":", 1)
                route_data = json.loads(route_data_str)  # Use JSON instead of eval

                # Update the agent's route
                self.agent.route = route_data
                print(MESSAGE_FORMATS['bus'].format(self.agent.bus_id, system_time.get_current_time(), f"Route updated: {route_data}"))
                
            except Exception as e:
                bus_logger.error(f"Error handling route update: {e}")




############################################################################################################
############################################################################################################
#-----------------------------          STATION AGENT          ---------------------------------------------
############################################################################################################
############################################################################################################




class StationAgent(Agent):
    """
    Station responsibilities:
    - Track waiting passengers
    - Monitor bus arrivals
    - Update passenger counts
    - Report station status
    """
    async def setup(self):
        self.station_id = int(str(self.jid).split('_')[1].split('@')[0])
        self.is_initialized = False
        self.registration_attempts = 0
        self.max_registration_attempts = 3
        
        if DEBUG_MESSAGING:
            print(MESSAGE_FORMATS['station'].format(self.station_id, system_time.get_current_time(), "StationAgent starting setup..."))
        
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(self.StationInitBehaviour())
        self.add_behaviour(self.StationBehaviour(self.station_id))
        station_logger.info(f"[Station] {self.station_id} setup complete")

        
    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------

    class StationInitBehaviour(CyclicBehaviour):
        async def run(self):
            if not self.agent.is_initialized and self.agent.registration_attempts < self.agent.max_registration_attempts:
                try:
                    msg = Message(to="control@laptop-ko0jtu4m")
                    msg.set_metadata("performative", "subscribe")
                    msg.body = f"REGISTER:{self.agent.station_id}"
                    log_message("SEND", f"Station_{self.agent.station_id}", "Control", msg.body,system_time, "INIT_CONFIRM")
                    await self.send(msg)
                
                    response = await self.receive(timeout=10)
                    
                    if response and response.body == "INIT_CONFIRM":
                        self.agent.is_initialized = True
                        station_logger.info(f"Station {self.agent.station_id} initialized")
                    else:
                        self.agent.registration_attempts += 1
                        await asyncio.sleep(1)
                except Exception as e:
                    station_logger.error(f"Error during station {self.agent.station_id} registration: {e}")
                    self.agent.registration_attempts += 1
                    await asyncio.sleep(1)


    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------


    class StationBehaviour(CyclicBehaviour):

        def __init__(self, station_id):
            super().__init__()
            self.station_id = station_id
            self.waiting_passengers = {i: random.randint(0, 30) for i in range(7)}
            self.eta_times = {}

        #---------------------------------------

        async def run(self):
            state = {
                'station_id': self.station_id,
                'waiting_passengers': self.waiting_passengers,
                'next_arrivals': self.eta_times,
                'timestamp': time.time(),
                'status': 'active'  } 
            
            # Save state to database asynchronously
            await db_manager.save_station_state(self.station_id, state)

            state = {
                'waiting_passengers': self.waiting_passengers.copy(),
                'next_arrivals': self.eta_times.copy() }
            

            state_manager.update_station_state(self.station_id, state)

            eta_template = Template()
            eta_template.set_metadata("performative", "inform")
            eta_template.body = "ETA:*"

            delay_template = Template()
            delay_template.set_metadata("performative", "inform")
            delay_template.body = "DELAY:*"

            msg = await self.receive(timeout=2)
            if msg:
                messaging_logger.debug(f"Station_{self.station_id} received message: {msg.body}")
                performative = msg.get_metadata("performative")

                if performative == "inform":

                    if eta_template.match(msg):  
                        route_id, eta = msg.body.split(":")[1:] # Handle ETA message
                        self.eta_times[route_id] = float(eta)
                        print(MESSAGE_FORMATS['station'].format(self.station_id, system_time.get_current_time(), f"Updated ETA for route {route_id}: {eta} minutes"))
                    
                    elif delay_template.match(msg):
                        route_id, delay = msg.body.split(":")[1:]# Handle delay message
                        if route_id in self.eta_times:
                            self.eta_times[route_id] += float(delay)
                            print(MESSAGE_FORMATS['station'].format(self.station_id, system_time.get_current_time(), f"Route {route_id} delayed by {delay} minutes"))
                
                if "ARRIVAL:" in msg.body:

                    route_id = msg.body.split(":")[1]
                    if route_id in self.waiting_passengers:
                        msg = Message(to="control@laptop-ko0jtu4m")
                        msg.set_metadata("performative", "inform")
                        msg.body = f"BUS_ARRIVED:{self.station_id}:{route_id}:{self.waiting_passengers[route_id]}"
                        messaging_logger.debug(f"Station_{self.station_id} sending BUS_ARRIVED message")
                        await self.send(msg)  # Remove reliable parameter
                        messaging_logger.debug(f"Station_{self.station_id} sent BUS_ARRIVED message")
                        station_logger.info(f"Station {self.station_id} sent BUS_ARRIVED message to control")
                        self.waiting_passengers[route_id] = 0

                if random.random() < 0.3:
                    route_id = random.randint(0, 6)
                    new_passengers = random.randint(1, 5)
                    self.waiting_passengers[route_id] += new_passengers
                    msg = Message(to="control@laptop-ko0jtu4m")
                    msg.set_metadata("performative", "inform")
                    msg.body = f"NEW_PASSENGERS:{self.station_id}:{route_id}:{new_passengers}"
                    await self.send(msg)
            else:
                station_logger.debug(MESSAGE_FORMATS['station'].format(self.station_id, system_time.get_current_time(), f"No messages received"))
    
    





############################################################################################################
############################################################################################################
#-----------------------------            CONTROL AGENT        ---------------------------------------------
############################################################################################################
############################################################################################################



class ControlAgent(Agent):
    """
    System coordinator:
    - Route generation using A*
    - Bus registration management
    - Incident response coordination
    - System state monitoring
    - Dynamic route reassignment
    """
    async def setup(self):
        print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Setup"))
        self.buses = []
        self.registered_buses = set()
        self.registered_stations = set()
        self.message_queue = []
        self.system_ready = False
        self.temporary_matrix = distance_matrix.copy()
        self.pending_registrations = {}
        self.registration_timeouts = {}
        
        template = Template()
        template.set_metadata("performative", "subscribe")
        self.add_behaviour(self.ControlBehaviour(self.buses))
        control_logger.info("Control agent setup complete")
    
    def buses_setter(self, buses):
        print('\n\nThe setter is called')
        self.buses = buses
        if len(self.behaviours) > 0:
            print('The setter entred\n\n\n')
            self.behaviours[0].buses = buses

    #---------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------


    class ControlBehaviour(CyclicBehaviour):  
        def __init__(self, buses):
            super().__init__()
            self.buses = []
            self.active_buses = {}
            self.ready_buses = set()
            self.routes = None
            self.initialization_phase = True
            self.station_status = defaultdict(lambda: {'waiting': {}})
            self.registered_buses = set()
            self.registered_stations = set()
            self.temporary_matrix = distance_matrix.copy()  # Add temporary_matrix to behavior
            self.incident_history = {}  # Track incidents and their recovery times

        async def run(self):
    
            msg = await self.receive(timeout=5)
            if not msg:
                return

            print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), f"Received message: {msg.body}\n\n"))
            message_type = self.get_message_type(msg.body)
            await self.handle_message(message_type, msg)
            
            
        def generate_routes(self):
            routes = []
            for i in range(len(lines)):
                start, goal = lines[i]
                path, distance = a_star(distance_matrix, start, goal)
                if path:
                    print('\n----------------------[Control Generating Routs]----------------------')
                    print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), f"Agent generated path {[city_list[city] for city in path]} to line {i}\n"))
                    routes.append((i, path))
            return routes
            

        def get_message_type(self, body):
            if "REGISTER" in body:
                return "REGISTER"
            elif "BUS_ARRIVED:" in body:
                return "BUS_ARRIVED"
            elif "NEW_PASSENGERS:" in body:
                return "NEW_PASSENGERS"
            elif "INCIDENT:" in body:
                return "INCIDENT"
            elif "ROUTE_COMPLETE:" in body:
                return "ROUTE_COMPLETE"
            elif "ROUTE_ACCEPTED:" in body:
                return "ROUTE_ACCEPTED"
            elif "ROUTE_RECALCULATION:" in body:
                return "ROUTE_RECALCULATION"
            return "UNKNOWN"

        async def handle_message(self, message_type, msg):
            handlers = {
                "REGISTER": self.handle_registration,
                "BUS_ARRIVED": self.handle_bus_arrived,
                "NEW_PASSENGERS": self.handle_new_passengers,
                "INCIDENT": self.handle_incident_message,
                "ROUTE_COMPLETE": self.handle_route_complete,
                "ROUTE_ACCEPTED": self.handle_route_accepted,
                "ROUTE_RECALCULATION": self.handle_route_recalculation
            }
            
            handler = handlers.get(message_type)
            if handler:
                await handler(msg)
            else:
                control_logger.warning(f"Unknown message type received: {msg.body}")

        async def handle_registration(self, msg):

            try:
                _, entity_id = msg.body.split(":")
                entity_id = int(entity_id)
                sender_type = str(msg.sender).split("_")[0]
                sender_full = str(msg.sender)
                response = Message(to=sender_full)
                response.set_metadata("performative", "inform")
                response.body = "INIT_CONFIRM"
                await self.send(response)

                if "bus" in sender_type.lower():
                    self.registered_buses.add(entity_id)
                    control_logger.info(f"\n[H.R] Bus {entity_id} registered. Total: {len(self.registered_buses)}. Expected: {len(self.buses)}\n")
                elif "station" in sender_type.lower():
                    self.registered_stations.add(entity_id)
                    control_logger.info(f"\n[H.R] Station {entity_id} registered. Total: {len(self.registered_stations)}. Expected: {len(city_list)}\n")


                if (len(self.registered_buses) == 6 and 
                    len(self.registered_stations) == len(city_list)):

                    control_logger.info("All agents registered. Starting route generation...")
                    self.initialization_phase = False
                    self.routes = self.generate_routes()
                    print(f'[Control] generated Routes: {self.routes}')
                    await self.assign_routes_to_buses()
                    control_logger.info("*******Control System fully Rigistered*******")

                return True

            except Exception as e:
                control_logger.error(f"Error in registration handler: {e}")
                return False



        async def handle_bus_arrived(self, msg):
            _, station_id, route_id, passengers = msg.body.split(":")
            station_id, route_id = int(station_id), int(route_id)
            self.station_status[station_id]['waiting'][route_id] = int(passengers)
            print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), f"Bus {route_id} arrived at Station {station_id} with {passengers} passengers."))


        async def handle_new_passengers(self, msg):
            _, station_id, route_id, new_passengers = msg.body.split(":")
            station_id, route_id = int(station_id), int(route_id)
            if route_id in self.station_status[station_id]['waiting']:
                self.station_status[station_id]['waiting'][route_id] += int(new_passengers)
            else:
                self.station_status[station_id]['waiting'][route_id] = int(new_passengers)
            print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), f"{new_passengers} new passengers waiting for Route {route_id} at Station {station_id}."))


        async def handle_incident_message(self, msg):
            _, route_id, incident_type, current_city, next_city = msg.body.split(":")
            route_id, current_city, next_city = map(int, [route_id, current_city, next_city])
            await self.handle_incident(route_id, current_city, next_city, incident_type)


        async def handle_route_complete(self, msg):
            _, route_id = msg.body.split(":")
            route_id = int(route_id)
            if route_id in self.active_buses:
                del self.active_buses[route_id]
                print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), f"Route {route_id} completed. Removing from active buses."))

        async def handle_route_accepted(self, msg):
            try:
                _, bus_id = msg.body.split(":")
                bus_id = int(bus_id)
                control_logger.info(f"Bus {bus_id} accepted its route")
            except Exception as e:
                control_logger.error(f"Error handling route acceptance: {e}")



        async def handle_route_recalculation(self, msg):
            try:
                _, route_id, current_city, next_city = msg.body.split(":")
                route_id = int(route_id)
                current_city = int(current_city)
                next_city = int(next_city)
                
                
                destination = None  # Find the destination city from the original routes
                for route in self.routes:
                    if route[0] == route_id // 3:  # Use integer division to find line_id
                        destination = route[1][-1]  # Get the last city in the route
                        break
                
                if destination is None:
                    raise ValueError(f"No destination found for route {route_id}")
                

                new_path, _ = a_star(self.temporary_matrix, current_city, destination) # Calculate new route from current city to destination
                

                if new_path: # Send the new route to the bus
                    msg = Message(to=f"bus_{route_id}@laptop-ko0jtu4m")
                    msg.set_metadata("performative", "inform")
                    msg.body = f"ROUTE_ASSIGNMENT:{json.dumps(new_path)}"  # Use JSON serialization
                    await self.send(msg)
                    control_logger.info(f"New route sent to bus {route_id}: {new_path}")
                else:
                    control_logger.error(f"No alternative route found for bus {route_id}")
            except Exception as e:
                control_logger.error(f"Error calculating new route: {e}")




        async def assign_routes_to_buses(self):
            print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Assigning routes to buses...\n\n "))
            
            for bus in self.buses:
                try:
                    bus_id = int(bus.name.split('_')[1])
                    line_id = bus_id // 3  # Use integer division
                    
                    if line_id < len(self.routes):
                        route_data = self.routes[line_id][1]
                        assigned = await self.retry_route_assignment(bus, route_data)
                        
                        if assigned:
                            self.active_buses[bus_id] = route_data
                        else:
                            control_logger.error(f"Failed to assign route to bus {bus_id}")
                
                except Exception as e:
                    control_logger.error(f"Error in route assignment process: {str(e)}")
                    continue

        async def retry_route_assignment(self, bus, route_data, max_retries=3):
            bus_id = int(bus.name.split('_')[1])
            for attempt in range(max_retries):
                try:
                    msg = Message(to=f"bus_{bus_id}@laptop-ko0jtu4m")  # Fixed message creation
                    msg.set_metadata("performative", "inform")
                    msg.body = f"ROUTE_ASSIGNMENT:{json.dumps(route_data)}"  # Use JSON serialization
                    await self.send(msg)

                    template = Template()
                    template.set_metadata("performative", "confirm")
                    ack = await self.receive(timeout=5)

                    if ack and "ROUTE_ACCEPTED" in ack.body:
                        control_logger.info(f"Bus {bus_id} accepted route on retry {attempt + 1}")
                        return True
                    
                    await asyncio.sleep(1)
                except Exception as e:
                    control_logger.error(f"Retry attempt {attempt + 1} failed for bus {bus_id}: {str(e)}")
                    await asyncio.sleep(1)

            control_logger.error(f"Failed to assign route to bus {bus_id} after {max_retries} attempts")
            return False
        
        
                
        
        
        

        async def handle_incident(self, route_id, current_city, next_city, incident_type):
            """
            Incident response protocol:
            1. Update network state
            2. Calculate recovery time
            3. Adjust path costs
            4. Trigger rerouting if needed
            """
            print(MESSAGE_FORMATS['incident'].format(system_time.get_current_time(), route_id, incident_type, current_city, next_city))
            
            recovery_times = {
                'light_traffic': 6,  
                'heavy_traffic': 12, 
                'closed_road': 24   }
            
            incident_key = f"{current_city}-{next_city}"
            self.incident_history[incident_key] = {
                'type': incident_type,
                'start_time': system_time.get_current_time(),
                'recovery_time': recovery_times[incident_type] }

            
            if incident_type == 'light_traffic':  
                self.temporary_matrix[current_city][next_city] *= 1.2
                print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Light traffic will last for 6 hours"))
            elif incident_type == 'heavy_traffic':
                self.temporary_matrix[current_city][next_city] *= 1.7
                print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Heavy traffic will last for 12 hours"))
            elif incident_type == 'closed_road':
                self.temporary_matrix[current_city][next_city] = float('inf')
                self.temporary_matrix[next_city][current_city] = float('inf')
                print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Road closure will last for 24 hours"))

            
            if route_id in self.active_buses: # Handle rerouting if needed
                current_route = self.active_buses[route_id]
                try:
                    new_path, _ = a_star(self.temporary_matrix, current_city, current_route[-1])
                    if new_path:
                        print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), f"New route calculated for Bus {route_id}: {new_path}"))
                        self.active_buses[route_id] = new_path
                        msg = Message(to=f"bus_{route_id}@laptop-ko0jtu4m")
                        msg.set_metadata("performative", "inform")
                        msg.body = f"ROUTE_UPDATE:{json.dumps(new_path)}"  # Use JSON serialization
                        await self.send(msg)
                        control_logger.info(f"New route sent to bus {route_id}: {new_path}")
                except Exception as e:
                    control_logger.error(f"Error calculating new route for bus {route_id}: {e}")

       

        async def update_incident_effects(self):  # update incident effects periodically

            current_time = system_time.get_current_time()
            incidents_to_remove = []
            
            for incident_key, incident in self.incident_history.items():
                elapsed_time = current_time - incident['start_time']

                if elapsed_time >= incident['recovery_time']:
                    current_city, next_city = map(int, incident_key.split('-'))
                    self.temporary_matrix[current_city][next_city] = distance_matrix[current_city][next_city]

                    if incident['type'] == 'closed_road':
                        self.temporary_matrix[next_city][current_city] = distance_matrix[next_city][current_city]

                    incidents_to_remove.append(incident_key)
                    print(MESSAGE_FORMATS['control'].format(current_time, f"Incident cleared between cities {current_city} and {next_city}"))

            for key in incidents_to_remove:
                del self.incident_history[key]


 

############################################################################################################
############################################################################################################
############################################################################################################



if __name__ =="__main__":
    """
    Initialization sequence:
    1. Start time management
    2. Initialize control agent
    3. Create station agents
    4. Launch bus agents
    5. Begin system operation
    """
    async def main():
        time_task = asyncio.create_task(system_time.update_time())
        
        
        buses = []
        for line_id in range(len(lines)):
            for i in range(3):
                bus_id = line_id * 3 + i
                bus = BusAgent(f"bus_{bus_id}@laptop-ko0jtu4m", "password")
                buses.append(bus)

        # Start control agent
        control = ControlAgent("control@laptop-ko0jtu4m", "00000000")
        
        await control.start()
        await asyncio.sleep(2)
        print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Agent initialized"))
        control.buses_setter(buses)
        
        
        stations = []
        for i in range(len(city_list)):
            station = StationAgent(f"station_{i}@laptop-ko0jtu4m", "00000000")
            await station.start()
            stations.append(station)
            await asyncio.sleep(1)

        print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Station Agents initialized"))


        for bus in buses:
            await bus.start()
            await asyncio.sleep(1)

        print(MESSAGE_FORMATS['control'].format(system_time.get_current_time(), "Bus Agents initialized"))
        print(f'LENGTH BUSES: {len(buses)}')



        print("----------------------- ALL AGENTS INITIALIZED ------------------------- ")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            for bus in buses:
                await bus.stop()
            for station in stations:
                await station.stop()
            await control.stop()
            time_task.cancel()


    asyncio.run(main())