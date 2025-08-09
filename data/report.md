# Technical Report: Multi-Agent Bus Network Management System

## 1. Project Overview
This project implements a multi-agent system (MAS) for managing a city bus network using the SPADE (Smart Python Agent Development Environment) framework. The system simulates and manages bus routes, passenger flows, and handles real-time incidents in a public transportation network.

## 2. System Architecture

### 2.1 Core Components
- **Bus Agents**: Autonomous agents representing individual buses
- **Station Agents**: Manage passenger queues and bus arrivals at each station
- **Control Agent**: Central coordinator managing route assignments and incident responses
- **Dashboard**: Real-time visualization of system state
- **State Management System**: Centralized state tracking

### 2.2 Dependencies
```
- Python 3.8+
- SPADE (Smart Python Agent Development Environment)
- Pandas
- Streamlit
- AsyncIO
- Logging
- Pathlib
```

## 3. Detailed Component Analysis

### 3.1 Bus Agent System
Bus Agent System is responsible for route following with dynamic path adjustment, enabling efficient navigation even when unexpected events arise. It also supports incident reporting, ensuring timely updates on disruptions, while handling passenger capacity management to prevent overload. Additionally, the system shares real-time status updates, providing a continuous view of a bus’s location and operational condition. It features autonomous decision making for robust route planning, real-time communication with stations and the control center for effective coordination, dynamic route recalculation to respond to changing conditions, and the ability to address incidents such as traffic congestion or road closures.

### 3.2 Station Management
Station Management oversees passenger queue organization, ensuring smooth boarding processes and balanced distribution of passengers across available buses. It provides real-time tracking of bus arrivals, enabling precise forecasting of station wait times. This management also facilitates dynamic passenger allocation whenever routes change or disruptions occur, and continuously evaluates incident impacts to adapt passenger flows as needed. The system maintains essential details like waiting passengers per route and expected arrival times, ensuring timely station status updates.

### 3.3 Control Center
The Control Center coordinates route generation and assignment across the network, ensuring that all buses operate efficiently. It handles incident response by swiftly adapting routes in the event of traffic problems or road closures. Through real-time monitoring, the Control Center optimizes schedules, employs pathfinding (such as A*), recalculates routes, and keeps the network running smoothly despite evolving conditions.

### 3.4 State Management System
The State Management System centralizes data storage for buses and stations, ensuring that all components share the same view of the current operational state. It supports real-time synchronization to keep information consistent, applies a uniform data format, and integrates error handling to maintain system reliability. These measures allow for accurate, up-to-date insights into each agent’s status in the network.

### 3.5 Dashboard Implementation
The Dashboard leverages the Streamlit framework to display live data updates, offering straightforward visualization of ongoing incidents, bus status, and station activity. Interactive features allow users to explore the network in real time and examine passenger queues or bus schedules more closely. This consolidated view aids system supervisors in decision-making, as it provides immediate feedback on events occurring across the network.

## 4. Communication Protocol

### 4.1 Message Types
```
- REGISTER: Agent registration
- ROUTE_ASSIGNMENT: New route distribution
- INCIDENT: Problem reporting
- BUS_ARRIVED: Arrival notification
- NEW_PASSENGERS: Passenger updates
```

### 4.2 Message Format
```python
{
    'performative': str,
    'body': str,
    'metadata': dict
}
```

## 5. Time Management

### 5.1 System Time
- Simulated time system
- Configurable time multiplication
- Incident duration tracking

### 5.2 Time Ratios
```
- 1 real minute = 60 system hours
- Incident durations:
  - Light traffic: 6 hours
  - Heavy traffic: 12 hours
  - Road closure: 24 hours
```

## 6. Data Management

### 6.1 File Structure
```
/data
  /state
    - bus_states.csv
    - station_states.csv
/cities.csv
```

### 6.2 State Persistence
- CSV-based storage
- Real-time updates
- Atomic write operations

## 7. Error Handling

### 7.1 Implementation
- Comprehensive logging system
- Error recovery mechanisms
- State consistency checking

### 7.2 Logging Levels
```python
{
    'bus': 'BUS-%(bus_id)s',
    'station': 'STATION-%(station_id)s',
    'control': 'CONTROL',
    'messaging': 'MESSAGING'
}
```

## 8. Performance Considerations

### 8.1 Optimization Techniques
- Asynchronous operations
- Efficient state updates
- Minimal message overhead

### 8.2 Scalability Features
- Distributed agent architecture
- Independent state management
- Modular component design

## Implementation Details

### Agent Communication Patterns
1. **Registration Flow**
   - Buses and stations register with Control Agent
   - Control confirms registration with INIT_CONFIRM
   - System activates once all agents are registered

2. **Message Types & Protocols**
   ```python
   - REGISTER: {agent_id}
   - ROUTE_ASSIGNMENT: {route_data}
   - ROUTE_ACCEPTED: {bus_id}
   - INCIDENT: {bus_id}:{type}:{current}:{next}
   - BUS_ARRIVED: {station}:{route}:{passengers}
   - NEW_PASSENGERS: {station}:{route}:{count}
   ```

3. **Incident Management**
   - Light Traffic: 20% delay, 6h duration
   - Heavy Traffic: 70% delay, 12h duration
   - Road Closure: Infinite cost, 24h duration
   - Dynamic route recalculation using A* algorithm

4. **State Management**
   - Centralized state tracking via SystemState
   - CSV persistence for buses and stations
   - Real-time state updates and monitoring

5. **Time System**
   - Simulated time with configurable multiplier
   - Incident duration tracking
   - Synchronized agent operations

### Agent Behaviors

1. **Bus Agent**
   ```python
   Behaviors:
   - InitBehavior: Registration and setup
   - BusBehavior: Route following and incident handling
   - MessageHandler: Route updates and communication
   ```

2. **Station Agent**
   ```python
   Behaviors:
   - StationInitBehaviour: Registration
   - StationBehaviour: Passenger and arrival management
   ```

3. **Control Agent**
   ```python
   Behaviors:
   - ControlBehaviour: System coordination
   - Route Management: Generation and assignment
   - Incident Response: Network updates and rerouting
   ```
