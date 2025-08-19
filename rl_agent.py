"""
PPO Reinforcement Learning Agent for Traffic Routing
Handles edge-by-edge decision making for bus routing
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict, Any
import logging

class PPONetwork(nn.Module):
    def __init__(self, global_state_size: int, action_state_size: int, hidden_size: int = 512):
        super(PPONetwork, self).__init__()
        
        # Global state processing - increased depth
        self.global_encoder = nn.Sequential(
            nn.Linear(global_state_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.ReLU()
        )
        
        # Action-specific state processing - increased depth
        self.action_encoder = nn.Sequential(
            nn.Linear(action_state_size, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4)
        )
        
        # Combined processing - increased depth
        combined_size = hidden_size//2 + hidden_size//4
        self.combined_layers = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU()
        )
        
        # Policy head (actor) - increased depth
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)  # Single action value per street option
        )
        
        # Value head (critic) - uses global features only, increased depth
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
        
    def forward(self, global_state: torch.Tensor, action_states: torch.Tensor):
        """
        Forward pass
        global_state: (batch_size, global_state_size)
        action_states: (batch_size, num_actions, action_state_size)
        """
        batch_size, num_actions, action_state_size = action_states.shape
        
        # Process global state
        global_features = self.global_encoder(global_state)  # (batch_size, hidden_size//2)
        
        # Process each action state
        action_features = self.action_encoder(action_states.view(-1, action_state_size))  # (batch_size * num_actions, hidden_size//4)
        action_features = action_features.view(batch_size, num_actions, -1)  # (batch_size, num_actions, hidden_size//4)
        
        # Combine global and action features for each action
        global_expanded = global_features.unsqueeze(1).expand(-1, num_actions, -1)  # (batch_size, num_actions, hidden_size//2)
        combined = torch.cat([global_expanded, action_features], dim=2)  # (batch_size, num_actions, combined_size)
        
        # Process combined features
        combined_flat = combined.view(-1, combined.shape[2])  # (batch_size * num_actions, combined_size)
        processed = self.combined_layers(combined_flat)  # (batch_size * num_actions, hidden_size)
        processed = processed.view(batch_size, num_actions, -1)  # (batch_size, num_actions, hidden_size)
        
        # Get policy logits for each action
        policy_flat = processed.view(-1, processed.shape[2])  # (batch_size * num_actions, hidden_size//2)
        action_logits = self.policy_head(policy_flat)  # (batch_size * num_actions, 1)
        action_logits = action_logits.view(batch_size, num_actions)  # (batch_size, num_actions)
        
        # Get state value from global features only (better for critic)
        state_value = self.value_head(global_features)  # (batch_size, 1)
        
        return action_logits, state_value

class PPOAgent:
    def __init__(self, global_state_size: int = 5, action_state_size: int = 7, 
                 learning_rate: float = 3e-4, gamma: float = 0.99, 
                 lambda_gae: float = 0.95, epsilon: float = 0.2,
                 value_coeff: float = 0.5, entropy_coeff: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = PPONetwork(global_state_size, action_state_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        
        # Experience buffer - now grouped by episodes
        self.episode_memory = {}  # bus_id -> episode data
        self.completed_episodes = []  # Store completed episodes for training
        
        # Logger
        self.logger = logging.getLogger('rl_agent')
        
        # Route tracking for enhanced reward calculation
        self.route_tracking = {}
        
        # Action logging
        self.action_logger = logging.getLogger('rl_actions')
        
    def start_episode(self, bus_id: int):
        """Start a new episode for a bus"""
        self.episode_memory[bus_id] = {
            'global_states': [],
            'action_states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'astar_indicators': []  # New: track if action is in A* path
        }
        
    def get_action(self, bus_id: int, global_state: np.ndarray, action_states: List[np.ndarray]) -> Tuple[int, float, float, bool]:
        """
        Select action based on current state
        Returns: (action_index, action_probability, value_estimate, is_astar_action)
        """
        if not action_states:
            return 0, 1.0, 0.0, False
            
        # Convert to tensors
        global_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(np.array(action_states)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, state_value = self.network(global_tensor, action_tensor)
            
            # Apply softmax to get probabilities
            action_probs = torch.softmax(action_logits, dim=1)
            
            # Sample action
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_prob = action_probs[0, action.item()].item()
            log_prob = action_dist.log_prob(action).item()
            value_estimate = state_value[0, 0].item()
            
            # Check if action is in A* path (index 6 in action_states)
            is_astar_action = action_states[action.item()][6] > 0.5 if len(action_states[action.item()]) > 6 else False
        
        # Store in current episode
        if bus_id in self.episode_memory:
            episode = self.episode_memory[bus_id]
            episode['global_states'].append(global_state.copy())
            episode['action_states'].append([act.copy() for act in action_states])
            episode['actions'].append(action.item())
            episode['values'].append(value_estimate)
            episode['log_probs'].append(log_prob)
            episode['astar_indicators'].append(is_astar_action)
        
        # Log detailed action information
        self.action_logger.info(f"Bus {bus_id} | State: {global_state} | Action: {action.item()} | "
                               f"Action_Prob: {action_prob:.4f} | Value_Est: {value_estimate:.4f} | "
                               f"A*_Action: {is_astar_action}")
            
        return action.item(), action_prob, value_estimate, is_astar_action
    
    def store_reward(self, bus_id: int, reward: float, done: bool = False):
        """Store reward for the last action taken by a bus"""
        if bus_id in self.episode_memory:
            episode = self.episode_memory[bus_id]
            episode['rewards'].append(reward)
            episode['dones'].append(done)
            
            if done:
                # Episode completed, move to training data
                self.complete_episode(bus_id)
    
    def complete_episode(self, bus_id: int):
        """Complete an episode and move it to training data"""
        if bus_id not in self.episode_memory:
            return
            
        episode = self.episode_memory[bus_id]
        
        # Only add if episode has experiences
        if len(episode['rewards']) > 0:
            # Calculate advantages and returns using proper GAE
            advantages, returns = self.compute_gae_for_episode(episode)
            episode['advantages'] = advantages
            episode['returns'] = returns
            
            # Add to completed episodes for training
            self.completed_episodes.append(episode.copy())
            
            self.logger.info(f"Episode completed for bus {bus_id} with {len(episode['rewards'])} steps")
        
        # Clear episode memory for this bus
        del self.episode_memory[bus_id]
    
    def compute_gae_for_episode(self, episode: Dict) -> Tuple[List[float], List[float]]:
        """Compute GAE for a single complete episode"""
        rewards = episode['rewards']
        values = episode['values']
        dones = episode['dones']
        
        advantages = []
        gae = 0
        
        # Process in reverse order
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                # Last step - next value is 0 if done, otherwise we need to estimate
                next_value = 0 if dones[i] else values[i]  # For terminal states
            else:
                # Use next step's value estimate
                next_value = values[i + 1]
                
            # TD error calculation
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            
            # GAE calculation
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        # Returns are advantages + values
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update_policy(self, batch_size: int = 64, epochs: int = 4):
        """Update policy using PPO algorithm with episode-based training"""
        if len(self.completed_episodes) == 0:
            return
        
        # Combine all completed episodes
        all_global_states = []
        all_action_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        for episode in self.completed_episodes:
            # Skip episodes with mismatched lengths
            min_len = min(len(episode['global_states']), len(episode['actions']), 
                         len(episode['advantages']), len(episode['log_probs']))
            
            if min_len == 0:
                continue
                
            # Truncate all lists to minimum length
            all_global_states.extend(episode['global_states'][:min_len])
            all_action_states.extend(episode['action_states'][:min_len])
            all_actions.extend(episode['actions'][:min_len])
            all_advantages.extend(episode['advantages'][:min_len])
            all_returns.extend(episode['returns'][:min_len])
            all_old_log_probs.extend(episode['log_probs'][:min_len])
        
        if len(all_global_states) < batch_size:
            return
        
        # Convert to tensors
        global_states = torch.FloatTensor(np.array(all_global_states)).to(self.device)
        
        # Handle action states with padding
        max_actions = max(len(acts) for acts in all_action_states)
        padded_action_states = []
        for acts in all_action_states:
            if len(acts) < max_actions:
                padding = np.zeros((max_actions - len(acts), len(acts[0]) if len(acts) > 0 else 7))
                acts = acts + [padding[i] for i in range(len(padding))]
            padded_action_states.append(acts[:max_actions])
        
        action_states = torch.FloatTensor(np.array(padded_action_states)).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss_sum = 0
        for epoch in range(epochs):
            # Get current policy
            action_logits, values = self.network(global_states, action_states)
            action_probs = torch.softmax(action_logits, dim=1)
            
            # Get log probabilities for taken actions
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8).squeeze()
            
            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Entropy loss
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
            entropy_loss = -self.entropy_coeff * entropy
            
            # Total loss
            total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss
            total_loss_sum += total_loss.item()
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear completed episodes after training
        num_episodes = len(self.completed_episodes)
        self.completed_episodes.clear()
        
        avg_loss = total_loss_sum / epochs
        self.logger.info(f"Policy updated using {num_episodes} episodes - Avg Loss: {avg_loss:.4f}")
    
    def old_compute_gae(self, rewards: List[float], values: List[float], 
                   next_values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Old GAE function - kept for reference but not used"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i] if not dones[i] else 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lambda_gae * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def start_route_tracking(self, bus_id: int, start_station_id: int, target_station_id: int, 
                           astar_path: List[str], astar_edges: List[str], astar_distance: float, astar_steps: int):
        """Initialize route tracking for a bus going from start to target station"""
        self.route_tracking[bus_id] = {
            'start_station_id': start_station_id,
            'target_station_id': target_station_id,
            'astar_path': astar_path,
            'astar_edges': set(astar_edges),  # Use set for O(1) lookup
            'astar_distance': astar_distance,
            'astar_steps': astar_steps,
            'rl_path': [],
            'rl_edges': [],
            'rl_distance': 0.0,
            'rl_steps': 0,
            'start_time': None,
            'completed': False
        }
    
    def update_route_tracking(self, bus_id: int, from_node: str, to_node: str, edge_distance: float, start_time: float = None):
        """Update route tracking with new step"""
        if bus_id not in self.route_tracking:
            return
        
        tracking = self.route_tracking[bus_id]
        if tracking['completed']:
            return
        
        # Set start time if this is the first step
        if start_time is not None and tracking['start_time'] is None:
            tracking['start_time'] = start_time
        
        # Update RL path
        if not tracking['rl_path'] or tracking['rl_path'][-1] != from_node:
            tracking['rl_path'].append(from_node)
        tracking['rl_path'].append(to_node)
        
        # Update RL edges and distance
        edge_id = f"{from_node}_{to_node}"
        tracking['rl_edges'].append(edge_id)
        tracking['rl_distance'] += edge_distance
        tracking['rl_steps'] += 1
    
    def complete_route_tracking(self, bus_id: int, end_time: float):
        """Mark route as completed and calculate final metrics"""
        if bus_id not in self.route_tracking:
            return
        
        tracking = self.route_tracking[bus_id]
        tracking['completed'] = True
        tracking['end_time'] = end_time
        tracking['total_time'] = end_time - (tracking['start_time'] or end_time)
    
    def is_edge_in_astar_path(self, bus_id: int, edge_id: str) -> bool:
        """Check if an edge is part of the A* optimal path"""
        if bus_id not in self.route_tracking:
            return False
        return edge_id in self.route_tracking[bus_id]['astar_edges']
    
    def calculate_enhanced_reward(self, bus_id: int, from_node: str, to_node: str, 
                                edge_distance: float, reached_target: bool = False, 
                                current_time: float = None, prev_distance_to_goal: float = None,
                                new_distance_to_goal: float = None) -> float:
        """
        Calculate enhanced reward with potential-based reward shaping:
        - Large reward for reaching target
        - Potential-based shaping reward
        - A* edge bonus for using optimal edges
        - Step penalty (time-based if time provided)
        """
        if bus_id not in self.route_tracking:
            return -1.0  # Default penalty if no tracking
        
        tracking = self.route_tracking[bus_id]
        
        # 1. Large reward for reaching target station
        target_reward = 0.0
        if reached_target:
            target_reward = 200.0  # Large completion reward
        
        # 2. Potential-based reward shaping
        # Potential function: negative distance to goal (closer = higher potential)
        shaping_reward = 0.0
        if prev_distance_to_goal is not None and new_distance_to_goal is not None:
            # Φ(s') - Φ(s) where Φ(s) = -distance_to_goal
            prev_potential = -prev_distance_to_goal
            new_potential = -new_distance_to_goal
            shaping_reward = self.gamma * new_potential - prev_potential
            # Scale the shaping reward
            shaping_reward *= 0.1  # Scale factor to prevent overwhelming other rewards
        
        # 3. A* edge bonus - reward for using edges that are in the A* path
        astar_edge_bonus = 0.0
        edge_id = f"{from_node}_{to_node}"
        if self.is_edge_in_astar_path(bus_id, edge_id):
            astar_edge_bonus = 5.0  # Bonus for using A* edges
        
        # 4. Step penalty - time-based if available, otherwise fixed
        step_penalty = -1.0  # Base step penalty
        if current_time is not None and tracking.get('start_time') is not None:
            # Time-based penalty - increases with elapsed time
            elapsed_time = current_time - tracking['start_time']
            time_penalty = -elapsed_time * 0.5  # Penalty increases with time
            step_penalty = min(step_penalty, time_penalty)  # Use worse penalty
        
        # 5. Small novelty bonus for exploration (reduced to prevent over-exploration)
        novelty_bonus = 0.0
        if to_node not in tracking['rl_path']:
            novelty_bonus = 0.5  # Small bonus for exploring new nodes
        
        total_reward = (target_reward + shaping_reward + astar_edge_bonus + 
                       step_penalty + novelty_bonus)
        
        return total_reward
    
    def get_route_stats(self, bus_id: int) -> Dict[str, Any]:
        """Get current route statistics for logging"""
        if bus_id not in self.route_tracking:
            return {}
        
        tracking = self.route_tracking[bus_id]
        return {
            'rl_steps': tracking['rl_steps'],
            'astar_steps': tracking['astar_steps'],
            'rl_distance': tracking['rl_distance'],
            'astar_distance': tracking['astar_distance'],
            'distance_ratio': tracking['rl_distance'] / tracking['astar_distance'] if tracking['astar_distance'] > 0 else 0,
            'elapsed_time': tracking.get('end_time', 0) - tracking['start_time'] if tracking['start_time'] else 0
        }
    
    def cleanup_route_tracking(self, bus_id: int):
        """Clean up completed route tracking"""
        if bus_id in self.route_tracking:
            del self.route_tracking[bus_id]
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
