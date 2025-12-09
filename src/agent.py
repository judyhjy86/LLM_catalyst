"""
Agent class for multiagent simulation
"""
import numpy as np
from typing import Dict, List, Any, Optional


class Agent:
    """
    Base agent class for the multiagent simulation.
    Each agent has its own state, beliefs, and decision-making process.
    """
    
    def __init__(self, agent_id: int, config: Dict[str, Any]):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary with agent parameters
        """
        self.id = agent_id
        self.config = config
        
        # Agent state
        self.position = np.array(config.get('initial_position', [0.0, 0.0]))
        self.velocity = np.array(config.get('initial_velocity', [0.0, 0.0]))
        
        # Latent cognitive state
        self.beliefs = config.get('initial_beliefs', {})
        self.memory = []
        self.goals = config.get('goals', [])
        
        # Agent properties
        self.max_speed = config.get('max_speed', 1.0)
        self.perception_radius = config.get('perception_radius', 5.0)
        self.energy = config.get('initial_energy', 100.0)
        
    def perceive(self, environment: Dict[str, Any], other_agents: List['Agent']) -> Dict[str, Any]:
        """
        Agent perceives the environment and nearby agents.
        
        Args:
            environment: Current environment state
            other_agents: List of other agents in the simulation
            
        Returns:
            Dictionary of perceived information
        """
        perception = {
            'nearby_agents': [],
            'environment_features': {}
        }
        
        # Detect nearby agents
        for agent in other_agents:
            distance = np.linalg.norm(self.position - agent.position)
            if distance <= self.perception_radius and agent.id != self.id:
                perception['nearby_agents'].append({
                    'id': agent.id,
                    'position': agent.position.copy(),
                    'distance': distance
                })
        
        # Perceive environment features (can be extended)
        perception['environment_features'] = environment.get('features', {})
        
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent makes a decision based on perception and internal state.
        This is where latent cognition processes occur.
        
        Args:
            perception: Information from perceive() method
            
        Returns:
            Dictionary with action decision
        """
        # Update beliefs based on perception
        self._update_beliefs(perception)
        
        # Simple decision-making (to be extended with more sophisticated cognition)
        action = self._select_action(perception)
        
        return action
    
    def _update_beliefs(self, perception: Dict[str, Any]):
        """Update internal beliefs based on new information."""
        # Store recent perception in memory
        self.memory.append(perception)
        if len(self.memory) > self.config.get('memory_size', 10):
            self.memory.pop(0)
        
        # Update beliefs (simplified - can be made more sophisticated)
        if perception['nearby_agents']:
            self.beliefs['has_neighbors'] = True
        else:
            self.beliefs['has_neighbors'] = False
    
    def _select_action(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action based on current state and beliefs.
        This is a simple implementation - can be extended with RL, planning, etc.
        """
        action = {
            'type': 'move',
            'direction': np.array([0.0, 0.0])
        }
        
        # Simple behavior: move towards goal or away from neighbors
        if self.goals:
            goal = np.array(self.goals[0])
            direction = goal - self.position
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            action['direction'] = direction
        
        # Adjust based on nearby agents (simple avoidance)
        if perception['nearby_agents']:
            avoidance = np.array([0.0, 0.0])
            for neighbor in perception['nearby_agents']:
                diff = self.position - neighbor['position']
                distance = neighbor['distance']
                avoidance += diff / (distance ** 2 + 1e-6)
            action['direction'] = 0.7 * action['direction'] + 0.3 * avoidance
            action['direction'] = action['direction'] / (np.linalg.norm(action['direction']) + 1e-6)
        
        return action
    
    def act(self, action: Dict[str, Any], dt: float = 1.0):
        """
        Execute the selected action.
        
        Args:
            action: Action dictionary from decide()
            dt: Time step size
        """
        if action['type'] == 'move':
            # Update velocity
            self.velocity = action['direction'] * self.max_speed
            
            # Update position
            self.position += self.velocity * dt
            
            # Consume energy
            self.energy -= self.config.get('energy_cost_per_step', 0.1)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the agent."""
        return {
            'id': self.id,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'energy': self.energy,
            'beliefs': self.beliefs.copy(),
            'num_memories': len(self.memory)
        }
    
    def step(self, environment: Dict[str, Any], other_agents: List['Agent'], dt: float = 1.0):
        """
        Complete agent step: perceive, decide, act.
        
        Args:
            environment: Current environment state
            other_agents: List of other agents
            dt: Time step size
        """
        perception = self.perceive(environment, other_agents)
        action = self.decide(perception)
        self.act(action, dt)

