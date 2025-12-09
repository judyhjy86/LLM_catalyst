"""
Simulation framework for multiagent system
"""
import numpy as np
from typing import List, Dict, Any
import json
import os
from datetime import datetime

from src.agent import Agent


class Simulation:
    """
    Main simulation class that manages agents and environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulation.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        sim_config = config.get('simulation', {})
        
        # Simulation parameters
        self.num_agents = sim_config.get('num_agents', 10)
        self.max_steps = sim_config.get('max_steps', 100)
        self.dt = sim_config.get('dt', 1.0)
        self.world_size = sim_config.get('world_size', [100.0, 100.0])
        
        # Initialize agents
        self.agents: List[Agent] = []
        self._initialize_agents()
        
        # Environment
        self.environment = {
            'features': {},
            'step': 0
        }
        
        # Results storage
        self.history = []
        
    def _initialize_agents(self):
        """Initialize all agents."""
        agent_config = self.config.get('agent', {})
        
        for i in range(self.num_agents):
            # Random initial positions
            initial_pos = np.array([
                np.random.uniform(0, self.world_size[0]),
                np.random.uniform(0, self.world_size[1])
            ])
            
            # Random goals
            goals = [
                np.array([
                    np.random.uniform(0, self.world_size[0]),
                    np.random.uniform(0, self.world_size[1])
                ])
            ]
            
            agent_cfg = agent_config.copy()
            agent_cfg['initial_position'] = initial_pos
            agent_cfg['goals'] = goals
            
            agent = Agent(i, agent_cfg)
            self.agents.append(agent)
    
    def step(self):
        """Execute one simulation step."""
        # Update environment
        self.environment['step'] += 1
        
        # Collect all agent states before updating
        agent_states = [agent.get_state() for agent in self.agents]
        
        # Each agent perceives, decides, and acts
        for agent in self.agents:
            other_agents = [a for a in self.agents if a.id != agent.id]
            agent.step(self.environment, other_agents, self.dt)
        
        # Apply boundary conditions
        self._apply_boundaries()
        
        # Record state
        self.history.append({
            'step': self.environment['step'],
            'agents': [agent.get_state() for agent in self.agents],
            'environment': self.environment.copy()
        })
    
    def _apply_boundaries(self):
        """Apply boundary conditions to keep agents in world."""
        for agent in self.agents:
            # Reflective boundaries
            for i in range(2):
                if agent.position[i] < 0:
                    agent.position[i] = 0
                    agent.velocity[i] *= -0.5
                elif agent.position[i] > self.world_size[i]:
                    agent.position[i] = self.world_size[i]
                    agent.velocity[i] *= -0.5
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the complete simulation.
        
        Returns:
            List of history records
        """
        print(f"Starting simulation with {self.num_agents} agents for {self.max_steps} steps...")
        
        for step in range(self.max_steps):
            self.step()
            
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{self.max_steps}")
        
        print("Simulation completed!")
        return self.history
    
    def save_results(self, output_dir: str):
        """
        Save simulation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full history as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(output_dir, f"history_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_history = convert_to_serializable(self.history)
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        # Save summary statistics
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
        summary = {
            'num_agents': self.num_agents,
            'num_steps': self.max_steps,
            'final_positions': [agent.get_state()['position'].tolist() for agent in self.agents],
            'final_energies': [agent.energy for agent in self.agents]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(convert_to_serializable(summary), f, indent=2)
        
        print(f"Results saved to {output_dir}")

