"""
Configuration management
"""
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}. Using default configuration.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'simulation': {
            'num_agents': 20,
            'max_steps': 100,
            'dt': 1.0,
            'world_size': [100.0, 100.0]
        },
        'agent': {
            'max_speed': 2.0,
            'perception_radius': 10.0,
            'initial_energy': 100.0,
            'energy_cost_per_step': 0.1,
            'memory_size': 10,
            'initial_beliefs': {}
        }
    }

