# OB639 Final Project: Latent Cognition Multiagent Simulation

A multiagent simulation framework for studying latent cognition in agent-based systems.

## Project Structure

```
OB639_Final_Latent_Cognition/
├── main.py                 # Main entry point
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py
│   ├── agent.py          # Agent class with latent cognition
│   ├── simulation.py     # Simulation framework
│   └── config.py         # Configuration management
├── data/                  # Data files (created during runs)
└── results/               # Simulation results (created during runs)
```

## Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

Basic usage:
```bash
python main.py
```

With custom parameters:
```bash
python main.py --num-agents 50 --steps 200 --output results/experiment1
```

### Configuration

Edit `config.yaml` to customize:
- Number of agents
- Simulation duration
- Agent properties (speed, perception, energy)
- World size

## Key Components

### Agent (`src/agent.py`)
- **Perception**: Agents perceive nearby agents and environment
- **Decision-making**: Latent cognitive processes (beliefs, memory)
- **Action**: Agents act based on their decisions

### Simulation (`src/simulation.py`)
- Manages multiple agents
- Tracks simulation history
- Saves results to JSON files

## Extending the Simulation

### Adding New Agent Behaviors
Modify the `Agent.decide()` method in `src/agent.py` to implement more sophisticated decision-making.

### Adding Environment Features
Extend the `environment` dictionary in `Simulation` to include more complex environmental factors.

### Visualization
Add visualization using matplotlib to plot agent trajectories and interactions.

## Results

Simulation results are saved in the `results/` directory:
- `history_*.json`: Complete simulation history
- `summary_*.json`: Summary statistics

