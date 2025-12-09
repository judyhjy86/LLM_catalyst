import mesa
import random
import matplotlib.pyplot as plt

# --- HELPER FUNCTIONS ---

def calculate_jaccard_distance(schema_a, schema_b):
    """
    Calculates cognitive distance based on the Jaccard Index.
    Paper definition: 1 - (Intersection / Union).
    """
    intersection = sum([1 for i, j in zip(schema_a, schema_b) if i == j and i == 1])
    union = sum([1 for i, j in zip(schema_a, schema_b) if i == 1 or j == 1])
    
    if union == 0:
        return 0 # Avoid division by zero; treat as identical
    
    jaccard_index = intersection / union
    return 1 - jaccard_index  # Distance is the inverse of similarity

# --- AGENT CLASS ---

class CatalystAgent(mesa.Agent):
    def __init__(self, unique_id, model, stance, schema_vector):
        super().__init__(unique_id, model)
        self.stance = stance            # Expressed Cognition (0 or 1)
        self.schema = schema_vector     # Latent Cognition (List of bits)
    
    def step(self):
        # 1. Select a random neighbor (or any random agent if fully connected)
        # For this 'Consensus Race', we assume a fully connected network (like a meeting room)
        # Convert AgentSet to list for random.choice
        agents_list = list(self.model.schedule.agents)
        other_agent = self.random.choice(agents_list)
        
        # 2. Check if Stances Differ (Disagreement)
        if self.stance != other_agent.stance:
            
            # 3. Calculate Latent Cognitive Distance
            dist = calculate_jaccard_distance(self.schema, other_agent.schema)
            
            # 4. Calculate Probability of Changing Mind
            # Base probability of influence
            prob_change = 0.1 
            
            if not self.model.observability:
                # --- CONDITION: OBSCURED ---
                # Hypothesis 1: Novelty drives influence.
                # Higher distance -> Higher probability of influence [cite: 43, 201]
                prob_change += (dist * 0.4) 
                
            else:
                # --- CONDITION: OBSERVABLE ---
                # Hypothesis 2: Similarity-Attraction bias.
                # If they are too different, we ignore them.
                # If they are similar, we might listen (but the argument is less novel).
                if dist > 0.5:
                     # Penalize influence from dissimilar sources [cite: 74, 217]
                    prob_change = 0.0
                else:
                    # Slight boost for similar sources, but less than novelty bonus
                    prob_change += 0.1 

            # 5. Attempt to Change Stance
            if self.random.random() < prob_change:
                self.stance = other_agent.stance # Copy the other agent's stance

# --- MODEL CLASS ---

class ConsensusModel(mesa.Model):
    def __init__(self, N, observability=False, schema_length=10):
        self.num_agents = N
        self.observability = observability # The main toggle
        self.schedule = mesa.time.RandomActivation(self)
        # Don't use self.agents - Mesa's Model.agents is an AgentSet property
        # We'll use self.schedule.agents instead
        
        # Initialize Agents
        for i in range(self.num_agents):
            
            # Random Stance (Binary: 0 or 1)
            start_stance = random.choice([0, 1])
            
            # Random Schema (Latent Cognition)
            # A random bitstring, e.g., [0, 1, 1, 0...]
            start_schema = [random.choice([0, 1]) for _ in range(schema_length)]
            
            a = CatalystAgent(i, self, start_stance, start_schema)
            self.schedule.add(a)
            # Agents are automatically added to self.schedule.agents (AgentSet)
            
        # Data Collector: Tracks the % of agents holding Stance 1
        self.datacollector = mesa.DataCollector(
            model_reporters={"Consensus": compute_consensus}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# --- METRIC FUNCTION ---

def compute_consensus(model):
    # Returns the percentage of agents holding Stance 1
    # Perfect consensus is either 0.0 or 1.0
    # Convert AgentSet to list for iteration
    agents_list = list(model.schedule.agents)
    stances = [a.stance for a in agents_list]
    return sum(stances) / len(stances) if len(stances) > 0 else 0.0

# --- EXECUTION & PLOTTING ---

# Run 1: Obscured (The "Hidden Catalyst" Scenario)
print("Running Simulation 1: Obscured Differences...")
model_obscured = ConsensusModel(N=100, observability=False)
for i in range(100):
    model_obscured.step()

# Run 2: Observable (The "Identity Bias" Scenario)
print("Running Simulation 2: Observable Differences...")
model_observable = ConsensusModel(N=100, observability=True)
for i in range(100):
    model_observable.step()

# Extract Data
results_obscured = model_obscured.datacollector.get_model_vars_dataframe()
results_observable = model_observable.datacollector.get_model_vars_dataframe()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(results_obscured["Consensus"], label="Obscured (Novelty Driven)", color="green", linewidth=2)
plt.plot(results_observable["Consensus"], label="Observable (Identity Driven)", color="red", linestyle="--")
plt.title("The Hidden Catalyst: How Visibility of Differences Affects Consensus")
plt.xlabel("Time Steps")
plt.ylabel("Consensus (% holding Stance 1)")
plt.legend()
plt.grid(True)
plt.show()