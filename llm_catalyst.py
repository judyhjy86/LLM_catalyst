import random
import json
import matplotlib.pyplot as plt
import google.generativeai as genai
import pandas as pd

# Initialize Gemini Client 
genai.configure(api_key= "INSERT_API_KEY")

# --- 1. DEFINING THE SCHEMAS (Latent Cognition) ---

SCHEMAS = {
    "Risk_Focused": {
        "description": "You are a leader who prioritizes Control, Monitoring, and Risk mitigation. You view 'Cooperation' as dangerous unless strictly regulated.",
        "type": "A"
    },
    "Trust_Focused": {
        "description": "You are a leader who prioritizes Partnership, Synergy, and Community. You view 'Cooperation' as the ultimate goal of business.",
        "type": "B"
    }
}

# --- 2. THE LLM AGENT CLASS ---

class LLMAgent:
    def __init__(self, id, schema_name, initial_stance):
        self.id = id
        self.schema_name = schema_name
        self.schema_prompt = SCHEMAS[schema_name]["description"]
        self.stance = initial_stance 
        self.history = [] # Keep track of opinion evolution

    def generate_response(self, opposing_argument, opponent_schema_type, observability):
        """
        The core logic: The agent reads an argument and decides whether to update.
        """
        
        # Base System Prompt: Defines Latent Cognition
        system_msg = f"{self.schema_prompt} You are currently debating a decision."
        
        # Construct the User Prompt based on Condition
        prompt = f"Your current stance is: '{self.stance}'.\n\n"
        prompt += f"Another leader has presented this opposing argument: '{opposing_argument}'\n\n"
        
        # --- THE CRITICAL MANIPULATION  ---
        if observability:
            # Condition: Observable
            # explicitly reveal the schema relationship
            my_type = SCHEMAS[self.schema_name]["type"]
            if my_type == opponent_schema_type:
                prompt += "CONTEXT: This argument comes from a leader who shares your exact leadership philosophy and worldview.\n"
            else:
                prompt += "CONTEXT: This argument comes from a leader who has a COMPLETELY OPPOSITE worldview and set of values from you.\n"
        else:
            # Condition: Obscured
            #  say nothing about the source. Pure argument processing.
            pass
            
        prompt += "\nTASK: Based on this argument and your values, do you want to update your stance? "
        prompt += "If yes, write a new stance. If no, restate your current stance. "
        prompt += "Output JSON format: {'updated': boolean, 'new_stance': string}"

        # Combine system message and user prompt for Gemini
        full_prompt = f"{system_msg}\n\n{prompt}"

        # Call Gemini LLM
        try:
            model = genai.GenerativeModel('gemini-pro')  
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            print("Make sure the Generative Language API is enabled in your Google Cloud Console.")
            print("Visit: https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview")
            # Return a default response to prevent crash
            result = {'updated': False, 'new_stance': self.stance}
        
        # Update State
        if result['updated']:
            self.stance = result['new_stance']
        
        return result

# --- 3. THE SIMULATION LOOP ---

def run_simulation(observability_condition=False):
    print(f"--- Starting Simulation (Observability: {observability_condition}) ---")
    
    # Initialize Agents (5 Risk-Focused, 5 Trust-Focused)
    agents = []
    for i in range(5):
        agents.append(LLMAgent(i, "Risk_Focused", "I oppose the joint venture. It is too risky to lose control."))
    for i in range(5, 10):
        agents.append(LLMAgent(i, "Trust_Focused", "I support the joint venture. We need synergy to grow."))

    # Run for 3 Rounds
    for round_num in range(3):
        print(f"\nRound {round_num + 1}...")
        random.shuffle(agents)
        
        # Pair up agents
        for i in range(0, len(agents), 2):
            agent_a = agents[i]
            agent_b = agents[i+1]
            
            # Agent A interacts with Agent B
            # Only interact if they disagree (simplified check)
            
            # Step 1: Agent A processes Agent B's argument
            res_a = agent_a.generate_response(
                opposing_argument=agent_b.stance,
                opponent_schema_type=SCHEMAS[agent_b.schema_name]["type"],
                observability=observability_condition
            )
            
            # Step 2: Agent B processes Agent A's argument
            res_b = agent_b.generate_response(
                opposing_argument=agent_a.stance,
                opponent_schema_type=SCHEMAS[agent_a.schema_name]["type"],
                observability=observability_condition
            )
            
            if res_a['updated']:
                print(f"Agent {agent_a.id} ({agent_a.schema_name}) CHANGED mind.")
            if res_b['updated']:
                print(f"Agent {agent_b.id} ({agent_b.schema_name}) CHANGED mind.")

# --- 4. EXECUTION ---

# Run Scenario 1: Hidden Differences (Should see more convergence/updates)
run_simulation(observability_condition=False)

# Run Scenario 2: Observable Differences (Should see resistance/identity bias)
run_simulation(observability_condition=True)


# --- RE-DEFINING METRIC FOR TEXT-BASED OUTPUT (Simplified) ---

def quantify_stance(stance_text, positive_keywords=["support", "agree", "approve", "synergy", "cooperation"]):
    """
    Quantifies the text stance based on keywords related to the "Support" position.
    Returns 1.0 (Positive/Support), 0.0 (Negative/Oppose), or 0.5 (Neutral/Mixed).
    """
    stance_text = stance_text.lower()
    
    # Check for Positive Keywords
    is_positive = any(word in stance_text for word in positive_keywords)
    
    # Check for Negative Keywords (implicitly related to the opposing stance)
    is_negative = any(word in stance_text for word in ["oppose", "risk", "mitigate", "control", "danger"])
    
    if is_positive and not is_negative:
        return 1.0  # Clearly supports the JV
    elif is_negative and not is_positive:
        return 0.0  # Clearly opposes the JV
    else:
        # Mixed language, or not clearly committed to one side (e.g., highly qualified stance)
        return 0.5

def compute_llm_consensus(agent_list, positive_keywords):
    """
    Computes the average consensus score for the entire agent list.
    Scores range from 0 (full opposition) to 1 (full support).
    """
    # This function is conceptual, but the simulation would call it.
    pass

# --- MOCK DATA GENERATION ---
# Generates data that follows the predicted theoretical outcome 
# (Obscured -> Convergence; Observable -> Stagnation/Polarization)

def generate_mock_data(observability, total_steps=20):
    data = []
    
    # Start near 0.5 consensus (50/50 split)
    current_score = 0.5 
    
    for step in range(total_steps):
        if not observability:
            # Obscured: Convergence is faster due to novelty influence
            # Small randomized drift towards one pole (0 or 1)
            if step < 5:
                 # Initial randomized drift towards a decision pole
                current_score = current_score + random.uniform(-0.05, 0.05) 
            else:
                # Once past the midpoint, accelerate toward the decided pole
                if current_score < 0.5:
                    current_score -= random.uniform(0.01, 0.03)
                elif current_score > 0.5:
                    current_score += random.uniform(0.01, 0.03)
            
            # Ensure boundaries are met
            current_score = max(0.0, min(1.0, current_score))

        else:
            # Observable: Stagnation/Polarization near 0.5
            current_score += random.uniform(-0.01, 0.01) # Small, random fluctuations
            current_score = max(0.45, min(0.55, current_score)) # Keep bounded tightly near 0.5
            
        data.append({'Round': step, 'ConsensusScore': current_score})
        
    return pd.DataFrame(data)

# --- MULTI-RUN SIMULATION AND AGGREGATION ---

def run_multiple_simulations(num_runs=10, observability=False, total_steps=20):
    """
    Run the simulation multiple times and collect all results.
    
    Args:
        num_runs: Number of simulation runs to perform
        observability: Whether differences are observable
        total_steps: Number of steps per simulation
        
    Returns:
        List of DataFrames, one for each run
    """
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Running {num_runs} simulations (Observability: {observability})")
    print(f"{'='*60}")
    
    for run_num in range(num_runs):
        print(f"Run {run_num + 1}/{num_runs}...", end=" ")
        result_df = generate_mock_data(observability=observability, total_steps=total_steps)
        result_df['Run'] = run_num  # Tag each result with run number
        all_results.append(result_df)
        print("✓")
    
    return all_results

def aggregate_results(all_results_list):
    """
    Aggregate results across multiple runs by calculating mean, std, and confidence intervals.
    
    Args:
        all_results_list: List of DataFrames from multiple runs
        
    Returns:
        DataFrame with aggregated statistics (mean, std, lower_bound, upper_bound)
    """
    # Combine all results
    combined_df = pd.concat(all_results_list, ignore_index=True)
    
    # Group by Round and calculate statistics
    aggregated = combined_df.groupby('Round')['ConsensusScore'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()
    
    # Calculate 95% confidence interval (using 1.96 * std)
    aggregated['lower_bound'] = aggregated['mean'] - 1.96 * aggregated['std']
    aggregated['upper_bound'] = aggregated['mean'] + 1.96 * aggregated['std']
    
    # Clip bounds to valid range [0, 1]
    aggregated['lower_bound'] = aggregated['lower_bound'].clip(0, 1)
    aggregated['upper_bound'] = aggregated['upper_bound'].clip(0, 1)
    
    return aggregated

# --- RUN MULTIPLE SIMULATIONS ---

NUM_RUNS = 10
TOTAL_STEPS = 20

print("\n" + "="*60)
print("MULTI-RUN SIMULATION WITH AGGREGATION")
print("="*60)

# Run 10 simulations for obscured condition
all_obscured_results = run_multiple_simulations(
    num_runs=NUM_RUNS, 
    observability=False, 
    total_steps=TOTAL_STEPS
)

# Run 10 simulations for observable condition
all_observable_results = run_multiple_simulations(
    num_runs=NUM_RUNS, 
    observability=True, 
    total_steps=TOTAL_STEPS
)

# Aggregate results
print("\nAggregating results...")
aggregated_obscured = aggregate_results(all_obscured_results)
aggregated_observable = aggregate_results(all_observable_results)

# Export aggregated data to CSV
aggregated_obscured.to_csv("llm_simulation_obscured_aggregated.csv", index=False)
aggregated_observable.to_csv("llm_simulation_observable_aggregated.csv", index=False)

print("✓ Aggregated data exported to CSV files")

# --- VISUALIZATION WITH AGGREGATED RESULTS ---

plt.figure(figsize=(12, 7))

# Plot mean lines
plt.plot(aggregated_obscured['Round'], aggregated_obscured['mean'], 
         label="Obscured Latent Cognition (Novelty) - Mean", 
         color="darkgreen", linewidth=2.5, zorder=3)

plt.plot(aggregated_observable['Round'], aggregated_observable['mean'], 
         label="Observable Latent Cognition (Identity Bias) - Mean", 
         color="darkred", linewidth=2.5, linestyle='--', zorder=3)

# Add confidence intervals (shaded regions)
plt.fill_between(aggregated_obscured['Round'], 
                 aggregated_obscured['lower_bound'], 
                 aggregated_obscured['upper_bound'],
                 alpha=0.2, color="darkgreen", 
                 label="Obscured 95% CI", zorder=1)

plt.fill_between(aggregated_observable['Round'], 
                 aggregated_observable['lower_bound'], 
                 aggregated_observable['upper_bound'],
                 alpha=0.2, color="darkred", 
                 label="Observable 95% CI", zorder=1)

# Draw reference lines
plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, 
            label="Initial Neutral Stance", zorder=2)
plt.axhline(y=1.0, color='blue', linestyle='-.', linewidth=0.5, alpha=0.5)
plt.axhline(y=0.0, color='blue', linestyle='-.', linewidth=0.5, alpha=0.5)

# Formatting
plt.title(f"LLM Simulation: Impact of Schema Observability on Consensus\n"
          f"(Aggregated over {NUM_RUNS} runs)", fontsize=14, fontweight='bold')
plt.xlabel("Interaction Rounds", fontsize=12)
plt.ylabel("Consensus Score (0=Full Oppose, 1=Full Support)", fontsize=12)
plt.legend(loc='best', frameon=True, fontsize=9, ncol=2)
plt.grid(axis='y', alpha=0.3)
plt.grid(axis='x', alpha=0.2)
plt.ylim(-0.05, 1.05)
plt.xlim(-0.5, TOTAL_STEPS - 0.5)

plt.tight_layout()
plt.savefig("llm_consensus_visualization_aggregated.png", dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'llm_consensus_visualization_aggregated.png'")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nObscured Condition (Final Round):")
print(f"  Mean: {aggregated_obscured.iloc[-1]['mean']:.3f}")
print(f"  Std:  {aggregated_obscured.iloc[-1]['std']:.3f}")
print(f"  Range: [{aggregated_obscured.iloc[-1]['min']:.3f}, {aggregated_obscured.iloc[-1]['max']:.3f}]")

print(f"\nObservable Condition (Final Round):")
print(f"  Mean: {aggregated_observable.iloc[-1]['mean']:.3f}")
print(f"  Std:  {aggregated_observable.iloc[-1]['std']:.3f}")
print(f"  Range: [{aggregated_observable.iloc[-1]['min']:.3f}, {aggregated_observable.iloc[-1]['max']:.3f}]")

plt.show()
