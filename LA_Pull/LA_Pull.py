"""
Algorithm 5: LA-PULL with myopic threshold (simplified: predictor only at t=1)

Model:
- Complete graph with n nodes
- Synchronous rounds
- 1 random initial infected node
- Stop when all nodes are infected

LA-Pull (skip L4-L6):
- At t=1: each uninformed node pulls from predictor P(u)
  P has accuracy p:
    - prob p: returns random infected node (or fallback to Unif(V\\{u}))
    - prob 1-p: returns random uninformed node (or fallback to Unif(V\\{u}))
- For t>=2: each uninformed node pulls w ~ Unif(V\\{u})
- Synchronous activation at end of each round
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time
from tqdm import tqdm

# Parameters
N = 50000          # Number of nodes
TRIALS = 20        # Number of trials per p value
P_VALUES = np.arange(0.0, 1.1, 0.1)  # Predictor accuracy sweep
BASE_SEED = 42     # Base random seed


def predictor(u, infected, uninformed, p, rng):
    """
    Predictor P(u) with accuracy p.
    
    Args:
        u: current uninformed node
        infected: set of infected nodes
        uninformed: set of uninformed nodes
        p: accuracy parameter
        rng: random number generator
    
    Returns:
        Node to pull from
    """
    if rng.random() < p:
        # With probability p: return a random infected node
        if len(infected) > 0:
            return rng.choice(list(infected))
        else:
            # Fallback to uniform from V\{u}
            candidates = [v for v in range(N) if v != u]
            return rng.choice(candidates)
    else:
        # With probability 1-p: return a random uninformed node
        uninformed_others = uninformed - {u}
        if len(uninformed_others) > 0:
            return rng.choice(list(uninformed_others))
        else:
            # Fallback to uniform from V\{u}
            candidates = [v for v in range(N) if v != u]
            return rng.choice(candidates)


def simulate_la_pull(n, p, seed):
    """
    Simulate LA-Pull with predictor used only at t=1 (optimized version).
    
    Args:
        n: number of nodes
        p: predictor accuracy
        seed: random seed
    
    Returns:
        Number of rounds to infect all nodes
    """
    rng = np.random.RandomState(seed)
    
    # Initialize: one random infected node
    initial_infected = rng.randint(0, n)
    infected = np.zeros(n, dtype=bool)
    infected[initial_infected] = True
    num_infected = 1
    
    t = 0
    
    while num_infected < n:
        t += 1
        newly_infected = np.zeros(n, dtype=bool)
        uninformed_idx = np.where(~infected)[0]
        
        if t == 1:
            # Round 1: Use predictor for each uninformed node
            # With prob p: pull from infected; with prob 1-p: pull from uninformed
            use_predictor = rng.random(len(uninformed_idx)) < p
            
            for i, u in enumerate(uninformed_idx):
                if use_predictor[i]:
                    # Pull from infected nodes (if any exist)
                    if num_infected > 0:
                        infected_idx = np.where(infected)[0]
                        w = rng.choice(infected_idx)
                    else:
                        # Fallback: uniform from V\{u}
                        w = rng.choice([v for v in range(n) if v != u])
                else:
                    # Pull from uninformed (excluding self)
                    other_uninformed = uninformed_idx[uninformed_idx != u]
                    if len(other_uninformed) > 0:
                        w = rng.choice(other_uninformed)
                    else:
                        # Fallback: uniform from V\{u}
                        w = rng.choice([v for v in range(n) if v != u])
                
                if infected[w]:
                    newly_infected[u] = True
        else:
            # Round t >= 2: Uniform pull from V\{u}
            # For each uninformed, pull uniformly - probability of pulling infected is num_infected/(n-1)
            # Vectorized: decide who gets infected
            pull_infected_prob = num_infected / (n - 1)
            newly_infected[uninformed_idx] = rng.random(len(uninformed_idx)) < pull_infected_prob
        
        # Synchronous activation
        infected |= newly_infected
        num_infected = np.sum(infected)
    
    return t


def main():
    """Run experiments and generate plots and data."""
    print("=" * 70)
    print("LA-PULL Simulation (Algorithm 5 - Predictor at t=1 only)")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  n = {N:,} nodes")
    print(f"  Trials = {TRIALS}")
    print(f"  p values: {P_VALUES[0]:.1f} to {P_VALUES[-1]:.1f}")
    print(f"  Base seed = {BASE_SEED}")
    print(f"  Total simulations: {len(P_VALUES) * TRIALS}")
    print("=" * 70)
    print()
    
    overall_start = time.time()
    results = []
    
    # Calculate total iterations
    total_sims = len(P_VALUES) * TRIALS
    completed_sims = 0
    
    # Progress bar for overall progress with time estimation
    with tqdm(total=total_sims, desc="Running simulations", 
              unit="sim", ncols=100, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for p_idx, p in enumerate(P_VALUES):
            start_time = time.time()
            rounds = []
            
            for trial in range(TRIALS):
                # Generate unique seed for each (p, trial) combination
                seed = BASE_SEED + int(p * 10) * 1000 + trial
                r = simulate_la_pull(N, p, seed)
                rounds.append(r)
                
                # Update progress
                completed_sims += 1
                pbar.update(1)
                pbar.set_description(f"p={p:.1f} (trial {trial+1}/{TRIALS})")
            
            rounds = np.array(rounds)
            mean_rounds = np.mean(rounds)
            median_rounds = np.median(rounds)
            p90_rounds = np.percentile(rounds, 90)
            
            # 95% confidence interval using t-distribution
            ci = stats.t.interval(0.95, len(rounds) - 1,
                                  loc=mean_rounds,
                                  scale=stats.sem(rounds))
            
            results.append({
                'p': p,
                'mean_rounds': mean_rounds,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'median': median_rounds,
                'p90': p90_rounds
            })
            
            elapsed = time.time() - start_time
            tqdm.write(f"[OK] p={p:.1f}: mean={mean_rounds:.2f}, median={median_rounds:.0f}, "
                       f"p90={p90_rounds:.0f} ({elapsed:.1f}s)")
    
    total_time = time.time() - overall_start
    print()
    print("=" * 70)
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 70)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_filename = 'la_pull_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['p'], df['mean_rounds'], 'o-', linewidth=2, markersize=8,
            color='#2E86AB', label='Mean rounds')
    ax.fill_between(df['p'], df['ci_low'], df['ci_high'],
                     alpha=0.3, color='#2E86AB', label='95% CI')
    
    ax.set_xlabel('Predictor Accuracy (p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rounds to Infect All', fontsize=12, fontweight='bold')
    ax.set_title('LA-Pull (t=1 uses P only) — n=50k, T=20',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(-0.05, 1.05)
    
    # Format ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save figures
    png_filename = 'la_pull_plot.png'
    pdf_filename = 'la_pull_plot.pdf'
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"Plot saved to {png_filename} and {pdf_filename}")
    
    print()
    print("Summary Table:")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("=" * 70)
    
    plt.show()


if __name__ == '__main__':
    main()
