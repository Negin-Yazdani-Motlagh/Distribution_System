#!/usr/bin/env python3
"""
LA-PULL vs Uniform PULL Comparison

Creates a single plot with 6 curves:
- LA-Pull (uses predictor P at t=1) for n = 10k, 25k, 50k (curves that vary with p)
- Uniform Pull (no predictor, always uniform) for n = 10k, 25k, 50k (horizontal lines)

Y-axis: Round Complexity (rounds to infect all nodes)
X-axis: Predictor Accuracy (p)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Parameters
N_VALUES = [10_000, 25_000, 50_000]  # Network sizes
TRIALS = 20                           # Number of trials per (n, p) pair
P_VALUES = np.arange(0.0, 1.1, 0.1)   # Predictor accuracy sweep [0.0, 0.1, ..., 1.0]
BASE_SEED = 42                        # Base random seed


def simulate_la_pull(n, p, seed):
    r"""
    Simulate LA-Pull algorithm (predictor P used only at t=1).
    
    Algorithm:
    - t=1: Each uninformed node uses predictor P(u)
           With prob p: pull from random infected node
           With prob 1-p: pull from random uninformed node
    - t>=2: Each uninformed node pulls uniformly from V\{u}
    
    Args:
        n: number of nodes
        p: predictor accuracy [0, 1]
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
            # Round 1: Use predictor P for each uninformed node
            use_predictor = rng.random(len(uninformed_idx)) < p
            
            for i, u in enumerate(uninformed_idx):
                if use_predictor[i]:
                    # Good prediction (prob p): pull from infected node
                    if num_infected > 0:
                        infected_idx = np.where(infected)[0]
                        w = rng.choice(infected_idx)
                    else:
                        # Fallback: uniform from V\{u}
                        w = rng.choice([v for v in range(n) if v != u])
                else:
                    # Bad prediction (prob 1-p): pull from uninformed node
                    other_uninformed = uninformed_idx[uninformed_idx != u]
                    if len(other_uninformed) > 0:
                        w = rng.choice(other_uninformed)
                    else:
                        # Fallback: uniform from V\{u}
                        w = rng.choice([v for v in range(n) if v != u])
                
                # Check if pulled node is infected
                if infected[w]:
                    newly_infected[u] = True
        else:
            # Round t >= 2: Uniform pull from V\{u}
            # Each uninformed node has probability num_infected/(n-1) of pulling an infected node
            pull_infected_prob = num_infected / (n - 1)
            newly_infected[uninformed_idx] = rng.random(len(uninformed_idx)) < pull_infected_prob
        
        # Synchronous activation
        infected |= newly_infected
        num_infected = np.sum(infected)
    
    return t


def simulate_uniform_pull(n, seed):
    r"""
    Simulate uniform Pull (no predictor, always uniform random pull).
    
    Algorithm:
    - Every round: Each uninformed node pulls uniformly from V\{u}
    
    Args:
        n: number of nodes
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
        
        # Every round: Uniform pull from V\{u}
        # Each uninformed node has probability num_infected/(n-1) of pulling an infected node
        pull_infected_prob = num_infected / (n - 1)
        newly_infected[uninformed_idx] = rng.random(len(uninformed_idx)) < pull_infected_prob
        
        # Synchronous activation
        infected |= newly_infected
        num_infected = np.sum(infected)
    
    return t


def run_single_la_pull(args):
    """Helper function for parallel LA-Pull execution."""
    n, p, trial = args
    seed = BASE_SEED + n + int(p * 100) * 1000 + trial
    return simulate_la_pull(n, p, seed)


def run_single_uniform_pull(args):
    """Helper function for parallel uniform Pull execution."""
    n, trial = args
    seed = BASE_SEED + n + 999999 + trial  # Use different seed space
    return simulate_uniform_pull(n, seed)


def run_experiments():
    """Run all experiments in parallel."""
    print("=" * 80)
    print("LA-PULL vs UNIFORM PULL COMPARISON")
    print("=" * 80)
    print(f"Network sizes (n):     {N_VALUES}")
    print(f"Trials per config:     {TRIALS}")
    print(f"P values:              {P_VALUES[0]:.1f} to {P_VALUES[-1]:.1f} (step 0.1)")
    print(f"CPU cores:             {cpu_count()}")
    print("=" * 80)
    print()
    
    results = []
    
    # ========================================================================
    # Part 1: LA-Pull experiments (for each n, sweep p)
    # ========================================================================
    print("Running LA-Pull experiments...")
    for n in N_VALUES:
        # Prepare tasks for this n
        tasks = [(n, p, trial) for p in P_VALUES for trial in range(TRIALS)]
        
        # Run in parallel
        with Pool(cpu_count()) as pool:
            rounds_results = list(tqdm(
                pool.imap(run_single_la_pull, tasks),
                total=len(tasks),
                desc=f"LA-Pull n={n:,}"
            ))
        
        # Organize results by p
        idx = 0
        for p in P_VALUES:
            rounds_list = rounds_results[idx:idx + TRIALS]
            idx += TRIALS
            
            rounds_arr = np.array(rounds_list)
            mean_rounds = rounds_arr.mean()
            median_rounds = np.median(rounds_arr)
            p90_rounds = np.percentile(rounds_arr, 90)
            
            # 95% CI using t-distribution
            if len(rounds_arr) > 1:
                ci = stats.t.interval(0.95, len(rounds_arr) - 1,
                                     loc=mean_rounds,
                                     scale=stats.sem(rounds_arr))
            else:
                ci = (mean_rounds, mean_rounds)
            
            results.append({
                'algorithm': 'LA-Pull',
                'n': n,
                'p': p,
                'mean': mean_rounds,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'median': median_rounds,
                'p90': p90_rounds
            })
    
    # ========================================================================
    # Part 2: Uniform Pull experiments (for each n, p doesn't matter)
    # ========================================================================
    print("\nRunning Uniform Pull experiments...")
    for n in N_VALUES:
        # Prepare tasks for this n (no p dependency)
        tasks = [(n, trial) for trial in range(TRIALS)]
        
        # Run in parallel
        with Pool(cpu_count()) as pool:
            rounds_results = list(tqdm(
                pool.imap(run_single_uniform_pull, tasks),
                total=len(tasks),
                desc=f"Uniform Pull n={n:,}"
            ))
        
        rounds_arr = np.array(rounds_results)
        mean_rounds = rounds_arr.mean()
        median_rounds = np.median(rounds_arr)
        p90_rounds = np.percentile(rounds_arr, 90)
        
        # 95% CI using t-distribution
        if len(rounds_arr) > 1:
            ci = stats.t.interval(0.95, len(rounds_arr) - 1,
                                 loc=mean_rounds,
                                 scale=stats.sem(rounds_arr))
        else:
            ci = (mean_rounds, mean_rounds)
        
        # Add result for all p values (uniform Pull doesn't depend on p)
        # This allows us to plot horizontal lines across all p values
        for p in P_VALUES:
            results.append({
                'algorithm': 'Uniform Pull',
                'n': n,
                'p': p,
                'mean': mean_rounds,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'median': median_rounds,
                'p90': p90_rounds
            })
    
    return pd.DataFrame(results)


def plot_comparison(df):
    """
    Plot 6 curves on the same figure:
    - LA-Pull for n = 10k, 25k, 50k (curves that decrease with increasing p)
    - Uniform Pull for n = 10k, 25k, 50k (horizontal lines, p-independent)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color scheme: Different colors for different n values
    # LA-Pull: solid lines with blue shades
    # Uniform Pull: dashed lines with red shades
    colors_la = ['#1f77b4', '#5aa3d6', '#9fcbea']      # Dark to light blue
    colors_uniform = ['#d62728', '#e65b5d', '#f59396']  # Dark to light red
    
    markers = ['o', 's', '^']  # Circle, square, triangle
    line_styles_la = ['-', '-', '-']
    line_styles_uniform = ['--', '--', '--']
    
    # Plot LA-Pull curves (3 curves, one per n)
    for i, n in enumerate(N_VALUES):
        subset = df[(df['algorithm'] == 'LA-Pull') & (df['n'] == n)].sort_values('p')
        
        # Plot line with markers
        ax.plot(subset['p'], subset['mean'], 
               marker=markers[i], markersize=7, linewidth=2.5,
               color=colors_la[i], linestyle=line_styles_la[i],
               label=f'LA-Pull (n={n//1000}k)',
               zorder=3)
        
        # Add CI band (shaded region)
        ax.fill_between(subset['p'], subset['ci_low'], subset['ci_high'],
                       alpha=0.15, color=colors_la[i], zorder=1)
        
        # Add error bars with caps (only where CI is visible)
        for _, row in subset.iterrows():
            ci_width = row['ci_high'] - row['ci_low']
            if ci_width >= 0.05:
                ax.errorbar(row['p'], row['mean'],
                          yerr=[[row['mean'] - row['ci_low']], 
                                [row['ci_high'] - row['mean']]],
                          color=colors_la[i], linewidth=1.2, 
                          capsize=3, capthick=1.2, zorder=2, alpha=0.7)
    
    # Plot Uniform Pull lines (3 horizontal lines with markers, one per n)
    for i, n in enumerate(N_VALUES):
        # Get the constant value (same for all p)
        subset = df[(df['algorithm'] == 'Uniform Pull') & (df['n'] == n)].sort_values('p')
        if len(subset) > 0:
            mean_val = subset.iloc[0]['mean']
            ci_low = subset.iloc[0]['ci_low']
            ci_high = subset.iloc[0]['ci_high']
            
            # Plot horizontal line with markers (same style as LA-Pull)
            ax.plot(subset['p'], [mean_val] * len(subset), 
                   marker=markers[i], markersize=7, linewidth=2.5,
                   color=colors_uniform[i], linestyle=line_styles_uniform[i],
                   label=f'Uniform Pull (n={n//1000}k)',
                   zorder=2)
            
            # Add shaded CI band (horizontal)
            ax.fill_between(subset['p'], ci_low, ci_high, 
                           alpha=0.10, color=colors_uniform[i], zorder=0)
    
    # Formatting
    ax.set_xlabel('Predictor Accuracy (p)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Round Complexity', fontsize=13, fontweight='bold')
    
    # Legend with two columns for compactness
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95, ncol=2,
             columnspacing=1.0, handlelength=2.5)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_xlim(-0.05, 1.05)
    
    # Format ticks
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xticks(P_VALUES)
    
    plt.tight_layout()
    
    # Save figure (PNG only)
    png_file = 'la_pull_comparison_plot.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {png_file}")


def print_summary(df):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Rounds for Selected Configurations")
    print("=" * 80)
    print(f"{'Algorithm':<18} {'n':<10} {'p=0.0':<10} {'p=0.2':<10} {'p=0.5':<10} "
          f"{'p=0.8':<10} {'p=1.0':<10}")
    print("-" * 80)
    
    for algo in ['LA-Pull', 'Uniform Pull']:
        for n in N_VALUES:
            row_str = f"{algo:<18} {n:<10,}"
            for p_val in [0.0, 0.2, 0.5, 0.8, 1.0]:
                match = df[(df['algorithm'] == algo) & 
                          (df['n'] == n) & 
                          (np.isclose(df['p'], p_val, atol=0.01))]
                if len(match) > 0:
                    mean_val = match.iloc[0]['mean']
                    row_str += f"{mean_val:<10.2f}"
                else:
                    row_str += f"{'N/A':<10}"
            print(row_str)
    
    print("=" * 80)


def main():
    """Main execution."""
    # Run all experiments
    df = run_experiments()
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(df)
    
    # Print summary table
    print_summary(df)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] All experiments complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

