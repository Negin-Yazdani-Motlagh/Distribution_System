#!/usr/bin/env python3
"""
LA-Push-Pull vs Uniform Push-Pull Comparison

Compares:
- LA-Push-Pull (PP-H1Q with additive noise) for epsilon = 0.01, 0.05, 0.1
- Uniform Push-Pull (no predictor, uniform every round)

Uses ADDITIVE noise: f_hat_t = f_t + epsilon, u_hat_t = u_t + epsilon
Shows 4 curves on the same plot (n=50k).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Parameters
N = 50_000                              # Network size
TRIALS = 20                             # Number of trials per (q, epsilon) pair
Q_VALUES = np.arange(0.1, 1.1, 0.1)     # [0.1, 0.2, ..., 1.0]
EPSILONS = [0.01, 0.05, 0.1]            # Additive noise levels
BASE_SEED = 42                          # Reproducibility


def simulate_la_pushpull_additive(n, q, epsilon, seed, p=None, debug=False):
    """
    Simulate LA-Push-Pull with ADDITIVE noise (PP-H1Q).
    
    PP-H1Q Adaptive Policy with ADDITIVE noise:
    - Round 1: Always use predictor P for pulls (seeding phase)
    - Later rounds:
      * PULL: Use P if p > f_hat_t where f_hat_t = f_t + epsilon
      * PUSH: Use Q if q > u_hat_t where u_hat_t = u_t + epsilon
    
    Parameters
    ----------
    n : int
        Number of nodes
    q : float
        Push predictor accuracy (0 to 1)
    epsilon : float
        Additive noise for fraction estimates
    seed : int
        Random seed
    p : float, optional
        Pull predictor accuracy (default: same as q)
    debug : bool, optional
        If True, print debug information (default: False)
        
    Returns
    -------
    rounds : int
        Number of rounds until all nodes infected
    """
    if p is None:
        p = q  # Default: p = q
    
    rng = np.random.default_rng(seed)
    
    # Initialize: one random infected node
    infected = np.zeros(n, dtype=bool)
    initial = rng.integers(0, n)
    infected[initial] = True
    
    rounds = 0
    
    if debug:
        print(f"[DEBUG] LA-Push-Pull Additive: n={n}, q={q:.2f}, p={p:.2f}, epsilon={epsilon}")
        print(f"[DEBUG] Initial infected: node {initial}")
    
    while not infected.all():
        rounds += 1
        
        # Get current infected and uninformed sets
        infected_indices = np.flatnonzero(infected)
        uninformed_indices = np.flatnonzero(~infected)
        num_infected = len(infected_indices)
        num_uninformed = len(uninformed_indices)
        
        if num_uninformed == 0:
            break  # All infected
        
        # TRUE fractions
        f_t = num_infected / n  # Informed fraction
        u_t = num_uninformed / n  # Uninformed fraction
        
        # ADDITIVE noise estimates
        f_hat_t = f_t + epsilon
        u_hat_t = u_t + epsilon
        
        # Collect new infections this round
        new_infections = set()
        
        # ====================================================================
        # PULL PHASE (uninformed nodes)
        # ====================================================================
        use_predictor_pull = (rounds == 1) or (p > f_hat_t)
        
        if use_predictor_pull:
            # Use predictor P: with prob p, pull from infected node
            for u_idx in uninformed_indices:
                if rng.random() < p:
                    # GOOD pull: choose an infected node
                    if num_infected > 0:
                        target = rng.choice(infected_indices)
                        new_infections.add(u_idx)
                    # else: no infected nodes, pull fails (shouldn't happen after round 1)
                else:
                    # BAD pull: choose an uninformed node (no infection)
                    pass
        else:
            # Uniform pull: each uninformed pulls from random peer != self
            pull_targets = rng.integers(0, n - 1, size=num_uninformed)
            for i in range(num_uninformed):
                u_idx = uninformed_indices[i]
                target = pull_targets[i]
                if target >= u_idx:
                    target += 1
                # If pulled node is infected, uninformed node becomes infected
                if infected[target]:
                    new_infections.add(u_idx)
        
        # ====================================================================
        # PUSH PHASE (informed nodes)
        # ====================================================================
        use_predictor_push = (q > u_hat_t)
        
        if use_predictor_push:
            # Use predictor Q: with prob q, push to uninformed node
            for v_idx in infected_indices:
                if rng.random() < q:
                    # GOOD push: choose an uninformed node
                    if num_uninformed > 0:
                        target = rng.choice(uninformed_indices)
                        new_infections.add(target)
                    # else: no uninformed nodes, done (shouldn't happen)
                else:
                    # BAD push: choose an informed node (no new infection)
                    pass
        else:
            # Uniform push: each informed pushes to random peer != self
            push_targets = rng.integers(0, n - 1, size=num_infected)
            for i in range(num_infected):
                v_idx = infected_indices[i]
                target = push_targets[i]
                if target >= v_idx:
                    target += 1
                # If target is uninformed, it becomes infected
                if not infected[target]:
                    new_infections.add(target)
        
        # ====================================================================
        # SYNCHRONOUS UPDATE
        # ====================================================================
        if len(new_infections) == 0:
            # Stalled (shouldn't happen, but check)
            if debug:
                print(f"[DEBUG] Round {rounds}: STALLED (no new infections)")
            return -1
        
        for node in new_infections:
            infected[node] = True
        
        if debug and rounds % 5 == 0:
            print(f"[DEBUG] Round {rounds}: infected={num_infected+len(new_infections)}/{n}, "
                  f"new={len(new_infections)}, f_t={f_t:.4f}, u_t={u_t:.4f}, "
                  f"f_hat={f_hat_t:.4f}, u_hat={u_hat_t:.4f}")
    
    return rounds


def simulate_uniform_pushpull(n, seed):
    r"""
    Simulate uniform Push-Pull (no predictor, always uniform).
    
    Algorithm:
    - Every round: Each uninformed node pulls uniformly from V\{u}
    - Every round: Each informed node pushes uniformly to V\{v}
    
    Args:
        n: number of nodes
        seed: random seed
    
    Returns:
        Number of rounds to infect all nodes
    """
    rng = np.random.default_rng(seed)
    
    # Initialize: one random infected node
    initial = rng.integers(0, n)
    infected = np.zeros(n, dtype=bool)
    infected[initial] = True
    
    rounds = 0
    
    while not infected.all():
        rounds += 1
        
        infected_indices = np.flatnonzero(infected)
        uninformed_indices = np.flatnonzero(~infected)
        num_infected = len(infected_indices)
        num_uninformed = len(uninformed_indices)
        
        if num_uninformed == 0:
            break
        
        new_infections = set()
        
        # PULL: Each uninformed pulls uniformly
        pull_targets = rng.integers(0, n - 1, size=num_uninformed)
        for i in range(num_uninformed):
            u_idx = uninformed_indices[i]
            target = pull_targets[i]
            if target >= u_idx:
                target += 1
            if infected[target]:
                new_infections.add(u_idx)
        
        # PUSH: Each informed pushes uniformly
        push_targets = rng.integers(0, n - 1, size=num_infected)
        for i in range(num_infected):
            v_idx = infected_indices[i]
            target = push_targets[i]
            if target >= v_idx:
                target += 1
            if not infected[target]:
                new_infections.add(target)
        
        # Update
        for node in new_infections:
            infected[node] = True
    
    return rounds


def run_single_la_pushpull(args):
    """Helper for parallel LA-Push-Pull execution."""
    q, epsilon, trial, n = args
    seed = BASE_SEED + int(q * 1000) * 10000 + int(epsilon * 1000) * 100 + trial
    return simulate_la_pushpull_additive(n, q, epsilon, seed)


def run_single_uniform_pushpull(args):
    """Helper for parallel uniform Push-Pull execution."""
    trial, n = args
    seed = BASE_SEED + n + 999999 + trial  # Different seed space
    return simulate_uniform_pushpull(n, seed)


def run_experiments():
    """Run all experiments in parallel."""
    print("=" * 80)
    print("LA-PUSH-PULL vs UNIFORM PUSH-PULL COMPARISON")
    print("=" * 80)
    print(f"Network size (n):      {N:,}")
    print(f"Trials per config:     {TRIALS}")
    print(f"Q values:              {Q_VALUES[0]:.1f} to {Q_VALUES[-1]:.1f} (step 0.1)")
    print(f"Epsilon values:        {EPSILONS} (additive noise)")
    print(f"CPU cores:             {cpu_count()}")
    print("=" * 80)
    print()
    
    results = []
    
    # ========================================================================
    # Part 1: LA-Push-Pull experiments (for each epsilon, sweep q)
    # ========================================================================
    print("Running LA-Push-Pull experiments...")
    for epsilon in EPSILONS:
        # Prepare tasks for this epsilon
        tasks = [(q, epsilon, trial, N) for q in Q_VALUES for trial in range(TRIALS)]
        
        # Run in parallel
        with Pool(cpu_count()) as pool:
            rounds_results = list(tqdm(
                pool.imap(run_single_la_pushpull, tasks),
                total=len(tasks),
                desc=f"LA-PP epsilon={epsilon}"
            ))
        
        # Organize results by q
        idx = 0
        for q in Q_VALUES:
            rounds_list = rounds_results[idx:idx + TRIALS]
            idx += TRIALS
            
            # Filter out stalls
            rounds_arr = np.array([r for r in rounds_list if r > 0])
            if len(rounds_arr) == 0:
                continue
            
            mean_rounds = rounds_arr.mean()
            median_rounds = np.median(rounds_arr)
            p90_rounds = np.percentile(rounds_arr, 90)
            
            # 95% CI
            if len(rounds_arr) > 1:
                ci = stats.t.interval(0.95, len(rounds_arr) - 1,
                                     loc=mean_rounds,
                                     scale=stats.sem(rounds_arr))
            else:
                ci = (mean_rounds, mean_rounds)
            
            results.append({
                'algorithm': 'LA-Push-Pull',
                'epsilon': epsilon,
                'q': q,
                'mean': mean_rounds,
                'ci_low': ci[0],
                'ci_high': ci[1],
                'median': median_rounds,
                'p90': p90_rounds,
                'stalled': TRIALS - len(rounds_arr)
            })
    
    # ========================================================================
    # Part 2: Uniform Push-Pull experiments (q/epsilon doesn't matter)
    # ========================================================================
    print("\nRunning Uniform Push-Pull experiments...")
    
    # Prepare tasks
    tasks = [(trial, N) for trial in range(TRIALS)]
    
    # Run in parallel
    with Pool(cpu_count()) as pool:
        rounds_results = list(tqdm(
            pool.imap(run_single_uniform_pushpull, tasks),
            total=len(tasks),
            desc=f"Uniform PP n={N:,}"
        ))
    
    rounds_arr = np.array(rounds_results)
    mean_rounds = rounds_arr.mean()
    median_rounds = np.median(rounds_arr)
    p90_rounds = np.percentile(rounds_arr, 90)
    
    # 95% CI
    if len(rounds_arr) > 1:
        ci = stats.t.interval(0.95, len(rounds_arr) - 1,
                             loc=mean_rounds,
                             scale=stats.sem(rounds_arr))
    else:
        ci = (mean_rounds, mean_rounds)
    
    # Add result for all q values (uniform doesn't depend on q or epsilon)
    for q in Q_VALUES:
        results.append({
            'algorithm': 'Uniform Push-Pull',
            'epsilon': None,
            'q': q,
            'mean': mean_rounds,
            'ci_low': ci[0],
            'ci_high': ci[1],
            'median': median_rounds,
            'p90': p90_rounds,
            'stalled': 0
        })
    
    return pd.DataFrame(results)


def plot_comparison(df):
    """
    Plot 4 curves on the same figure:
    - LA-Push-Pull for epsilon = 0.01, 0.05, 0.1 (curves depend on q)
    - Uniform Push-Pull (horizontal line, q-independent)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color scheme
    colors_la = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    colors_uniform = '#d62728'  # Red
    
    markers = ['o', 's', '^']
    line_styles_la = ['-', '-', '-']
    line_style_uniform = '--'
    
    # Plot LA-Push-Pull curves (3 curves, one per epsilon)
    for i, epsilon in enumerate(EPSILONS):
        subset = df[(df['algorithm'] == 'LA-Push-Pull') & (df['epsilon'] == epsilon)].sort_values('q')
        
        # Plot line with markers
        ax.plot(subset['q'], subset['mean'], 
               marker=markers[i], markersize=7, linewidth=2.5,
               color=colors_la[i], linestyle=line_styles_la[i],
               label=f'LA-Push-Pull (ε={epsilon})',
               zorder=3)
        
        # Add CI band
        ax.fill_between(subset['q'], subset['ci_low'], subset['ci_high'],
                       alpha=0.15, color=colors_la[i], zorder=1)
        
        # Add error bars with caps
        for _, row in subset.iterrows():
            ci_width = row['ci_high'] - row['ci_low']
            if ci_width >= 0.05:
                ax.errorbar(row['q'], row['mean'],
                          yerr=[[row['mean'] - row['ci_low']], 
                                [row['ci_high'] - row['mean']]],
                          color=colors_la[i], linewidth=1.2, 
                          capsize=3, capthick=1.2, zorder=2, alpha=0.7)
    
    # Plot Uniform Push-Pull line (horizontal line)
    subset = df[df['algorithm'] == 'Uniform Push-Pull']
    if len(subset) > 0:
        mean_val = subset.iloc[0]['mean']
        ci_low = subset.iloc[0]['ci_low']
        ci_high = subset.iloc[0]['ci_high']
        
        # Plot horizontal line
        ax.axhline(y=mean_val, color=colors_uniform, 
                  linestyle=line_style_uniform, linewidth=2.5,
                  label=f'Uniform Push-Pull',
                  zorder=2)
        
        # Add shaded CI band
        ax.axhspan(ci_low, ci_high, alpha=0.10, color=colors_uniform, zorder=0)
    
    # Formatting
    ax.set_xlabel('Predictor Accuracy (p = q)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Round Complexity', fontsize=13, fontweight='bold')
    
    # Legend
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95,
             title='Algorithm (additive noise)', title_fontsize=9)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_xlim(0.05, 1.05)
    
    # Format ticks
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xticks(Q_VALUES)
    
    plt.tight_layout()
    
    # Save figure (PNG only)
    png_file = 'la_pushpull_comparison_plot.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {png_file}")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Rounds for Selected Configurations")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Epsilon':<10} {'q=0.2':<10} {'q=0.5':<10} {'q=0.8':<10} {'q=1.0':<10}")
    print("-" * 80)
    
    for algo in ['LA-Push-Pull', 'Uniform Push-Pull']:
        if algo == 'LA-Push-Pull':
            for epsilon in EPSILONS:
                row_str = f"{algo:<25} {epsilon:<10.2f}"
                for q_val in [0.2, 0.5, 0.8, 1.0]:
                    match = df[(df['algorithm'] == algo) & 
                              (df['epsilon'] == epsilon) & 
                              (np.isclose(df['q'], q_val, atol=0.01))]
                    if len(match) > 0:
                        mean_val = match.iloc[0]['mean']
                        row_str += f"{mean_val:<10.2f}"
                    else:
                        row_str += f"{'N/A':<10}"
                print(row_str)
        else:
            row_str = f"{algo:<25} {'N/A':<10}"
            for q_val in [0.2, 0.5, 0.8, 1.0]:
                match = df[(df['algorithm'] == algo) & 
                          (np.isclose(df['q'], q_val, atol=0.01))]
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
    
    # Generate plot
    print("\nGenerating comparison plot...")
    plot_comparison(df)
    
    # Print summary
    print_summary(df)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] All experiments complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()



