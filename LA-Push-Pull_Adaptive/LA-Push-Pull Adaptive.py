import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ============================================================================
# PARAMETERS
# ============================================================================

N = 50_000                              # Graph size
TRIALS = 20                             # Repetitions per (q, epsilon) pair
Q_VALUES = np.linspace(0.1, 1.0, 10)    # [0.1, 0.2, ..., 1.0]
EPSILONS = [0.01, 0.05, 0.1]            # Estimation noise levels
BASE_SEED = 42                          # Reproducibility

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_la_pushpull_adaptive(n, q, epsilon, seed, p=None, debug=False):
    """
    Simulate LA-Push-Pull with adaptive thresholds (Algorithm 6: PP-H1Q).
    
    PP-H1Q Adaptive Policy:
    - Round 1: Always use predictor P for pulls (seeding phase)
    - Later rounds:
      * PULL: Use P if p > f_hat_t (estimated informed fraction + noise)
      * PUSH: Use Q if q > u_hat_t (estimated uninformed fraction + noise)
    
    Parameters
    ----------
    n : int
        Number of nodes
    q : float
        Push predictor accuracy (0 to 1)
    epsilon : float
        Standard deviation of Gaussian noise for fraction estimates
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
        print(f"[DEBUG] LA-Push-Pull Adaptive: n={n}, q={q:.2f}, p={p:.2f}, epsilon={epsilon}")
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
        
        # NOISY estimates (Gaussian noise with std = epsilon)
        f_hat_t = f_t + rng.normal(0, epsilon)
        u_hat_t = u_t + rng.normal(0, epsilon)
        
        # Clip to [0, 1] to keep estimates valid
        f_hat_t = np.clip(f_hat_t, 0, 1)
        u_hat_t = np.clip(u_hat_t, 0, 1)
        
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
            # Stalled (shouldn't happen with adaptive policy, but check)
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


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def run_single_trial(args):
    """Wrapper for multiprocessing."""
    q, epsilon, trial, n = args
    seed = BASE_SEED + int(q * 1000) * 10000 + int(epsilon * 1000) * 100 + trial
    rounds = simulate_la_pushpull_adaptive(n, q, epsilon, seed)
    return (q, epsilon, trial, rounds)


def run_experiments(n, q_values, epsilons, trials):
    """Run all experiments in parallel."""
    print("="*80)
    print("LA-PUSH-PULL ADAPTIVE (Algorithm 6: PP-H1Q)")
    print("="*80)
    print(f"Graph size (n):        {n:,}")
    print(f"Trials per (q, eps):   {trials}")
    print(f"Q values:              {len(q_values)} points in [{q_values.min():.1f}, {q_values.max():.1f}]")
    print(f"Epsilon values:        {epsilons}")
    print(f"Total simulations:     {len(q_values) * len(epsilons) * trials:,}")
    print(f"CPU cores:             {cpu_count()}")
    print("="*80)
    
    # Prepare all parameter combinations
    tasks = [
        (q, epsilon, trial, n)
        for q in q_values
        for epsilon in epsilons
        for trial in range(trials)
    ]
    
    # Run in parallel
    results = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(run_single_trial, tasks),
                          total=len(tasks),
                          desc="Simulations"):
            results.append(result)
    
    return results


# ============================================================================
# STATISTICS
# ============================================================================

def compute_statistics(data, q_values, epsilons, trials):
    """Compute mean, CI, median, p90 for each (q, epsilon) pair."""
    stats_list = []
    
    for epsilon in epsilons:
        for q in q_values:
            # Filter data for this (q, epsilon)
            subset = [r for r in data if r[0] == q and r[1] == epsilon]
            rounds_list = [r[3] for r in subset if r[3] > 0]  # Exclude stalls (-1)
            
            if len(rounds_list) == 0:
                # All trials stalled
                stats_list.append({
                    'epsilon': epsilon,
                    'q': q,
                    'mean': np.nan,
                    'ci_low': np.nan,
                    'ci_high': np.nan,
                    'median': np.nan,
                    'p90': np.nan,
                    'stalled': trials
                })
                continue
            
            rounds_arr = np.array(rounds_list)
            mean_val = np.mean(rounds_arr)
            median_val = np.median(rounds_arr)
            p90_val = np.percentile(rounds_arr, 90)
            
            # 95% confidence interval
            if len(rounds_arr) > 1:
                sem = stats.sem(rounds_arr)
                ci = stats.t.interval(0.95, len(rounds_arr) - 1, loc=mean_val, scale=sem)
                ci_low, ci_high = ci
            else:
                ci_low, ci_high = mean_val, mean_val
            
            num_stalled = trials - len(rounds_list)
            
            stats_list.append({
                'epsilon': epsilon,
                'q': q,
                'mean': mean_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'median': median_val,
                'p90': p90_val,
                'stalled': num_stalled
            })
    
    return pd.DataFrame(stats_list)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_rounds_vs_q(df, output_prefix):
    """Plot rounds vs q with one curve per epsilon."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    markers = ['o', 's', '^']
    
    for i, epsilon in enumerate(EPSILONS):
        subset = df[df['epsilon'] == epsilon].copy()
        subset = subset.sort_values('q')
        
        # Plot mean with error bars
        for j, row in subset.iterrows():
            ci_width = row['ci_high'] - row['ci_low']
            
            # If CI is tiny, use a minimum visible whisker
            if ci_width < 0.05:
                plot_ci_low = row['mean'] - 0.10
                plot_ci_high = row['mean'] + 0.10
            else:
                plot_ci_low = row['ci_low']
                plot_ci_high = row['ci_high']
            
            ax.fill_between([row['q'] - 0.02, row['q'] + 0.02],
                           plot_ci_low, plot_ci_high,
                           alpha=0.2, color=colors[i], linewidth=0)
        
        # Plot line and markers
        ax.plot(subset['q'], subset['mean'], 
               marker=markers[i], markersize=6, linewidth=2,
               color=colors[i], label=f'epsilon = {epsilon}',
               zorder=3)
        
        # Add errorbars with caps
        ax.errorbar(subset['q'], subset['mean'],
                   yerr=[subset['mean'] - subset['ci_low'], 
                         subset['ci_high'] - subset['mean']],
                   fmt='none', ecolor=colors[i], alpha=0.6, capsize=3, capthick=1.5,
                   zorder=2)
    
    ax.set_xlabel('Predictor Accuracy (p = q)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rounds to Full Infection (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('LA-Push-Pull Adaptive (PP-H1Q): Effect of Estimation Noise',
                fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fontsize=10, loc='best',
             title='Estimation Noise (std)')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0.05, 1.05)
    
    # Set y-limits after tight_layout
    plt.tight_layout()
    y_min = df['ci_low'].min()
    y_max = df['ci_high'].max()
    ax.set_ylim(y_min - 1, y_max + 1)
    
    # Save
    fig.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"\n[SAVED] {output_prefix}.png")
    print(f"[SAVED] {output_prefix}.pdf")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Run experiments
    results = run_experiments(N, Q_VALUES, EPSILONS, TRIALS)
    
    # Compute statistics
    print("\n" + "="*80)
    print("Computing statistics...")
    df_stats = compute_statistics(results, Q_VALUES, EPSILONS, TRIALS)
    
    # Save CSV
    csv_path = 'la_pushpull_adaptive_results.csv'
    df_stats.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"[SAVED] {csv_path}")
    
    # Plot
    print("\n" + "="*80)
    print("Generating plots...")
    plot_rounds_vs_q(df_stats, 'la_pushpull_adaptive_rounds_vs_q')
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Mean Rounds for Selected Configurations")
    print("="*80)
    print(f"{'Epsilon':<12} {'q=0.2':<10} {'q=0.4':<10} {'q=0.6':<10} {'q=0.8':<10} {'q=1.0':<10}")
    print("-"*80)
    
    for epsilon in EPSILONS:
        row_str = f"{epsilon:<12.2f}"
        for q_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
            match = df_stats[(df_stats['epsilon'] == epsilon) & 
                            (np.isclose(df_stats['q'], q_val, atol=0.01))]
            if len(match) > 0:
                mean_val = match.iloc[0]['mean']
                if np.isnan(mean_val):
                    row_str += f"{'STALL':<10}"
                else:
                    row_str += f"{mean_val:<10.2f}"
            else:
                row_str += f"{'N/A':<10}"
        print(row_str)
    
    print("="*80)
    print("\n[VERIFICATION] First 5 rows of data:")
    print(df_stats.head())
    print("\n" + "="*80)
    print("[SUCCESS] LA-Push-Pull Adaptive experiments complete!")
    print("="*80)


if __name__ == '__main__':
    main()

