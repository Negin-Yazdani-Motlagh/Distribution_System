"""
LA-Push with Time-Threshold Predictor Switch (Algorithm 3)

Simulates push-based epidemic spreading on a complete graph with:
- Synchronous rounds
- One random initial infected node
- Time threshold τ: use uniform push before τ, predictor Q after
- Predictor Q with accuracy q: q → uninformed, 1-q → informed
"""

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
TRIALS = 20                             # Repetitions per (q, τ) pair
Q_VALUES = np.linspace(0.0, 1.0, 11)    # [0.0, 0.1, ..., 1.0]
TAUS = [0, 2, 3, 4, 5, 6]               # Threshold values
Q_SLICES = [0.1, 0.2, 0.4, 0.6]         # For τ sweep plot
BASE_SEED = 42                          # Reproducibility

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_la_push(n, tau, q, seed):
    """
    Simulate one LA-Push epidemic with time-threshold predictor.
    
    Parameters
    ----------
    n : int
        Number of nodes
    tau : int
        Time threshold (use uniform if t <= tau, predictor if t > tau)
    q : float
        Predictor accuracy (0 to 1)
    seed : int
        Random seed
        
    Returns
    -------
    rounds : int
        Number of rounds until all nodes infected
    """
    rng = np.random.default_rng(seed)
    
    # Initialize: one random infected node
    infected = np.zeros(n, dtype=bool)
    initial = rng.integers(0, n)
    infected[initial] = True
    
    rounds = 0
    
    while not infected.all():
        rounds += 1
        
        # Get current infected and uninformed sets
        infected_indices = np.flatnonzero(infected)
        num_infected = len(infected_indices)
        
        # Array to collect new infections this round
        new_infections = set()
        
        if rounds <= tau:
            # ============================================================
            # UNIFORM PHASE (t <= τ): sample from V \ {v}
            # ============================================================
            # Sample from [0, n-1) and map to avoid self
            targets = rng.integers(0, n - 1, size=num_infected)
            # If target >= sender, increment to skip sender
            for i in range(num_infected):
                if targets[i] >= infected_indices[i]:
                    targets[i] += 1
            
            # New infections: targets that were uninformed
            for target in targets:
                if not infected[target]:
                    new_infections.add(target)
        
        else:
            # ============================================================
            # PREDICTOR PHASE (t > τ): use Q with accuracy q
            # ============================================================
            uninformed_indices = np.flatnonzero(~infected)
            num_uninformed = len(uninformed_indices)
            
            if num_uninformed == 0:
                break  # All infected
            
            # For each infected node, decide: aim at uninformed (q) or informed (1-q)
            aim_at_uninformed = rng.random(size=num_infected) < q
            
            # Good pushes: aim at uninformed nodes
            num_good = aim_at_uninformed.sum()
            if num_good > 0 and num_uninformed > 0:
                # Sample uniformly from uninformed set (with replacement)
                good_targets = rng.choice(uninformed_indices, size=num_good)
                for target in good_targets:
                    new_infections.add(target)
            
            # Bad pushes: aim at informed nodes (no new infections)
            # Fallback: if predictor produced zero new infections, use uniform push
            # to guarantee progress (otherwise epidemic can stall with q < 1)
            if len(new_infections) == 0 and num_uninformed > 0:
                # Use uniform push from all infected nodes
                targets = rng.integers(0, n - 1, size=num_infected)
                for i in range(num_infected):
                    if targets[i] >= infected_indices[i]:
                        targets[i] += 1
                for target in targets:
                    if not infected[target]:
                        new_infections.add(target)
                
                # Ultimate fallback: if still no infections (very unlucky collisions),
                # infect one random uninformed node to guarantee progress
                if len(new_infections) == 0:
                    new_infections.add(rng.choice(uninformed_indices))
        
        # Synchronous update: activate new infections at start of next round
        for node in new_infections:
            infected[node] = True
    
    return rounds


# ============================================================================
# PARALLEL HELPER
# ============================================================================

def run_single_trial(args):
    """Helper function for parallel execution."""
    n, tau, q, seed = args
    return simulate_la_push(n, tau, q, seed)


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_q_sweep():
    """
    Sweep q for each τ, compute statistics across trials (parallelized).
    
    Returns
    -------
    results : pd.DataFrame
        Columns: tau, q, mean, ci_low, ci_high, median, p90
    """
    results = []
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores for parallel execution")
    
    # Prepare all tasks
    tasks = []
    for tau in TAUS:
        for q in Q_VALUES:
            for trial in range(TRIALS):
                seed = BASE_SEED + 10_000 * int(10 * q) + 100 * tau + trial
                tasks.append((N, tau, q, seed))
    
    # Run in parallel with progress bar
    with Pool(num_cores) as pool:
        rounds_results = list(tqdm(
            pool.imap(run_single_trial, tasks),
            total=len(tasks),
            desc="Q sweep"
        ))
    
    # Organize results by (tau, q)
    idx = 0
    for tau in TAUS:
        for q in Q_VALUES:
            rounds_list = rounds_results[idx:idx + TRIALS]
            idx += TRIALS
            
            # Compute statistics
            rounds_arr = np.array(rounds_list)
            mean_rounds = rounds_arr.mean()
            std_rounds = rounds_arr.std(ddof=1)
            
            # 95% CI using Student's t
            t_crit = stats.t.ppf(0.975, df=TRIALS - 1)
            margin = t_crit * std_rounds / np.sqrt(TRIALS)
            ci_low = mean_rounds - margin
            ci_high = mean_rounds + margin
            
            median_rounds = np.median(rounds_arr)
            p90_rounds = np.percentile(rounds_arr, 90)
            
            results.append({
                'tau': tau,
                'q': q,
                'mean': mean_rounds,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'median': median_rounds,
                'p90': p90_rounds
            })
    
    return pd.DataFrame(results)


def run_tau_sweep():
    """
    Sweep τ for selected q values, compute statistics across trials (parallelized).
    
    Returns
    -------
    results : pd.DataFrame
        Columns: q, tau, mean, ci_low, ci_high, median, p90
    """
    results = []
    num_cores = cpu_count()
    
    # Prepare all tasks
    tasks = []
    for q in Q_SLICES:
        for tau in TAUS:
            for trial in range(TRIALS):
                seed = BASE_SEED + 10_000 * int(10 * q) + 100 * tau + trial
                tasks.append((N, tau, q, seed))
    
    # Run in parallel with progress bar
    with Pool(num_cores) as pool:
        rounds_results = list(tqdm(
            pool.imap(run_single_trial, tasks),
            total=len(tasks),
            desc="tau sweep"
        ))
    
    # Organize results by (q, tau)
    idx = 0
    for q in Q_SLICES:
        for tau in TAUS:
            rounds_list = rounds_results[idx:idx + TRIALS]
            idx += TRIALS
            
            # Compute statistics
            rounds_arr = np.array(rounds_list)
            mean_rounds = rounds_arr.mean()
            std_rounds = rounds_arr.std(ddof=1)
            
            # 95% CI using Student's t
            t_crit = stats.t.ppf(0.975, df=TRIALS - 1)
            margin = t_crit * std_rounds / np.sqrt(TRIALS)
            ci_low = mean_rounds - margin
            ci_high = mean_rounds + margin
            
            median_rounds = np.median(rounds_arr)
            p90_rounds = np.percentile(rounds_arr, 90)
            
            results.append({
                'q': q,
                'tau': tau,
                'mean': mean_rounds,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'median': median_rounds,
                'p90': p90_rounds
            })
    
    return pd.DataFrame(results)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_rounds_vs_q(df):
    """
    Plot rounds vs q (one curve per τ).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each τ
    colors = plt.cm.viridis(np.linspace(0, 1, len(TAUS)))
    
    for i, tau in enumerate(TAUS):
        subset = df[df['tau'] == tau]
        q_vals = subset['q'].values
        means = subset['mean'].values
        ci_lows = subset['ci_low'].values
        ci_highs = subset['ci_high'].values
        
        # Plot line
        ax.plot(q_vals, means, marker='o', label=f'τ = {tau}', 
                color=colors[i], linewidth=2, markersize=6)
        
        # Shaded CI
        ax.fill_between(q_vals, ci_lows, ci_highs, alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Predictor Accuracy (q)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rounds to Full Infection', fontsize=12, fontweight='bold')
    ax.set_title(f'LA-Push (Q after time threshold) — n={N:,}, T={TRIALS}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('la_push_rounds_vs_q.png', dpi=300, bbox_inches='tight')
    plt.savefig('la_push_rounds_vs_q.pdf', bbox_inches='tight')
    plt.close()
    
    print("+ Saved la_push_rounds_vs_q.png and .pdf")


def plot_rounds_vs_tau(df):
    """
    Plot rounds vs tau (one curve per selected q).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each q
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(Q_SLICES)))
    
    for i, q in enumerate(Q_SLICES):
        subset = df[df['q'] == q]
        tau_vals = subset['tau'].values
        means = subset['mean'].values
        ci_lows = subset['ci_low'].values
        ci_highs = subset['ci_high'].values
        
        # Plot line
        ax.plot(tau_vals, means, marker='s', label=f'q = {q:.1f}', 
                color=colors[i], linewidth=2, markersize=7)
        
        # Shaded CI
        ax.fill_between(tau_vals, ci_lows, ci_highs, alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Time Threshold (τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rounds to Full Infection', fontsize=12, fontweight='bold')
    ax.set_title(f'LA-Push — tuning threshold τ (n={N:,}, T={TRIALS})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(TAUS)
    
    plt.tight_layout()
    plt.savefig('la_push_rounds_vs_tau.png', dpi=300, bbox_inches='tight')
    plt.savefig('la_push_rounds_vs_tau.pdf', bbox_inches='tight')
    plt.close()
    
    print("+ Saved la_push_rounds_vs_tau.png and .pdf")


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_best_tau_summary(df_tau):
    """
    Print a summary table of best tau for each q in Q_SLICES.
    """
    print("\n" + "="*60)
    print("BEST TAU FOR EACH PREDICTOR ACCURACY")
    print("="*60)
    print(f"{'q':<8} {'Best tau':<10} {'Mean Rounds':<15} {'95% CI':<20}")
    print("-"*60)
    
    for q in Q_SLICES:
        subset = df_tau[df_tau['q'] == q]
        best_idx = subset['mean'].idxmin()
        best_row = subset.loc[best_idx]
        
        best_tau = int(best_row['tau'])
        best_mean = best_row['mean']
        ci_low = best_row['ci_low']
        ci_high = best_row['ci_high']
        
        print(f"{q:<8.1f} {best_tau:<10} {best_mean:<15.2f} "
              f"[{ci_low:.2f}, {ci_high:.2f}]")
    
    print("="*60 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("LA-PUSH WITH TIME-THRESHOLD PREDICTOR SWITCH")
    print("="*70)
    print(f"Graph size (n):        {N:,}")
    print(f"Trials per config:     {TRIALS}")
    print(f"Q values:              {len(Q_VALUES)} points from {Q_VALUES[0]:.1f} to {Q_VALUES[-1]:.1f}")
    print(f"Threshold values (tau): {TAUS}")
    print(f"Q slices for tau plot: {Q_SLICES}")
    print(f"Base seed:             {BASE_SEED}")
    print("="*70 + "\n")
    
    # Run experiments
    print("Running Q sweep (rounds vs q for each tau)...")
    df_q = run_q_sweep()
    df_q.to_csv('la_push_rounds_vs_q.csv', index=False)
    print("+ Saved la_push_rounds_vs_q.csv\n")
    
    print("Running tau sweep (rounds vs tau for each q)...")
    df_tau = run_tau_sweep()
    df_tau.to_csv('la_push_rounds_vs_tau.csv', index=False)
    print("+ Saved la_push_rounds_vs_tau.csv\n")
    
    # Generate plots
    print("Generating plots...")
    plot_rounds_vs_q(df_q)
    plot_rounds_vs_tau(df_tau)
    
    # Print summary
    print_best_tau_summary(df_tau)
    
    print("All done!")


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

