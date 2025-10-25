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

def simulate_la_push(n, tau, q, seed, debug=False):
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
    debug : bool, optional
        If True and q=0, print debug information (default: False)
        
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
            # With q < 1, predictor can produce zero infections (stalled epidemic)
            # This is theoretically correct behavior - the epidemic may never finish
        
        # Check for stalled epidemic (no new infections in predictor phase)
        if len(new_infections) == 0:
            if debug and q == 0.0:
                print(f"[DEBUG] q=0 STALLED at round {rounds} with {num_infected}/{n} infected")
                print(f"[DEBUG SUMMARY] q=0 epidemic stalled (infinite rounds) — tau={tau}")
            # Return -1 to indicate the epidemic stalled (infinite rounds)
            return -1
        
        # Synchronous update: activate new infections at start of next round
        for node in new_infections:
            infected[node] = True
        
        # Debug output for q=0 case
        if debug and q == 0.0 and rounds % 3 == 0:
            print(f"[DEBUG] q=0 run — tau={tau}, round={rounds}, num_infected={num_infected}, new_infections={len(new_infections)}")
    
    # Debug summary
    if debug and q == 0.0:
        print(f"[DEBUG SUMMARY] q=0 finished in {rounds} rounds (tau={tau}) — final infected={infected.sum()} / {n}")
    
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
            
            # Filter out stalled epidemics (-1 values)
            rounds_arr = np.array(rounds_list)
            successful = rounds_arr[rounds_arr > 0]
            num_stalled = np.sum(rounds_arr == -1)
            
            # Compute statistics only on successful runs
            if len(successful) > 0:
                mean_rounds = successful.mean()
                if len(successful) > 1:
                    std_rounds = successful.std(ddof=1)
                    t_crit = stats.t.ppf(0.975, df=len(successful) - 1)
                    margin = t_crit * std_rounds / np.sqrt(len(successful))
                else:
                    margin = 0
                ci_low = mean_rounds - margin
                ci_high = mean_rounds + margin
                median_rounds = np.median(successful)
                p90_rounds = np.percentile(successful, 90)
            else:
                # All trials stalled (infinite rounds)
                mean_rounds = np.inf
                ci_low = np.inf
                ci_high = np.inf
                median_rounds = np.inf
                p90_rounds = np.inf
            
            results.append({
                'tau': tau,
                'q': q,
                'mean': mean_rounds,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'median': median_rounds,
                'p90': p90_rounds,
                'num_stalled': num_stalled,
                'num_successful': len(successful)
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
            
            # Filter out stalled epidemics (-1 values)
            rounds_arr = np.array(rounds_list)
            successful = rounds_arr[rounds_arr > 0]
            num_stalled = np.sum(rounds_arr == -1)
            
            # Compute statistics only on successful runs
            if len(successful) > 0:
                mean_rounds = successful.mean()
                if len(successful) > 1:
                    std_rounds = successful.std(ddof=1)
                    t_crit = stats.t.ppf(0.975, df=len(successful) - 1)
                    margin = t_crit * std_rounds / np.sqrt(len(successful))
                else:
                    margin = 0
                ci_low = mean_rounds - margin
                ci_high = mean_rounds + margin
                median_rounds = np.median(successful)
                p90_rounds = np.percentile(successful, 90)
            else:
                # All trials stalled (infinite rounds)
                mean_rounds = np.inf
                ci_low = np.inf
                ci_high = np.inf
                median_rounds = np.inf
                p90_rounds = np.inf
            
            results.append({
                'q': q,
                'tau': tau,
                'mean': mean_rounds,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'median': median_rounds,
                'p90': p90_rounds,
                'num_stalled': num_stalled,
                'num_successful': len(successful)
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
        subset = df[df['tau'] == tau].copy()
        
        # Filter out infinite values (stalled epidemics)
        finite_mask = np.isfinite(subset['mean'])
        subset_finite = subset[finite_mask]
        
        if len(subset_finite) > 0:
            q_vals = subset_finite['q'].values
            means = subset_finite['mean'].values
            ci_lows = subset_finite['ci_low'].values
            ci_highs = subset_finite['ci_high'].values
            
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
    
    # Add note about q=0 stalling
    num_stalled_q0 = df[(df['q'] == 0.0) & (df['num_stalled'] > 0)]['num_stalled'].sum()
    if num_stalled_q0 > 0:
        ax.text(0.02, 0.98, 'Note: q=0 epidemics stall (infinite rounds)', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
        subset = df[df['q'] == q].copy()
        
        # Filter out infinite values (stalled epidemics)
        finite_mask = np.isfinite(subset['mean'])
        subset_finite = subset[finite_mask]
        
        if len(subset_finite) > 0:
            tau_vals = subset_finite['tau'].values
            means = subset_finite['mean'].values
            ci_lows = subset_finite['ci_low'].values
            ci_highs = subset_finite['ci_high'].values
            
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
    print("\n" + "="*70)
    print("BEST TAU FOR EACH PREDICTOR ACCURACY")
    print("="*70)
    print(f"{'q':<8} {'Best tau':<10} {'Mean Rounds':<15} {'95% CI':<25} {'Status':<15}")
    print("-"*70)
    
    for q in Q_SLICES:
        subset = df_tau[df_tau['q'] == q].copy()
        
        # Filter finite values
        finite_subset = subset[np.isfinite(subset['mean'])]
        
        if len(finite_subset) > 0:
            best_idx = finite_subset['mean'].idxmin()
            best_row = finite_subset.loc[best_idx]
            
            best_tau = int(best_row['tau'])
            best_mean = best_row['mean']
            ci_low = best_row['ci_low']
            ci_high = best_row['ci_high']
            num_stalled = int(best_row['num_stalled'])
            
            status = "OK" if num_stalled == 0 else f"{num_stalled} stalled"
            
            print(f"{q:<8.1f} {best_tau:<10} {best_mean:<15.2f} "
                  f"[{ci_low:.2f}, {ci_high:.2f}]    {status:<15}")
        else:
            print(f"{q:<8.1f} {'N/A':<10} {'inf':<15} "
                  f"{'N/A':<25} {'All stalled':<15}")
    
    print("="*70 + "\n")


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
    
    # Debug check for q=0 case
    print("Running debug check for q=0 (tau=0)...")
    simulate_la_push(200, tau=0, q=0.0, seed=42, debug=True)
    print()
    
    # Run experiments
    print("Running Q sweep (rounds vs q for each tau)...")
    df_q = run_q_sweep()
    df_q.to_csv('la_push_rounds_vs_q.csv', index=False)
    print("+ Saved la_push_rounds_vs_q.csv")
    
    # Report stalled configurations
    total_stalled = df_q['num_stalled'].sum()
    if total_stalled > 0:
        stalled_configs = df_q[df_q['num_stalled'] > 0][['tau', 'q', 'num_stalled']]
        print(f"\nWARNING: {int(total_stalled)} trial(s) stalled (infinite rounds)")
        print("Stalled configurations:")
        for _, row in stalled_configs.iterrows():
            print(f"  tau={int(row['tau'])}, q={row['q']:.1f}: {int(row['num_stalled'])}/{TRIALS} trials stalled")
    print()
    
    print("Running tau sweep (rounds vs tau for each q)...")
    df_tau = run_tau_sweep()
    df_tau.to_csv('la_push_rounds_vs_tau.csv', index=False)
    print("+ Saved la_push_rounds_vs_tau.csv")
    
    # Report stalled configurations
    total_stalled_tau = df_tau['num_stalled'].sum()
    if total_stalled_tau > 0:
        stalled_configs_tau = df_tau[df_tau['num_stalled'] > 0][['tau', 'q', 'num_stalled']]
        print(f"\nWARNING: {int(total_stalled_tau)} trial(s) stalled (infinite rounds)")
        print("Stalled configurations:")
        for _, row in stalled_configs_tau.iterrows():
            print(f"  tau={int(row['tau'])}, q={row['q']:.1f}: {int(row['num_stalled'])}/{TRIALS} trials stalled")
    print()
    
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

