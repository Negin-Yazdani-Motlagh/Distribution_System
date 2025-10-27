"""
LA-Push-Pull with Time-Threshold Predictor Switch

Simulates push-pull epidemic spreading on a complete graph with:
- Synchronous rounds
- One random initial infected node
- Time threshold τ: use uniform before τ, predictors P and Q after
- Pull predictor P with accuracy p (for uninformed nodes)
- Push predictor Q with accuracy q (for informed nodes)
- Default: p = q
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
Q_VALUES = np.linspace(0.1, 1.0, 10)    # [0.1, 0.2, ..., 1.0]
TAUS = [0, 2, 3, 4, 5, 6]               # Threshold values
Q_SLICES = [0.1, 0.2, 0.4, 0.6]         # For τ sweep plot
BASE_SEED = 42                          # Reproducibility

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_la_pushpull(n, tau, q, seed, p=None, debug=False):
    """
    Simulate LA-Push-Pull with PP-H1Q policy (paper's exact algorithm).
    
    PP-H1Q Policy:
    - Round 1: P-seeding (pull with P, push uniform)
    - Rounds 2+: Uniform push-pull until ut <= q
    - Late rounds: Q-guided push (push with Q, pull uniform) when ut <= q
    
    Parameters
    ----------
    n : int
        Number of nodes
    tau : int
        Time threshold (use uniform if t <= tau, else follow PP-H1Q)
        NOTE: tau=0 means use PP-H1Q from round 1 (recommended)
    q : float
        Push predictor accuracy (0 to 1)
    seed : int
        Random seed
    p : float, optional
        Pull predictor accuracy (default: same as q)
    debug : bool, optional
        If True, print debug information (default: False)
        
    Returns
    -------
    rounds : int
        Number of rounds until all nodes infected (-1 if stalled)
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
        print(f"\n[DEBUG] Starting LA-Push-Pull (PP-H1Q): n={n}, tau={tau}, q={q:.1f}, p={p:.1f}")
        print(f"[DEBUG] Initial infected: node {initial}")
    
    while not infected.all():
        rounds += 1
        
        # Get current infected and uninformed sets
        infected_indices = np.flatnonzero(infected)
        uninformed_indices = np.flatnonzero(~infected)
        num_infected = len(infected_indices)
        num_uninformed = len(uninformed_indices)
        ut = num_uninformed / n  # Uninformed fraction
        
        if num_uninformed == 0:
            break  # All infected
        
        # Collect new infections this round
        new_infections = set()
        
        if rounds <= tau:
            # ============================================================
            # PRE-PREDICTOR UNIFORM PHASE (t <= τ)
            # Only used if tau > 0 (for comparison)
            # ============================================================
            
            # PULL: Each uninformed node pulls from uniform random peer (≠ self)
            pull_targets = rng.integers(0, n - 1, size=num_uninformed)
            for i in range(num_uninformed):
                if pull_targets[i] >= uninformed_indices[i]:
                    pull_targets[i] += 1
                # If pulled node is infected, uninformed node becomes infected
                if infected[pull_targets[i]]:
                    new_infections.add(uninformed_indices[i])
            
            # PUSH: Each informed node pushes to uniform random peer (≠ self)
            push_targets = rng.integers(0, n - 1, size=num_infected)
            for i in range(num_infected):
                if push_targets[i] >= infected_indices[i]:
                    push_targets[i] += 1
                # If target is uninformed, it becomes infected
                if not infected[push_targets[i]]:
                    new_infections.add(push_targets[i])
        
        elif rounds == tau + 1:
            # ============================================================
            # PP-H1Q PHASE 1: P-SEEDING (Round 1 if tau=0)
            # Use P for PULL, uniform for PUSH
            # ============================================================
            
            if debug:
                print(f"[DEBUG] Round {rounds}: PP-H1Q Phase 1 - P-seeding")
            
            # PULL with P: Uninformed nodes pull with predictor P
            if num_uninformed > 0:
                pull_from_infected = rng.random(size=num_uninformed) < p
                
                for i, u in enumerate(uninformed_indices):
                    if pull_from_infected[i]:
                        # Good pull: pull from infected set → infect u
                        if num_infected > 0:
                            new_infections.add(u)
                    else:
                        # Bad pull: pull from uninformed set → no infection
                        uninformed_others = uninformed_indices[uninformed_indices != u]
                        if len(uninformed_others) > 0:
                            target = rng.choice(uninformed_others)
            
            # PUSH uniform: Informed nodes push uniformly
            if num_infected > 0:
                push_targets = rng.integers(0, n - 1, size=num_infected)
                for i in range(num_infected):
                    if push_targets[i] >= infected_indices[i]:
                        push_targets[i] += 1
                    if not infected[push_targets[i]]:
                        new_infections.add(push_targets[i])
        
        elif ut > q:
            # ============================================================
            # PP-H1Q PHASE 2: UNIFORM PUSH-PULL (until ut <= q)
            # Both pull and push are uniform
            # ============================================================
            
            if debug and rounds % 5 == 0:
                print(f"[DEBUG] Round {rounds}: PP-H1Q Phase 2 - Uniform push-pull (ut={ut:.4f} > q={q:.2f})")
            
            # PULL uniform: Each uninformed node pulls from uniform random peer
            pull_targets = rng.integers(0, n - 1, size=num_uninformed)
            for i in range(num_uninformed):
                if pull_targets[i] >= uninformed_indices[i]:
                    pull_targets[i] += 1
                if infected[pull_targets[i]]:
                    new_infections.add(uninformed_indices[i])
            
            # PUSH uniform: Each informed node pushes to uniform random peer
            push_targets = rng.integers(0, n - 1, size=num_infected)
            for i in range(num_infected):
                if push_targets[i] >= infected_indices[i]:
                    push_targets[i] += 1
                if not infected[push_targets[i]]:
                    new_infections.add(push_targets[i])
        
        else:
            # ============================================================
            # PP-H1Q PHASE 3: Q-GUIDED TAIL (when ut <= q)
            # Pull uniform, push with Q
            # ============================================================
            
            if debug and rounds % 3 == 0:
                print(f"[DEBUG] Round {rounds}: PP-H1Q Phase 3 - Q-guided tail (ut={ut:.4f} <= q={q:.2f})")
            
            # PULL uniform: Uninformed nodes pull uniformly
            pull_targets = rng.integers(0, n - 1, size=num_uninformed)
            for i in range(num_uninformed):
                if pull_targets[i] >= uninformed_indices[i]:
                    pull_targets[i] += 1
                if infected[pull_targets[i]]:
                    new_infections.add(uninformed_indices[i])
            
            # PUSH with Q: Informed nodes push with predictor Q
            if num_infected > 0 and num_uninformed > 0:
                push_to_uninformed = rng.random(size=num_infected) < q
                
                for i, v in enumerate(infected_indices):
                    if push_to_uninformed[i]:
                        # Good push: target uninformed → infect
                        target = rng.choice(uninformed_indices)
                        new_infections.add(target)
                    else:
                        # Bad push: target informed → no infection
                        target = rng.choice(infected_indices)
        
        # Check for stalled epidemic
        if len(new_infections) == 0:
            if debug:
                print(f"[DEBUG] STALLED at round {rounds} with {num_infected}/{n} infected (no new infections)")
            return -1  # Stalled
        
        # Synchronous update: apply all new infections
        for node in new_infections:
            infected[node] = True
    
    if debug:
        print(f"[DEBUG SUMMARY] Finished in {rounds} rounds — final infected={infected.sum()}/{n}")
    
    return rounds


# ============================================================================
# PARALLEL HELPER
# ============================================================================

def run_single_trial(args):
    """Helper function for parallel execution."""
    n, tau, q, seed = args
    return simulate_la_pushpull(n, tau, q, seed)


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_q_sweep():
    """
    Sweep q for each τ, compute statistics across trials (parallelized).
    
    Returns
    -------
    results : pd.DataFrame
        Columns: tau, q, mean, ci_low, ci_high, median, p90, num_stalled, num_successful
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
        Columns: q, tau, mean, ci_low, ci_high, median, p90, num_stalled, num_successful
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
    FIXED: Add visible CI whiskers even when CI ≈ 0
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each τ
    colors = plt.cm.viridis(np.linspace(0, 1, len(TAUS)))
    
    ci_width_zero_count = 0
    
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
            
            # FIXED: Check for CI ≈ 0 and use visible whisker
            ci_widths = ci_highs - ci_lows
            visible_ci_lows = np.where(ci_widths < 0.05, means - 0.10, ci_lows)
            visible_ci_highs = np.where(ci_widths < 0.05, means + 0.10, ci_highs)
            
            # Log CI ≈ 0 cases
            for j in range(len(ci_widths)):
                if ci_widths[j] < 0.05:
                    ci_width_zero_count += 1
                    print(f"[NOTE] CI ~ 0 at tau={tau}, q={q_vals[j]:.1f} (all trials equal or near-equal)")
            
            # Plot line (raised z-order for markers above band)
            line = ax.plot(q_vals, means, marker='o', label=f'τ = {tau}', 
                    color=colors[i], linewidth=2, markersize=6, zorder=3)
            
            # Shaded CI
            ax.fill_between(q_vals, visible_ci_lows, visible_ci_highs, alpha=0.2, color=colors[i], zorder=1)
            
            # FIXED: Add errorbar caps on top of filled region
            for j in range(len(q_vals)):
                if ci_widths[j] >= 0.05:
                    ax.errorbar(q_vals[j], means[j], 
                              yerr=[[means[j] - ci_lows[j]], [ci_highs[j] - means[j]]],
                              color=colors[i], linewidth=1.5, capsize=3, capthick=1.5, zorder=2)
    
    ax.set_xlabel('Predictor Accuracy (q = p)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rounds to Full Infection (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'LA-Push-Pull (PP-H1Q policy, p=q) — n={N:,}, T={TRIALS}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 1.05)  # Start from q=0.1
    
    # Add note about CI bands
    note_text = f"Bands show 95% CI; some disappear because all trials matched (CI ~ 0)"
    if ci_width_zero_count > 0:
        note_text += f"\n{ci_width_zero_count} point(s) with CI ~ 0"
    ax.text(0.02, 0.02, note_text, 
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    
    # FIXED: Set y-limits AFTER tight_layout to prevent it from resetting to 0
    y_min = df[np.isfinite(df['mean'])]['mean'].min() - 1
    y_max = df[np.isfinite(df['mean'])]['mean'].max() + 1
    ax.set_ylim(y_min, y_max)
    
    plt.savefig('la_pushpull_rounds_vs_q.png', dpi=300, bbox_inches='tight')
    plt.savefig('la_pushpull_rounds_vs_q.pdf', bbox_inches='tight')
    plt.close()
    
    print("+ Saved la_pushpull_rounds_vs_q.png and .pdf")
    
    # FIXED: Print verification table
    print("\nVerification (first 5 rows):")
    print(df.head(5)[['tau', 'q', 'mean', 'ci_low', 'ci_high']].to_string())
    print()


def plot_rounds_vs_tau(df):
    """
    Plot rounds vs tau (one curve per selected q).
    FIXED: Add visible CI whiskers even when CI ≈ 0
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each q
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(Q_SLICES)))
    
    ci_width_zero_count = 0
    
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
            
            # FIXED: Check for CI ≈ 0 and use visible whisker
            ci_widths = ci_highs - ci_lows
            visible_ci_lows = np.where(ci_widths < 0.05, means - 0.10, ci_lows)
            visible_ci_highs = np.where(ci_widths < 0.05, means + 0.10, ci_highs)
            
            # Log CI ≈ 0 cases
            for j in range(len(ci_widths)):
                if ci_widths[j] < 0.05:
                    ci_width_zero_count += 1
                    print(f"[NOTE] CI ~ 0 at q={q:.1f}, tau={tau_vals[j]} (all trials equal or near-equal)")
            
            # Plot line (raised z-order for markers above band)
            ax.plot(tau_vals, means, marker='s', label=f'q = {q:.1f}', 
                    color=colors[i], linewidth=2, markersize=7, zorder=3)
            
            # Shaded CI
            ax.fill_between(tau_vals, visible_ci_lows, visible_ci_highs, alpha=0.2, color=colors[i], zorder=1)
            
            # FIXED: Add errorbar caps on top of filled region
            for j in range(len(tau_vals)):
                if ci_widths[j] >= 0.05:
                    ax.errorbar(tau_vals[j], means[j], 
                              yerr=[[means[j] - ci_lows[j]], [ci_highs[j] - means[j]]],
                              color=colors[i], linewidth=1.5, capsize=3, capthick=1.5, zorder=2)
    
    ax.set_xlabel('Time Threshold (τ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rounds to Full Infection (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'LA-Push-Pull (PP-H1Q, p=q) — tuning threshold τ (n={N:,}, T={TRIALS})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(TAUS)
    
    # Add note about CI bands
    note_text = f"Bands show 95% CI; some disappear because all trials matched (CI ~ 0)"
    if ci_width_zero_count > 0:
        note_text += f"\n{ci_width_zero_count} point(s) with CI ~ 0"
    ax.text(0.02, 0.02, note_text, 
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    
    # FIXED: Set y-limits AFTER tight_layout to prevent it from resetting to 0
    y_min = df[np.isfinite(df['mean'])]['mean'].min() - 1
    y_max = df[np.isfinite(df['mean'])]['mean'].max() + 1
    ax.set_ylim(y_min, y_max)
    
    plt.savefig('la_pushpull_rounds_vs_tau.png', dpi=300, bbox_inches='tight')
    plt.savefig('la_pushpull_rounds_vs_tau.pdf', bbox_inches='tight')
    plt.close()
    
    print("+ Saved la_pushpull_rounds_vs_tau.png and .pdf")
    
    # FIXED: Print verification table
    print("\nVerification (first 5 rows):")
    print(df.head(5)[['q', 'tau', 'mean', 'ci_low', 'ci_high']].to_string())
    print()


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
    print("LA-PUSH-PULL WITH PP-H1Q POLICY (Paper's Exact Algorithm)")
    print("="*70)
    print(f"Graph size (n):        {N:,}")
    print(f"Trials per config:     {TRIALS}")
    print(f"Q values:              {len(Q_VALUES)} points from {Q_VALUES[0]:.1f} to {Q_VALUES[-1]:.1f}")
    print(f"Threshold values (tau): {TAUS}")
    print(f"Q slices for tau plot: {Q_SLICES}")
    print(f"Base seed:             {BASE_SEED}")
    print(f"Note: p = q (default)")
    print("="*70 + "\n")
    
    # Optional debug check
    print("Running debug checks...")
    print("\nDebug check 1: n=500, tau=0, q=0.1, p=0.1")
    simulate_la_pushpull(500, tau=0, q=0.1, seed=42, debug=True)
    
    print("\nDebug check 2: n=500, tau=0, q=0.6, p=0.6")
    simulate_la_pushpull(500, tau=0, q=0.6, seed=42, debug=True)
    print()
    
    # Run experiments
    print("Running Q sweep (rounds vs q for each tau)...")
    df_q = run_q_sweep()
    df_q.to_csv('la_pushpull_rounds_vs_q.csv', index=False)
    print("+ Saved la_pushpull_rounds_vs_q.csv")
    
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
    df_tau.to_csv('la_pushpull_rounds_vs_tau.csv', index=False)
    print("+ Saved la_pushpull_rounds_vs_tau.csv")
    
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

