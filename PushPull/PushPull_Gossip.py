import random
import matplotlib.pyplot as plt
import numpy as np
import time

def pushpull_gossip(num_nodes, trials=5, base_seed=12345):
    """
    Push-Pull gossip algorithm (push=True, pull=True)
    Combines both push and pull in each round
    """
    results = []
    random.seed(base_seed)

    for trial in range(trials):
        random.seed(base_seed + trial)

        infected = {random.randrange(num_nodes)}  # one random source
        rounds = 0

        while len(infected) < num_nodes:
            rounds += 1
            new_infected = set()

            # --- PUSH (standard): each informed contacts a uniform-random peer from ALL nodes
            for i in infected:
                target = random.randrange(num_nodes)   # may be self or already infected
                if target not in infected:
                    new_infected.add(target)

            # --- PULL (standard): each UNINFECTED contacts a uniform-random peer
            # iterate only over uninformed for speed
            for u in range(num_nodes):
                if u not in infected:
                    peer = random.randrange(num_nodes)
                    if peer in infected:
                        new_infected.add(u)

            infected.update(new_infected)

        results.append(rounds)
        print(f"  Trial {trial + 1}/{trials}: {rounds} rounds")
    
    return results

# Run simulations
print("PUSH-PULL GOSSIP ALGORITHM")
print("=" * 50)

node_counts = [10000, 50000, 100000]
all_results = {}

for num_nodes in node_counts:
    print(f"\nSimulating {num_nodes:,} nodes...")
    start_time = time.time()
    
    rounds_list = pushpull_gossip(num_nodes, trials=5)
    
    elapsed = time.time() - start_time
    avg_rounds = np.mean(rounds_list)
    std_rounds = np.std(rounds_list)
    min_rounds = min(rounds_list)
    max_rounds = max(rounds_list)
    
    all_results[num_nodes] = {
        'avg': avg_rounds,
        'std': std_rounds,
        'min': min_rounds,
        'max': max_rounds
    }
    
    print(f"  Average: {avg_rounds:.2f} rounds")
    print(f"  Range: [{min_rounds}, {max_rounds}]")
    print(f"  Time: {elapsed:.2f} seconds")

# Create plot
print("\nGenerating plot...")
node_counts_list = list(all_results.keys())
avg_rounds = [all_results[n]['avg'] for n in node_counts_list]
std_rounds = [all_results[n]['std'] for n in node_counts_list]

plt.figure(figsize=(10, 7))

# Plot with error bars
plt.errorbar(node_counts_list, avg_rounds, yerr=std_rounds, 
             marker='D', linewidth=2.5, markersize=10, capsize=8, capthick=2,
             color='purple', label='Push-Pull Gossip')

# Add theoretical O(log n) line
theoretical = [np.log2(n) for n in node_counts_list]
plt.plot(node_counts_list, theoretical, 'r--', linewidth=2, 
         label='Theoretical O(log n)', alpha=0.7)

# Formatting
plt.xlabel('Number of Nodes', fontsize=14, fontweight='bold')
plt.ylabel('Round Complexity (# of Rounds)', fontsize=14, fontweight='bold')
plt.title('Push-Pull Gossip Protocol\nRound Complexity vs Number of Nodes', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')

# Add value annotations
for i, (nodes, rounds) in enumerate(zip(node_counts_list, avg_rounds)):
    plt.annotate(f'{rounds:.1f}', 
                xy=(nodes, rounds), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/negin/UML/Gossip/pushpull_gossip_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'pushpull_gossip_plot.png'")
plt.close()

# Summary
print("\n" + "=" * 50)
print("PUSH-PULL GOSSIP SUMMARY")
print("=" * 50)
print("Nodes\t\tAvg Rounds\tStd Dev")
print("-" * 50)
for nodes in node_counts_list:
    print(f"{nodes:,}\t\t{all_results[nodes]['avg']:.2f}\t\t{all_results[nodes]['std']:.2f}")
print("=" * 50)
