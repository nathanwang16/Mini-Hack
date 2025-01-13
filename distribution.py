### draws and list important statistics of the activation values of the first 8000 sequences in the dataset

import matplotlib.pyplot as plt
import numpy as np

def load_training_data(file_path):
    """
    Reads a file where each line has at least:
       <sequence_id> <dna_sequence> <(optional) activation values...>

    Returns a list of dictionaries: {
        "id": str,
        "sequence": str,
        "activation_scores": list of floats
    }
    """
    training_data = []
    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue  # skip malformed lines

            seq_id = parts[0]
            dna_seq = parts[1]
            # The rest (if any) are the activation values
            if len(parts) > 2:
                act_vals = list(map(float, parts[2:]))
            else:
                act_vals = []

            training_data.append({
                "id": seq_id,
                "sequence": dna_seq,
                "activation_scores": act_vals
            })
    return training_data

def describe_distribution(data, name="Data"):
    """
    Prints summary statistics for a numeric list (data).
    """
    arr = np.array(data, dtype=float)
    if len(arr) == 0:
        print(f"{name}: No data available.\n")
        return

    print(f"{name} distribution stats:")
    print(f"  Count: {len(arr)}")
    print(f"  Min: {np.min(arr):.4f}")
    print(f"  Max: {np.max(arr):.4f}")
    print(f"  Mean: {np.mean(arr):.4f}")
    print(f"  Std: {np.std(arr):.4f}")
    print(f"  25% (Q1): {np.percentile(arr, 25):.4f}")
    print(f"  50% (Median): {np.median(arr):.4f}")
    print(f"  75% (Q3): {np.percentile(arr, 75):.4f}\n")

def plot_activation_distributions(training_data):
    """
    For the first 8000 sequences:
      1) Compute the average activation of each sequence. 
      2) Gather all activation values (pooled).
      3) Print detailed distribution stats.
      4) Draw histograms for both distributions.
    """
    # Slice the first 8,000 sequences
    subset = training_data[:8000]

    avg_activations = []
    all_activations = []

    for entry in subset:
        acts = entry["activation_scores"]
        if acts:  # Only sequences with non-empty activation
            avg_activations.append(np.mean(acts))
            all_activations.extend(acts)

    # -------------------
    # Print distribution stats
    # -------------------
    describe_distribution(avg_activations, name="Average Activation (per sequence)")
    describe_distribution(all_activations, name="All Activation Values (pooled)")

    # -------------------
    # Plot histograms
    # -------------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Histogram of per-sequence average activations
    axes[0].hist(avg_activations, bins=50, color='blue', edgecolor='black', alpha=0.7)
    axes[0].set_title("Distribution of Avg. Activation (first 8000 seq)")
    axes[0].set_xlabel("Average Activation Score")
    axes[0].set_ylabel("Count of Sequences")

    # Histogram of all activation values
    axes[1].hist(all_activations, bins=50, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_title("Distribution of All Activation Values (first 8000 seq)")
    axes[1].set_xlabel("Activation Score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    data_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/reversed_MPRA.txt"
    training_data = load_training_data(data_file)

    print(f"Loaded {len(training_data)} total sequences from {data_file}.")
    print("Analyzing first 8000 sequences for activation distributions...\n")

    # This function will print detailed stats and plot histograms.
    plot_activation_distributions(training_data)