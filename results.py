import math
import csv
import matplotlib.pyplot as plt
from Bio import motifs
from tqdm import tqdm  # For showing a progress bar

def load_training_data(file_path):
    """
    Reads a file where each line has at least:
       <sequence_id> <dna_sequence> <(optional) activation values...>

    Returns a list of dictionaries: {
        "id": ...,
        "sequence": ...,
        "activation_scores": [...]
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

def find_peak_pssm_position_for_motif(seq, motif):
    """
    Convert motif -> PWM -> PSSM. 
    Use pssm.search(seq, threshold=-9999999) to gather *all* positions.
    Return (best_start, best_center, best_score, all_positions_and_scores).
    """
    pwm = motif.counts.normalize(pseudocounts=0.1)
    pssm = pwm.log_odds()
    motif_len = motif.length

    best_score = float("-inf")
    best_start = None
    positions_and_scores = []

    for position, score in pssm.search(seq, threshold=10):
        positions_and_scores.append((position, score))
        if score > best_score:
            best_score = score
            best_start = position

    if best_start is not None:
        best_center = best_start + (motif_len // 2)
        return best_start, best_center, best_score, positions_and_scores
    else:
        return None, None, None, []

def find_best_motif_for_sequence(seq, motifs_list):
    """
    Among all motifs in motifs_list, find which motif yields the single
    highest PSSM score on 'seq'.
    Returns (best_motif, best_start, best_center, best_score, pos_scores).
    If no valid windows, returns (None, None, None, None, []).
    """
    best_info = {
        "motif": None,
        "start": None,
        "center": None,
        "score": float("-inf"),
        "positions_scores": []
    }

    for motif in motifs_list:
        start, center, sc, pos_scores = find_peak_pssm_position_for_motif(seq, motif)
        if sc is not None and sc > best_info["score"]:
            best_info["motif"] = motif
            best_info["start"] = start
            best_info["center"] = center
            best_info["score"] = sc
            best_info["positions_scores"] = pos_scores

    if best_info["score"] == float("-inf"):
        return None, None, None, None, []
    else:
        return (best_info["motif"],
                best_info["start"],
                best_info["center"],
                best_info["score"],
                best_info["positions_scores"])

def analyze_sequences(
    training_data, motifs_list, output_csv="results.csv"
):
    """
    1) For each sequence in training_data, find the best motif alignment.
    2) Compare the best motif center with:
         - the max-activation nucleotide position (highest)
         - the min-activation nucleotide position (lowest)
    3) Store results in a list and also write them to a CSV file.
    4) Return the results for later plotting or analysis.
    """
    results = []

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow([
            "seq_id",
            "max_act_position",
            "max_act_value",
            "min_act_position",
            "min_act_value",
            "best_motif_name",
            "best_start",
            "motif_center",
            "best_score",
            "overlap_distance_max",   # |max_act_position - motif_center|
            "overlap_distance_min",   # |min_act_position - motif_center|
            "best_subsequence"
        ])

        # Wrap the loop with tqdm for a progress bar
        for entry in tqdm(training_data, desc="Analyzing Sequences", unit="seq"):
            seq_id = entry["id"]
            seq = entry["sequence"]
            acts = entry["activation_scores"]

            # If no activation data, skip or set placeholders
            if len(acts) > 0:
                max_act_value = max(acts)
                max_act_position = acts.index(max_act_value)
                min_act_value = min(acts)
                min_act_position = acts.index(min_act_value)
            else:
                max_act_value = None
                max_act_position = None
                min_act_value = None
                min_act_position = None

            # Find best motif alignment among all motifs
            best_motif, best_start, best_center, best_score, pos_scores = find_best_motif_for_sequence(
                seq, motifs_list
            )

            if best_motif is not None:
                best_subsequence = seq[best_start : best_start + best_motif.length]

                # Overlap distances
                if max_act_position is not None:
                    overlap_distance_max = abs(max_act_position - best_center)
                else:
                    overlap_distance_max = None

                if min_act_position is not None:
                    overlap_distance_min = abs(min_act_position - best_center)
                else:
                    overlap_distance_min = None

                # Store in local list
                results.append({
                    "seq_id": seq_id,
                    "max_act_position": max_act_position,
                    "max_act_value": max_act_value,
                    "min_act_position": min_act_position,
                    "min_act_value": min_act_value,
                    "best_motif_name": best_motif.name,
                    "best_start": best_start,
                    "motif_center": best_center,
                    "best_score": best_score,
                    "overlap_distance_max": overlap_distance_max,
                    "overlap_distance_min": overlap_distance_min,
                    "best_subsequence": best_subsequence
                })

                # Also write to CSV
                writer.writerow([
                    seq_id,
                    max_act_position,
                    f"{max_act_value:.4f}" if max_act_value is not None else "None",
                    min_act_position,
                    f"{min_act_value:.4f}" if min_act_value is not None else "None",
                    best_motif.name,
                    best_start,
                    best_center,
                    f"{best_score:.4f}",
                    overlap_distance_max,
                    overlap_distance_min,
                    best_subsequence
                ])
            else:
                # If no valid alignment found
                results.append({
                    "seq_id": seq_id,
                    "max_act_position": max_act_position,
                    "max_act_value": max_act_value,
                    "min_act_position": min_act_position,
                    "min_act_value": min_act_value,
                    "best_motif_name": None,
                    "best_start": None,
                    "motif_center": None,
                    "best_score": None,
                    "overlap_distance_max": None,
                    "overlap_distance_min": None,
                    "best_subsequence": None
                })
                # Write partial info to CSV
                writer.writerow([
                    seq_id,
                    max_act_position,
                    f"{max_act_value:.4f}" if max_act_value is not None else "None",
                    min_act_position,
                    f"{min_act_value:.4f}" if min_act_value is not None else "None",
                    "None",
                    "None",
                    "None",
                    "None",
                    "None",
                    "None",
                    "None"
                ])

    return results

def plot_overlap_distance_hist(results):
    """
    Plot histograms of:
      - Overlap distance with max activation
      - Overlap distance with min activation
    for all sequences that have valid data.
    """
    # Filter out None
    overlap_vals_max = [
        r["overlap_distance_max"] for r in results 
        if r["overlap_distance_max"] is not None
    ]
    overlap_vals_min = [
        r["overlap_distance_min"] for r in results 
        if r["overlap_distance_min"] is not None
    ]

    if not overlap_vals_max and not overlap_vals_min:
        print("No valid overlap distances to plot.")
        return

    plt.figure(figsize=(12, 5))
    
    # 1) Overlap distance max histogram
    plt.subplot(1, 2, 1)
    if overlap_vals_max:
        plt.hist(overlap_vals_max, bins=20, color='green', edgecolor='black')
        plt.title("Overlap Distance (Max Activation)")
        plt.xlabel("Distance = |maxActPos - motifCenter|")
        plt.ylabel("Count")
    else:
        plt.title("No valid max activation overlap data.")
    
    # 2) Overlap distance min histogram
    plt.subplot(1, 2, 2)
    if overlap_vals_min:
        plt.hist(overlap_vals_min, bins=20, color='blue', edgecolor='black')
        plt.title("Overlap Distance (Min Activation)")
        plt.xlabel("Distance = |minActPos - motifCenter|")
        plt.ylabel("Count")
    else:
        plt.title("No valid min activation overlap data.")

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Example main script usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Parse multiple JASPAR motifs from file
    jaspar_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
    with open(jaspar_file, "r") as fh:
        motifs_list = motifs.parse(fh, "jaspar")
    print(f"Loaded {len(motifs_list)} motifs from {jaspar_file}.")

    # 2) Load training data with sequences + activation
    training_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/train_MPRA.txt"
    training_data = load_training_data(training_file)
    print(f"Loaded {len(training_data)} sequences from {training_file}.")

    # 3) Analyze all sequences => best motif + compare to max activation
    csv_output = "analysis_results.csv"
    results = analyze_sequences(training_data, motifs_list, output_csv=csv_output)
    print(f"Analysis complete. Results written to {csv_output}.")

    # 4) Plot a histogram of the overlap distance across all sequences
    plot_overlap_distance_hist(results)