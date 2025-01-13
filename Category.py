import numpy as np

def categorize_scores_2std(in_file, out_file):
    """
    Reads in_file, determines mean (mu) and std (sigma) from all activation scores,
    then writes out_file where each numeric activation value is replaced
    by 'L', 'M', or 'H' based on 2 std dev boundaries:
        L if val < mu - 2*sigma
        H if val > mu + 2*sigma
        M otherwise
    """

    # 1) Read the entire file, store lines and activation values
    lines = []
    all_scores = []
    
    with open(in_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                # Keep empty lines if needed
                lines.append(line)
                continue

            parts = line.split()
            if len(parts) < 2:
                # If not enough columns, just store as-is
                lines.append(line)
                continue

            # Attempt to parse the activation scores from columns[2:]
            score_strs = parts[2:]
            numeric_scores = []
            for s in score_strs:
                try:
                    val = float(s)
                    numeric_scores.append(val)
                except ValueError:
                    # Not a float
                    pass

            all_scores.extend(numeric_scores)
            lines.append(line)

    if not all_scores:
        print("No numeric activation scores found. Exiting.")
        return

    # 2) Compute mean and std
    arr = np.array(all_scores, dtype=float)
    mu = np.mean(arr)
    sigma = np.std(arr)

    lower_bound = mu - 2*sigma
    upper_bound = mu + 2*sigma

    print(f"Total numeric scores: {len(arr)}")
    print(f"Mean (mu): {mu:.4f}")
    print(f"Std (sigma): {sigma:.4f}")
    print(f"Lower bound: {lower_bound:.4f} = mu - 2*sigma")
    print(f"Upper bound: {upper_bound:.4f} = mu + 2*sigma\n")

    # 3) Define mapping function for each score
    def map_category(val):
        if val < lower_bound:
            return "L"
        elif val > upper_bound:
            return "H"
        else:
            return "M"

    # 4) Rewrite lines: replace numeric scores using the L/M/H scheme
    with open(out_file, "w") as outfh:
        for line in lines:
            if not line.strip():
                # Empty line
                outfh.write(line + "\n")
                continue

            parts = line.split()
            if len(parts) < 3:
                outfh.write(line + "\n")
                continue

            new_parts = parts[:2]  # Keep ID, sequence unchanged

            for s in parts[2:]:
                try:
                    val = float(s)
                    category = map_category(val)
                    new_parts.append(category)
                except ValueError:
                    # Non-numeric
                    new_parts.append(s)

            out_line = " ".join(new_parts)
            outfh.write(out_line + "\n")

    print(f"Categorized file written to: {out_file}")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    original_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/reversed_MPRA.txt"
    categorized_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/reversed_MPRA_categorized.txt"

    categorize_scores_2std(original_file, categorized_file)