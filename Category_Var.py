import numpy as np

def categorize_scores_by_percentile(
    in_file, 
    out_file, 
    lowerBoundPercent=2.5, 
    upperBoundPercent=97.5
):
    """
    Reads in_file, calculates percentile-based cutoffs for numeric scores,
    then writes out_file where each numeric score is replaced by 'L', 'M', or 'H':

      - 'L' if score < np.percentile(all_scores, lowerBoundPercent)
      - 'H' if score > np.percentile(all_scores, upperBoundPercent)
      - 'M' otherwise

    Keeps the first two columns (ID, Sequence) unchanged, 
    writes all other columns as L/M/H if numeric, 
    or as-is if non-numeric.

    Parameters
    ----------
    in_file : str
        Path to the original data file.
    out_file : str
        Path to the new file with categories instead of raw numeric scores.
    lowerBoundPercent : float
        Lower percentile threshold, e.g. 2.5 means 2.5th percentile.
    upperBoundPercent : float
        Upper percentile threshold, e.g. 97.5 means 97.5th percentile.
    """

    lines = []
    all_scores = []

    # 1) Read the entire file, store lines and numeric activation values
    with open(in_file, "r") as fh:
        for line in fh:
            line_stripped = line.strip()
            if not line_stripped:
                lines.append(line)
                continue

            parts = line_stripped.split()
            if len(parts) < 2:
                # Not enough columns => keep as is
                lines.append(line)
                continue

            # Attempt to parse columns[2:] as floats
            score_strs = parts[2:]
            numeric_scores = []
            for s in score_strs:
                try:
                    val = float(s)
                    numeric_scores.append(val)
                except ValueError:
                    pass  # not a float => skip

            all_scores.extend(numeric_scores)
            lines.append(line)

    # If no numeric scores found, just return
    if not all_scores:
        print("No numeric activation scores found. Exiting.")
        return

    # 2) Compute percentile cutoffs
    arr = np.array(all_scores, dtype=float)
    lowerCut = np.percentile(arr, lowerBoundPercent)
    upperCut = np.percentile(arr, upperBoundPercent)

    print(f"Total numeric scores: {len(arr)}")
    print(f"Lower percentile {lowerBoundPercent}% => cutoff: {lowerCut:.4f}")
    print(f"Upper percentile {upperBoundPercent}% => cutoff: {upperCut:.4f}\n")

    # 3) Define a function to map a float -> L/M/H
    def map_category(val):
        if val < lowerCut:
            return "L"
        elif val > upperCut:
            return "H"
        else:
            return "M"

    # 4) Rewrite lines: replace numeric columns using the L/M/H scheme
    with open(out_file, "w") as outfh:
        for line in lines:
            line_stripped = line.strip()

            # Handle blank lines
            if not line_stripped:
                outfh.write(line)
                continue

            parts = line_stripped.split()
            if len(parts) < 3:
                # If fewer than 3 columns => only ID, seq => write unchanged
                outfh.write(line)
                continue

            # Keep the first two columns the same
            new_parts = parts[:2]

            # Convert columns[2:] if numeric => L/M/H
            for s in parts[2:]:
                try:
                    val = float(s)
                    category = map_category(val)
                    new_parts.append(category)
                except ValueError:
                    # Not a float => keep original
                    new_parts.append(s)

            # Rebuild the line, preserving newlines
            out_line = " ".join(new_parts) + "\n"
            outfh.write(out_line)

    print(f"Categorized file written to: {out_file}")


# Example usage
if __name__ == "__main__":
    original_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/reversed_MPRA.txt"
    categorized_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/2983_reversed_MPRA_categorized.txt"
    
    # Example: 2.5% and 97.5% => about 5% of data in the L/H tails combined
    categorize_scores_by_percentile(
        in_file=original_file,
        out_file=categorized_file,
        lowerBoundPercent=29.5,
        upperBoundPercent=83.0
    )