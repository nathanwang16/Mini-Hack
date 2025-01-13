import pandas as pd

# DNA complement map
complement_map = str.maketrans("ACGT", "TGCA")

# Helper functions
def reverse_sequence(sequence):
    """Return the reverse of the DNA sequence."""
    return sequence[::-1]

def complement_sequence(sequence):
    """Return the complement of the DNA sequence."""
    return sequence.translate(complement_map)

def reverse_complement_sequence(sequence):
    """Return the reverse complement of the DNA sequence."""
    return reverse_sequence(complement_sequence(sequence))

# Process the data
def extend_dataset(file_path, output_path):
    # Load the data
    data = pd.read_csv(file_path, sep="\t", header=None)

    # Prepare new rows
    new_rows = []
    for _, row in data.iterrows():
        original_id = row[0]
        sequence = row[1]
        activity_values = row[2:].tolist()  # Activity values are in columns 3 to 297 (0-indexed)

        # Generate new sequences and activity values
        reverse = reverse_sequence(sequence)
        reverse_activities = activity_values[::-1]  # Reverse the activity values
        complement = complement_sequence(sequence)
        reverse_complement = reverse_complement_sequence(sequence)
        reverse_complement_activities = activity_values[::-1]  # Reverse the activity values for reverse complement

        # Append reverse, complement, and reverse complement
        new_rows.append([f"{original_id}_reverse", reverse, *reverse_activities])
        new_rows.append([f"{original_id}_complement", complement, *activity_values])
        new_rows.append([f"{original_id}_reverse_complement", reverse_complement, *reverse_complement_activities])

    # Add the new rows to the original data
    extended_data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)

    # Save the extended dataset
    extended_data.to_csv(output_path, sep="\t", header=False, index=False)

# File paths
input_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/Random_dataset.txt"  # Replace with the path to your input file
output_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/reversed_MPRA.txt"  # Replace with the desired output file path

# Run the function
extend_dataset(input_file, output_file)

print(f"Extended dataset saved to {output_file}")