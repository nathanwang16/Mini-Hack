import pandas as pd
import random

# Set a random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Probabilities for replacement bases
base_probabilities = {
    "A": 0.3,  # 30%
    "T": 0.3,  # 30%
    "C": 0.2,  # 20%
    "G": 0.2   # 20%
}

# Helper function to replace N with a random base
def replace_N(sequence):
    """Replace 'N' bases with random bases according to the given probabilities."""
    return ''.join(
        random.choices(
            population=["A", "T", "C", "G"],
            weights=[base_probabilities["A"], base_probabilities["T"], 
                     base_probabilities["C"], base_probabilities["G"]]
        )[0] if base == "N" else base
        for base in sequence
    )

# Process the data
def preprocess_dataset(file_path, output_path):
    # Load the data
    data = pd.read_csv(file_path, sep="\t", header=None)

    # Replace 'N' in sequences (assumes sequences are in the second column)
    data[1] = data[1].apply(replace_N)

    # Save the processed dataset
    data.to_csv(output_path, sep="\t", header=False, index=False)

# File paths
input_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/train_MPRA.txt" # Replace with the path to your input file
output_file = "Random_dataset.txt"  # Replace with the desired output file path

# Run the preprocessing function
preprocess_dataset(input_file, output_file)

print(f"Preprocessed dataset saved to {output_file}")