import numpy as np
import os
import re

def combine_npy_files_by_pattern(input_dir, file_pattern, output_file):
    """
    Combine .npy files that match a given pattern found in input_dir into a single .npy file,
    strictly following the numerical order indicated in the filenames.

    Parameters:
    - input_dir: Directory containing the .npy files to combine.
    - file_pattern: A regex pattern to match the files. Must contain a group for the number.
    - output_file: Path to the output .npy file.
    """
    # Compile the regex pattern for matching filenames
    pattern = re.compile(file_pattern)

    # List all files in the input directory
    files = os.listdir(input_dir)

    # Filter and sort files based on the pattern and the numerical order
    filtered_files = sorted(
        (file for file in files if pattern.match(file)),
        key=lambda x: int(pattern.match(x).group(1))
    )

    # Initialize an empty list to hold the data from each .npy file
    combined_data = []

    # Loop over the sorted list of .npy files
    for npy_file in filtered_files:
        # Construct the full path to the .npy file
        file_path = os.path.join(input_dir, npy_file)
        # Load the .npy file
        data = np.load(file_path)
        # Append the data to the combined_data list
        combined_data.append(data)

    # Concatenate all the data along the first axis
    combined_data = np.concatenate(combined_data, axis=0)

    # Save the combined data to the specified output file
    np.save(output_file, combined_data)

    print(f"Combined {len(filtered_files)} files into {output_file}")

# Example usage
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="Beijing")

    args = parser.parse_args()
    dataset = args.dataset

    input_dir = f'embeddings/{dataset}/'  # Directory containing your .npy files
    # Combine files matching 'poi_X.npy'
    combine_npy_files_by_pattern(input_dir, r'poi_(\d+)\.npy', f'embeddings/{dataset}/poi_embeddings.npy')
    # Combine files matching 'test_X.npy'
    combine_npy_files_by_pattern(input_dir, r'query_(\d+)\.npy', f'embeddings/{dataset}/test_embeddings.npy')
