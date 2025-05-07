import os
import numpy as np
import csv

def npy_to_csv_converter(networks_folder="networks", output_folder="result"):
    """
    Convert neural network weights and biases from .npy files to a single CSV file
    that can be loaded by the load_best_genome function.
    
    Parameters:
    -----------
    networks_folder : str
        Folder containing the .npy weight and bias files
    output_folder : str
        Folder where the CSV file will be saved
    """
    # Make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all .npy files from the networks folder
    npy_files = [f for f in os.listdir(networks_folder) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in {networks_folder}")
        return
    
    # Load all weight and bias matrices
    weights = []
    biases = []
    
    for file in sorted(npy_files):
        filepath = os.path.join(networks_folder, file)
        try:
            data = np.load(filepath, allow_pickle=True)
            
            if 'weights' in file:
                weights.append(data)
            elif 'biases' in file:
                biases.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not weights or not biases:
        print("Could not find all required weight and bias files")
        return
    
    # Flatten all weights and biases into a single 1D array (genome)
    genome = []
    
    # Add weights
    for w in weights:
        genome.extend(w.flatten())
    
    # Add biases
    for b in biases:
        genome.extend(b.flatten())
    
    # Create CSV file
    csv_path = os.path.join(output_folder, "best_genomes.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow(['Generation', 'Fitness'] + [f'Gene_{i}' for i in range(len(genome))])
        # Write genome data (assuming generation 1 and fitness 1.0 - modify as needed)
        writer.writerow([1, 1.0] + list(genome))
    
    print(f"Successfully converted {len(npy_files)} .npy files to {csv_path}")
    print(f"Total genome length: {len(genome)}")

    # Print the expected structure for verification
    expected_length = 0
    for i, w in enumerate(weights):
        expected_length += w.size
        print(f"Weight layer {i}: {w.shape} ({w.size} parameters)")
    
    for i, b in enumerate(biases):
        expected_length += b.size
        print(f"Bias layer {i}: {b.shape} ({b.size} parameters)")
    
    print(f"Expected genome length: {expected_length}")

# Example usage
if __name__ == "__main__":
    npy_to_csv_converter()