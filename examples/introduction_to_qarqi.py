"""
QARQI Introduction: Library Usage Example
-----------------------------------------
This example demonstrates how to use the QARQI library to load an image,
construct a quantum circuit using Qu-Dits, and simulate the reconstruction.
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from qarqi.core.circuit import QARQICircuit
from qarqi.core.results import QARQIResult
from qarqi.utils.math import angle_map, compute_register
from qarqi.utils.plots import bins_to_grid, grid_to_image

def main():
    # 1. Configuration
    img_path = os.path.join("resources", "lenna.jpg")
    n = 16  # image resolution
    shots = 5000
    
    # 2. Load and Prepare Image
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found. Please ensure you are running from the project root.")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (n, n))
    
    # Normalize and map to RY rotation angles
    theta_map = angle_map(img)
    
    # 3. Generate Coordinate Matrix for Registers
    pol_mag_matrix = []
    for r, c in np.ndindex(n, n):
        pol_mag_matrix.append(compute_register(n, r, c))
    
    # 4. Initialize and Run QARQI Circuit
    # d is the qudit dimension for magnitude registers
    d = round(n / 2)
    circuit = QARQICircuit(d)
    
    print(f"Uploading {n}x{n} image to quantum circuit...")
    circuit.upload_image(n, pol_mag_matrix, theta_map)
    
    print(f"Running simulation with {shots} shots...")
    counts, sv = circuit.simulate(shots=shots)
    
    # 5. Process Results
    result = QARQIResult(counts, d, mode='counts')
    prob_map = result.get_probability_map()
    
    # 6. Reconstruct and Display
    grid = bins_to_grid(prob_map, d)
    recon = grid_to_image(grid, d)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original (16x16)")
    axes[0].axis('off')
    
    axes[1].imshow(recon, cmap='gray')
    axes[1].set_title(f"QARQI Reconstruction ({shots} shots)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
