import cv2
import numpy as np
import argparse
from ..core.circuit import QARQICircuit
from ..core.results import QARQIResult
from ..utils.math import angle_map, compute_register
from ..utils.plots import bins_to_grid, grid_to_image, show_comparison, get_run_dir

def run_simulation(img_path, n=8, shots=1000, use_statevector=False):
    # 1. Load and process image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not find image at {img_path}")
        return
    img = cv2.resize(img, (n, n))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    N = img.shape[0]
    d = round(N / 2)
    
    # 2. Prepare metadata
    theta_map = angle_map(img)
    pol_mag_matrix = []
    for r, c in np.ndindex(N, N):
        pol_mag_matrix.append(compute_register(N, r, c))
    
    # 3. Build and simulate circuit
    circuit = QARQICircuit(d)
    if use_statevector:
        print("Using Ideal Ground Truth Statevector...")
        sv = circuit.compute_ground_truth_statevector(N, pol_mag_matrix, theta_map)
        result = QARQIResult(sv, d, mode='statevector')
    else:
        circuit.upload_image(N, pol_mag_matrix, theta_map)
        counts, sv = circuit.simulate(shots=shots)
        result = QARQIResult(counts, d, mode='counts')
    
    # 4. Process results
    prob_map = result.get_probability_map()
    
    # 5. Reconstruct and show
    grid = bins_to_grid(prob_map, d)
    recon_img = grid_to_image(grid, d)
    
    run_dir = get_run_dir()
    show_comparison(img, recon_img, run_dir=run_dir)

def main_cli():
    parser = argparse.ArgumentParser(description="QARQI: Quantum Architecture for Real-time Qu-DIT Imaging")
    parser.add_argument("--img", type=str, default="resources/lenna.jpg", help="Path to input image")
    parser.add_argument("-n", type=int, default=8, help="Image size (nxn)")
    parser.add_argument("--counts", "--shots", type=int, default=1000, help="Number of shots/counts")
    parser.add_argument("--statevector", action="store_true", help="Use ideal ground truth statevector")
    
    args = parser.parse_args()
    run_simulation(args.img, args.n, args.counts, args.statevector)

if __name__ == "__main__":
    main_cli()
