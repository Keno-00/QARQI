# QARQI: Quantum Image Processing

QARQI (Quantum Adaptive Reconstruction of Quaternary Images) is a Python-based project for simulating quantum image processing and reconstruction using qudits.

## Features
- **Quantum Image Initialization**: Methods to initialize quantum circuits with polarity and magnitude registers.
- **Intensity Upload**: Functionality to upload image intensity data into the quantum state.
- **Simulation**: Integration with `mqt.qudits` for simulating quantum operations with noise models.
- **Visualization**: Tools for plotting counts, grids, and comparing reconstructed images with ground truth.

## Project Structure
- `main.py`: Main entry point for running simulations, performing tests, and saving results.
- `circuit.py`: Defines the quantum circuits and registers used in QARQI.
- `utils.py`: Utility functions for mapping angles, computing registers, and processing bins.
- `plots.py`: Visualization utilities for displaying grids, images, and performance metrics (MSE/PSNR).

## Getting Started

### Prerequisites
- Python 3.8+
- OpenCV (`cv2`)
- Matplotlib
- NumPy
- `mqt.qudits`

### Usage
Run the main simulation with default parameters:
```bash
python main.py
```

## Results
Simulations track performance metrics like Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) across different shot counts. Results are appended to `qarqi_runs.csv` and visualized using plots.
