import numpy as np
import matplotlib.pyplot as plt

def _coords_from_key(b0, b1, x0, x1):
    sx = 1 if b1 == 1 else -1   # x polarity
    sy = 1 if b0 == 1 else -1   # y polarity
    return sx * (x1 + 1), sy * (x0 + 1)

def bins_to_grid(prob_map, d):
    size = 2*d + 1
    off = d
    grid = np.full((size, size), np.nan)
    for (b0, b1, x0, x1), val in prob_map.items():
        x, y = _coords_from_key(b0, b1, x0, x1)
        grid[y + off, x + off] = val
    return grid

def grid_to_image(grid, d, flip_vertical=True):
    """
    Convert the grid back to a standard uint8 image.
    """
    # Delete the empty center row/col
    cropped = np.delete(np.delete(grid, d, axis=0), d, axis=1)
    
    work = np.array(cropped, dtype=float)
    work[np.isnan(work)] = 0.0
    
    # Scale to 0-255
    vmin, vmax = np.nanmin(work), np.nanmax(work)
    if vmax == vmin: vmax = vmin + 1.0
    
    img = (np.clip(work, vmin, vmax) - vmin) / (vmax - vmin)
    img = (img * 255.0).round().astype(np.uint8)
    
    if flip_vertical:
        img = np.flipud(img)
    return img

import os
from datetime import datetime

def get_run_dir():
    """Create and return a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def show_comparison(orig, recon, run_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    if run_dir:
        save_path = os.path.join(run_dir, "comparison.png")
        plt.savefig(save_path)
        print(f"Result saved to {save_path}")
    
    # Non-blocking show if not in a headless environment
    if not os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        plt.show(block=False)
        plt.pause(2)  # Give time for the window to render
        plt.close()
