import math
import numpy as np

def angle_map(img, bit_depth=8):
    """
    Map image intensity to RY rotation angles.
    psi = theta/2, so theta = 2*arcsin(sqrt(norm_intensity))
    """
    max_val = (1 << bit_depth) - 1
    u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(u))
    return theta

def compute_register(N, r, c):
    """
    Map pixel (r, c) to QARQI register states: (i, q, i_p, q_p).
    i, q are polarity bits (quadrants).
    i_p, q_p are magnitudes relative to the image center.
    """
    R = N // 2
    midL, midR = R - 1, R

    i = 1 if r < R else 0          # top half
    q = 1 if c >= R else 0         # right half

    if r < midL: i_p = midL - r
    elif r > midR: i_p = r - midR
    else: i_p = 0

    if c < midL: q_p = midL - c
    elif c > midR: q_p = c - midR
    else: q_p = 0
    
    return (i, q, i_p, q_p)

def compose_rc(N, i, q, i_p, q_p):
    """
    Reconstruct pixel (r, c) from QARQI register states.
    """
    R = N // 2
    midL, midR = R - 1, R

    # reconstruct r
    if i_p == 0:
        r = midL if i == 1 else midR
    else:
        r = (midL - i_p) if i == 1 else (midR + i_p)

    # reconstruct c
    if q_p == 0:
        c = midR if q == 1 else midL
    else:
        c = (midR + q_p) if q == 1 else (midL - q_p)

    return r, c

def decode_index(index, d):
    """
    Decode a simulation result index into QARQI register values.
    Pattern: b0, b1, x0, x1, h
    where h is the intensity (hit/miss) bit.
    """
    idx = index
    h = idx % 2; idx //= 2
    x1 = idx % d; idx //= d
    x0 = idx % d; idx //= d
    b1 = idx % 2; idx //= 2
    b0 = idx
    return b0, b1, x0, x1, h
