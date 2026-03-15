import numpy as np
from collections import defaultdict
from typing import Any, Mapping, Optional

def angle_map(img, bit_depth=8):
    max_val = (1 << bit_depth) - 1  # 255
    u = np.clip(img.astype(np.float64) / max_val, 0.0, 1.0)
    theta = 2.0*np.arcsin(np.sqrt(u))       #RY(theta) to  psi = theta/2
    return theta


def compute_register(N, r, c):
    #polarity bits
    R   = N//2
    midL, midR = R-1, R

    i = 1 if r < R else 0          #top half
    q = 1 if c >= R else 0         #right half

    if   r <  midL: i_p = midL - r
    elif r >  midR: i_p = r - midR
    else:           i_p = 0

    if   c <  midL: q_p = midL - c
    elif c >  midR: q_p = c - midR
    else:           q_p = 0
    return (i, q, i_p, q_p)

def compose_rc(N, i, q, i_p, q_p):

    R = N // 2
    midL, midR = R - 1, R

    #reconstruct r
    if i_p == 0:
        r = midL if i == 1 else midR
    else:
        r = (midL - i_p) if i == 1 else (midR + i_p)

    #reconstruct c
    if q_p == 0:
        c = midR if q == 1 else midL
    else:
        c = (midR + q_p) if q == 1 else (midL - q_p)

    return r, c


def decode_index(i, d):
    h  = i % 2;   i //= 2
    x1 = i % d;   i //= d
    x0 = i % d;   i //= d
    b1 = i % 2;   i //= 2
    b0 = i        # 0 or 1
    return b0, b1, x0, x1, h

def empty_bin():
    return {"miss": 0.0, "hit": 0.0, "trials": 0.0}

def make_bins(counts, d):
    bins = defaultdict(empty_bin)

    if isinstance(counts, dict):
        for i, count in counts.items():
            b0, b1, x0, x1,h = decode_index(i,d)
            key = (b0, b1, x0, x1)
            if h == 1:
                bins[key]["hit"] += count
            else:
                bins[key]["miss"] += count
            bins[key]["trials"] += count
    else:
        for i in counts: #for each i na makuha natin, hanapin natin yung respective register
            #print(i) # i is a state count
            b0, b1, x0, x1,h = decode_index(i,d)
            key = (b0, b1, x0, x1)
            if h == 1:
                bins[key]["hit"] += 1
            else:
                bins[key]["miss"] += 1
            bins[key]["trials"] += 1

    return bins

def p_hat(bins, b0, b1, x0, x1, eps=0.0):
    v = bins[(b0, b1, x0, x1)]
    t = v["trials"]
    return (v["hit"] + eps) / (t + 2*eps) if t else float("nan") # hit over hit+miss


if __name__ == "__main__":
    #print(decode_index(0,4))
    counts = np.array([
    [13, 27, 15, 29, 12, 20, 27, 13, 22,  9, 26, 25,  4, 18,  2,  2, 26, 21, 12, 21, 25, 13, 23, 13,
     21, 15, 25, 25,  5, 18,  8, 21],
    ])

    bins = make_bins(counts,2)

    b0,b1,x0,x1,_= decode_index(0,4)
    #print(p_hat(bins,b0,b1,x0,x1))