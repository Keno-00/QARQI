from collections import defaultdict
import numpy as np
from ..utils.math import decode_index

class QARQIResult:
    """
    Wraps simulation results (counts/statevector) and provides
    methods to process them into binned data.
    """
    def __init__(self, data, d, mode='counts'):
        self.data = data
        self.d = d
        self.mode = mode
        self._bins = None

    @property
    def bins(self):
        if self._bins is None:
            self._bins = self._process_data()
        return self._bins

    def _process_data(self):
        bins = defaultdict(lambda: {"hit": 0.0, "miss": 0.0, "trials": 0.0})
        
        if self.mode == 'counts':
            # Handle dictionary of counts
            if isinstance(self.data, dict):
                for idx, count in self.data.items():
                    real_idx = int(idx) if isinstance(idx, (int, str)) else idx
                    b0, b1, x0, x1, h = decode_index(real_idx, self.d)
                    key = (b0, b1, x0, x1)
                    if h == 1:
                        bins[key]["hit"] += count
                    else:
                        bins[key]["miss"] += count
                    bins[key]["trials"] += count
            # Handle list of samples (mqt-qudits sequence)
            elif isinstance(self.data, list):
                for idx in self.data:
                    real_idx = int(idx)
                    b0, b1, x0, x1, h = decode_index(real_idx, self.d)
                    key = (b0, b1, x0, x1)
                    if h == 1:
                        bins[key]["hit"] += 1
                    else:
                        bins[key]["miss"] += 1
                    bins[key]["trials"] += 1
        elif self.mode == 'statevector':
            # Data is a complex array of amplitudes
            probs = np.abs(self.data) ** 2
            for idx, prob in enumerate(probs):
                if prob > 1e-15:
                    b0, b1, x0, x1, h = decode_index(idx, self.d)
                    key = (b0, b1, x0, x1)
                    if h == 1:
                        bins[key]["hit"] += prob
                    else:
                        bins[key]["miss"] += prob
                    bins[key]["trials"] += prob
        
        return bins

    def get_probability_map(self, eps=0.0):
        """
        Calculates p_hat = (hit + eps) / (trials + 2*eps)
        """
        prob_map = {}
        for key, v in self.bins.items():
            t = v["trials"]
            prob_map[key] = (v["hit"] + eps) / (t + 2*eps) if t > 0 else 0.0
        return prob_map
