import pytest
import numpy as np
from qarqi.core.circuit import QARQICircuit
from qarqi.core.results import QARQIResult
from qarqi.utils.math import angle_map, compute_register

def test_statevector_simulation(dummy_image):
    n = 4
    d = 2
    theta_map = angle_map(dummy_image)
    pol_mag_matrix = []
    for r in range(n):
        for c in range(n):
            pol_mag_matrix.append(compute_register(n, r, c))
    
    circuit = QARQICircuit(d)
    # Ground Truth Statevector
    sv = circuit.compute_ground_truth_statevector(n, pol_mag_matrix, theta_map)
    result = QARQIResult(sv, d, mode='statevector')
    
    prob_map = result.get_probability_map()
    assert len(prob_map) == n * n
    # Total trials probability in bins should be 1.0
    total_prob = sum(v["trials"] for v in result.bins.values())
    assert pytest.approx(total_prob, rel=1e-5) == 1.0

def test_shots_simulation(dummy_image):
    n = 4
    d = 2
    theta_map = angle_map(dummy_image)
    pol_mag_matrix = []
    for r in range(n):
        for c in range(n):
            pol_mag_matrix.append(compute_register(n, r, c))
    
    circuit = QARQICircuit(d)
    circuit.upload_image(n, pol_mag_matrix, theta_map)
    
    shots = 500
    counts, sv = circuit.simulate(shots=shots, noise=False)
    
    # Result should be a list of 500 samples
    assert isinstance(counts, list)
    assert len(counts) == shots
    
    result = QARQIResult(counts, d, mode='counts')
    prob_map = result.get_probability_map()
    
    # Check that we have results for our pixels
    assert len(prob_map) > 0
    # Total hits/misses trials should equal shots
    total_trials = sum(v["trials"] for v in result.bins.values())
    assert total_trials == shots

def test_circuit_init():
    d = 2
    circuit = QARQICircuit(d)
    # Check that registers are correctly initialized
    assert circuit.d == d
    assert circuit.qc.num_qudits == 5 # 1+1+1+1 + 1 (intensity)
