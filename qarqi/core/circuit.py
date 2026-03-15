import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.simulation.noise_tools.noise import NoiseModel

from ..utils.math import compose_rc

class QARQICircuit:
    """
    Handles QARQI circuit initialization, data upload, and simulation.
    Uses mqt.qudits as the underlying engine.
    """
    def __init__(self, d):
        self.d = d
        self.qc = QuantumCircuit()
        self.registers = self._init_registers()
        self._init_circuit()

    def _init_registers(self):
        polarity_i = QuantumRegister("polarity_i", 1, [2])
        polarity_q = QuantumRegister("polarity_q", 1, [2])
        magnitude_i = QuantumRegister("magnitude_i", 1, [self.d])
        magnitude_q = QuantumRegister("magnitude_q", 1, [self.d])
        intensity = QuantumRegister("intensity", 1, [2])
        
        return {
            "pol_i": polarity_i,
            "pol_q": polarity_q,
            "mag_i": magnitude_i,
            "mag_q": magnitude_q,
            "intensity": intensity
        }

    def _init_circuit(self):
        for reg in self.registers.values():
            self.qc.append(reg)
        # Place control qudits in uniform superposition
        for i in range(4):
            self.qc.h(i)

    def upload_image(self, N, pol_mag_matrix, angle_norm):
        """
        Uploads image intensity using control qudits.
        """
        controls = [0, 1, 2, 3]
        for coord in pol_mag_matrix:
            ctrl_states = list(coord)
            ctrl = ControlData(controls, ctrl_states)
            r, c = compose_rc(N, coord[0], coord[1], coord[2], coord[3])
            # intensity is at index 4
            self.qc.r(4, [0, 1, float(angle_norm[r, c]), 0.0], ctrl)

    def simulate(self, shots=1000, noise=True):
        """
        Runs the simulation using MQTQuditProvider.
        """
        provider = MQTQuditProvider()
        # Using 'fake' backend as in stash
        backend = provider.get_backend("faketraps2trits", shots=shots)
        
        nm = None
        if noise or shots > 0:
            nm = NoiseModel()
        job = backend.run(self.qc, shots=shots, noise_model=nm)
        result = job.result()
        
        return result.get_counts(), result.get_state_vector()

    def compute_ground_truth_statevector(self, N, pol_mag_matrix, angle_norm):
        """
        Manually generates the ideal statevector for the given encoding.
        Useful for verification since mqt-qudits lacks a direct SetStatevector gate.
        """
        dim = 8 * (self.d ** 2)
        state_vector = np.zeros(dim, dtype=complex)
        
        for coord in pol_mag_matrix:
            b0, b1, x0, x1 = coord
            r, c = compose_rc(N, b0, b1, x0, x1)
            theta = angle_norm[r, c]
            
            # Index calculation matching decode_index
            idx_miss = 0 + 2 * (x1 + self.d * (x0 + self.d * (b1 + 2 * b0)))
            idx_hit  = 1 + 2 * (x1 + self.d * (x0 + self.d * (b1 + 2 * b0)))
            
            state_vector[idx_miss] = (1.0 / N) * np.cos(theta / 2.0)
            state_vector[idx_hit]  = (1.0 / N) * np.sin(theta / 2.0)
            
        return state_vector
