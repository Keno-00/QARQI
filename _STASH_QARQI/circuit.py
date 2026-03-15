import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit import QuantumRegister
from mqt.qudits.quantum_circuit.gate import ControlData
import utils
from mqt.qudits.simulation import MQTQuditProvider


def QARQI_init(d):
    circuit = QuantumCircuit()
    polarity_reg_i = QuantumRegister("polarity_i", 1, [2]) #inphase sign
    polarity_reg_q = QuantumRegister("polarity_q", 1, [2]) #quadrature sign
    magnitude_reg_i = QuantumRegister("magnitude_i", 1, [d]) #magnitude i value
    magnitude_reg_q = QuantumRegister("magnitude_q", 1, [d]) #magnitude q value
    intensity_reg = QuantumRegister("intensity", 1, [2]) #probability encoded intensity

    QARQI_registers = [polarity_reg_i,polarity_reg_q,magnitude_reg_i,magnitude_reg_q,intensity_reg]
    for reg in (QARQI_registers):
        circuit.append(reg)
    for i in range(0,4):
        circuit.h(i)
    return(circuit,QARQI_registers)

def QARQI_upload_intensity(circuit,reg,N,pol_mag_matrix,img):
    QARQI_register = []
    for i in reg:
        QARQI_register.append(i[0])
    controls = [0,1,2,3]

    for i in pol_mag_matrix:
        ctrl_states = list(i)
        ctrl = ControlData(controls,ctrl_states)
        r,c = utils.compose_rc(N,i[0],i[1],i[2],i[3])
        r = circuit.r(4, [0, 1,float(img[r,c]),0.0] ,ctrl)    
    return circuit


def QARQI_simulate(circuit,shots):
    provider = MQTQuditProvider()
    provider.backends("fake")
    backend = provider.get_backend("faketraps2trits",shots=shots) 
    from mqt.qudits.simulation.noise_tools.noise import NoiseModel
    nm = NoiseModel()
    job = backend.run(circuit, shots=shots, noise_model=nm)
    result = job.result()
    from mqt.qudits.visualisation import draw_qudit_local

    draw_qudit_local(circuit)
    state_vector = result.get_state_vector()
    counts = result.get_counts()
    return counts,state_vector        
