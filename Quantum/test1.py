import numpy as np

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister

simulator = AerSimulator(method='statevector')
print(simulator.name)

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0,1)
circuit.ry(-3.*np.pi/4., 1)

circuit.draw('mpl')

