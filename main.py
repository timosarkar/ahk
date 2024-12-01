from qiskit import *
from qiskit_aer import Aer
from qiskit.compiler import transpile

import json

# load all mapped possible program outputs into memory
outputs = json.loads("outputs.json")

# load QASM code from file
qc = QuantumCircuit.from_qasm_file("file.qasm")

# program loop here--------

#while True:

# program starts with the default state
state = "0000000000000010011011"
print(outputs[state]) # print the value mapped to current program state to stdout


# transpile & run the program from here---------

# initiate new Aer simulator
#backend = Aer.get_backend("qasm_simulator")

#new = transpile(qc, backend)

#job = backend.run(new)
#result = job.result()
#counts = result.get_counts()
#print(counts)
