from qiskit import *
from qiskit_aer import Aer
from qiskit.compiler import transpile

qc = QuantumCircuit.from_qasm_file("file.qasm")

backend = Aer.get_backend("qasm_simulator")
new = transpile(qc, backend)
job = backend.run(new)

result = job.result()
counts = result.get_counts()
print(counts)
