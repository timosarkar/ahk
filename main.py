from qiskit import *
from qiskit_aer import Aer
from qiskit.compiler import transpile

# qasm_str = """OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[2];
# creg c[2];
# h q[0];
# cx q[0],q[1];
# measure q -> c;
# """

# qc = QuantumCircuit.from_qasm_str(qasm_str)
qc = QuantumCircuit.from_qasm_file("file.qasm")

backend = Aer.get_backend("qasm_simulator")
new = transpile(qc, backend)
job = backend.run(new)

result = job.result()
counts = result.get_counts()
print(counts)
