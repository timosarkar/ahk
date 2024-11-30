OPENQASM 2.0;
include "qelib1.inc";

// Quantum and classical registers
qreg q[4];
creg c[4];

// Apply gates to create a Bell state on q[0] and q[1]
h q[0];
cx q[0], q[1];

// Add additional quantum operations on q[2] and q[3]
h q[2];
cx q[2], q[3];

// Entangle q[1] and q[2] using a controlled-X gate
cx q[1], q[2];

// Apply some single-qubit gates for variety
t q[0];
t q[3];

// Measure all qubits
measure q -> c;

