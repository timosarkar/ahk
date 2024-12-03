from python import Python

qml = Python.import_module("pennylane")
np = Python.import_module("pennylane.numpy")
Python.import_module("sklearn.preprocessing")
Python.import_module("safetensors")
plt = Python.import_module("matplotlib.pyplot")

# Step 1: Prepare the dataset
data = np.array([
    [30, 0, 1],  # 30°C, no rain -> can run
    [22, 1, 0],  # 22°C, raining -> cannot run
    [25, 0, 1],  # 25°C, no rain -> can run
    [15, 1, 0],  # 15°C, raining -> cannot run
    [20, 0, 1],  # 20°C, no rain -> can run
])

X = data[:, :2]
y = data[:, 2]

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Step 2: Quantum device and QNode
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(weights, x):
    """Quantum circuit that encodes data and applies variational parameters."""
    # Encode input features using angle encoding
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    # Apply variational layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measure expectation value for classification
    return qml.expval(qml.PauliZ(0))

# QNode
@qml.qnode(dev)
def quantum_model(weights, x):
    return quantum_circuit(weights, x)

# Step 3: Define weights and hybrid cost function
n_layers = 3  # Increased layers for better learning
weights_shape = (n_layers, n_qubits, 3)  # Correct shape for StronglyEntanglingLayers
weights = np.random.uniform(low=-0.1, high=0.1, size=weights_shape)  # Improved initialization

def cost(weights, X, y):
    predictions = np.array([quantum_model(weights, x) for x in X])
    # Clip predictions to avoid log(0) or log(1)
    epsilon = 1e-9
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    bce_loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return bce_loss


def accuracy(weights, X, y):
    predictions = np.array([quantum_model(weights, x) for x in X])
    predictions_binary = (predictions >= 0.5).astype(int)
    return np.mean(predictions_binary == y)

# Step 4: Train the model
opt = qml.AdamOptimizer(stepsize=0.01)
epochs = 400
losses = []
accuracies = []

for epoch in range(epochs):
    weights = opt.step(lambda w: cost(w, X_normalized, y), weights)
    current_loss = cost(weights, X_normalized, y)
    current_acc = accuracy(weights, X_normalized, y)
    losses.append(current_loss)
    accuracies.append(current_acc)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {current_loss:.4f}, Accuracy = {current_acc:.4f}")

# Save the trained weights in safetensor format
trained_weights = {"weights": weights}
save_file(trained_weights, "out/model.safetensors")

# Save the QASM2 circuit
qasm = quantum_model.qtape.to_openqasm()

with open('out/circuit.qasm', 'w') as file:
    file.write(qasm)

# Step 5: Plot training loss and accuracy
# Save training loss and accuracy plots
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(range(epochs), accuracies, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig("out/metrics.png")
