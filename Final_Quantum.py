import numpy as np
import tensorflow as tf
from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import sidechainnet as scn
# Load the data in the appropriate format for training.
dataloader = scn.load(
             with_pytorch="dataloaders",
             batch_size=4, 
             dynamic_batching=False)
print("Available Dataloaders =", list(dataloader.keys()))

batch = next(iter(dataloader['train']))
print("Protein IDs\n   ", batch.pids)
print("Sequences\n   ", batch.seqs.shape)
print("Evolutionary Data\n   ", batch.evos.shape)
print("Secondary Structure\n   ", batch.secs.shape)
print("Angle Data\n   ", batch.angs.shape)
print("Coordinate Data\n   ", batch.crds.shape)
print("Concatenated Data (seq/evo/2ndary)\n   ", batch.seq_evo_sec.shape)
print("Integer sequence")
print("\tShape:", batch.int_seqs.shape)
print("\tEx:", batch.int_seqs[0,:3])
print("1-hot sequence")
print("\tShape:", batch.seqs.shape)
print("\tEx:\n", batch.seqs[0,:3])

# Quantum Circuit Definition
def create_quantum_circuit(n_qubits, params):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)  # Feature Encoding
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)  # Entanglement
    for i in range(n_qubits):
        qc.rx(params[i + n_qubits], i)  # (next slot)Additional Rotations
        qc.ry(params[i + 2 * n_qubits], i) #(2nd slot)
    return qc

# Quantum Layer
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_params, shots=1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        self.params = [Parameter(f'θ_{i}') for i in range(n_params)] #(convert 1= θ_1)
    
    def call(self, inputs):
        outputs = []
        for params in inputs.numpy():
            qc = create_quantum_circuit(self.n_qubits, params)
            transpiled = transpile(qc, self.backend)
            result = self.backend.run(transpiled, shots=self.shots).result()
            counts = result.get_counts()
            expectation = sum(int(k, 2) * v for k, v in counts.items()) / self.shots #k=01 : v=500 
            outputs.append(expectation)
        return tf.convert_to_tensor(outputs)

# Hybrid Model Definition
def create_hybrid_model(input_shape, n_qubits=4):
    inputs = tf.keras.Input(shape=input_shape)
    quantum_output = QuantumLayer(n_qubits, 3 * n_qubits)(inputs) #3rd slot
    x = tf.keras.layers.LSTM(256, activation='relu', return_sequences=True)(quantum_output)
    x = tf.keras.layers.LSTM(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs, outputs)

# Metrics Logger
class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, csv_file="epoch_metrics.csv"):
        super().__init__()
        self.validation_data = validation_data
        self.csv_file = csv_file
        self.epoch_data = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        mse_train = mean_squared_error(y_train, self.model.predict(X_train))
        mae_train = mean_absolute_error(y_train, self.model.predict(X_train))
        rmse_train = np.sqrt(mse_train)
        mse_val = mean_squared_error(y_val, y_pred)
        mae_val = mean_absolute_error(y_val, y_pred)
        rmse_val = np.sqrt(mse_val)
        metrics = {
            'epoch': epoch + 1,
            'train_accuracy': logs.get('accuracy'),
            'train_MSE': mse_train,
            'train_MAE': mae_train,
            'train_RMSE': rmse_train,
            'val_accuracy': logs.get('val_accuracy'),
            'val_MSE': mse_val,
            'val_MAE': mae_val,
            'val_RMSE': rmse_val
        }
        self.epoch_data.append(metrics)
        pd.DataFrame(self.epoch_data).to_csv(self.csv_file, index=False)

# Plot Metrics
def plot_metrics(csv_file):
    df = pd.read_csv(csv_file)
    df.plot(x='epoch', y=['train_MSE', 'val_MSE'], marker='o', title='Epoch vs MSE')
    df.plot(x='epoch', y=['train_MAE', 'val_MAE'], marker='o', title='Epoch vs MAE')
    df.plot(x='epoch', y=['train_accuracy', 'val_accuracy'], marker='o', title='Epoch vs Accuracy')
    plt.show()

# Data and Model Setup
sequence_length, num_amino_acids = 400, 80
input_shape = (sequence_length, num_amino_acids)
X_train, y_train = np.random.rand(100, sequence_length, num_amino_acids), np.random.rand(100, 3)
X_val, y_val = np.random.rand(20, sequence_length, num_amino_acids), np.random.rand(20, 3)

model = create_hybrid_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', 'accuracy', 'rmse'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64, callbacks=[MetricsLogger(validation_data=(X_val, y_val))])

# Save the trained model
model.save("model.h5")
print("Model saved as model.h5")

# Generate and Plot Metrics
plot_metrics("epoch_metrics.csv")
