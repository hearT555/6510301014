import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Common parameters
pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)  # 70% for training

# First Dataset: Modular Arithmetic
def gen_data_mod(x):
    return (x % pitch) / pitch

t = np.arange(1, N + 1)
y_mod = np.array([gen_data_mod(i) for i in t])

# Second Dataset: Sine Wave with Noise
np.random.seed(42)  # Reproducibility
y_sine = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N)

# Function to convert sequence into matrix format for time-series modeling
def convert_to_matrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])
    return np.array(X), np.array(Y)

# Function to train and evaluate a SimpleRNN model
def train_and_evaluate(data, label):
    # Split into train and test sets
    train, test = data[:n_train], data[n_train:]
    x_train, y_train = convert_to_matrix(train, step)
    x_test, y_test = convert_to_matrix(test, step)

    # Reshape for RNN input
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Build the RNN model
    model = Sequential([
        SimpleRNN(units=40, input_shape=(step, 1), activation="relu"),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model
    history = model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=0)

    # Make predictions
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # Combine predictions
    y_pred = np.concatenate([y_pred_train.flatten(), y_pred_test.flatten()])

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label="Training Loss", color="blue")
    plt.title(f"Training Loss ({label})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot predictions vs. original
    plt.figure(figsize=(12, 6))
    plt.plot(t, data, label="Original Data", color="blue")
    plt.plot(t[step:step + len(y_pred)], y_pred, label="Predicted Data", color="red", linestyle="--")
    plt.axvline(x=n_train, color="purple", linestyle="-", label="Train/Test Split")
    plt.legend()
    plt.title(f"Original vs Predicted ({label})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

# Train and evaluate on Modular Arithmetic Dataset
print("Training on Modular Arithmetic Dataset...")
train_and_evaluate(y_mod, label="Modular Arithmetic")

# Train and evaluate on Sine Wave with Noise Dataset
print("Training on Sine Wave with Noise Dataset...")
train_and_evaluate(y_sine, label="Sine Wave with Noise")
