import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import time
import json
from sklearn.metrics import r2_score, mean_squared_error

# ============== Parameters ==============
epochs = 2000
patience = int(0.05 * epochs)
activation_function = "tanh"
info = "atom"                                   # "atom", "FG", "FR"
hidden_layers = [50]
seeds = [1, 2, 3] #, 4, 5, 6, 7, 8, 9, 10]    # multiple runs for averaging

# ============== Path Management ==============
num_layers = len(hidden_layers)
num_neurons = "_".join(map(str, hidden_layers))
subfolder = f"{num_layers}layers_{num_neurons}neurons"

base_dir = "Results_multi_seed"
results_dir = os.path.join(base_dir, info, activation_function, subfolder)
os.makedirs(results_dir, exist_ok=True)

# ============== Data Loading ==============
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 22

if info == "atom":
    X = X[:, :num_atoms]
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups]
elif info == "FR":
    X = X
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# Fixed split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_labels, test_size=0.2, random_state=42
)

# Scaling
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# ============== Cumulative Lists ==============
all_r2_scores = []
all_mse_original = []         # MSE per property (original space)
all_mse_scaled = []           # MSE per property (scaled space)
all_mse_global_original = []  # MSE entire dataset (original space)
all_mse_global_scaled = []    # MSE entire dataset (scaled space)
all_times = []

# ============== Loop over seeds ==============
for s in seeds:
    print(f"\n======= [RUN with seed {s}] =======")
    tf.keras.utils.set_random_seed(s)

    # Model
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for units in hidden_layers:
        net.add(tf.keras.layers.Dense(units, activation=activation_function))
    net.add(tf.keras.layers.Dense(Y_labels.shape[1], activation="linear"))

    net.compile(optimizer='adam', loss='mse', metrics=['mse'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    # Training
    start_time = time.time()
    history = net.fit(
        X_train_scaled, Y_train_scaled,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    training_time = time.time() - start_time

    # Predictions (scaled + original)
    Y_pred_scaled = net.predict(X_test_scaled, verbose=0)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    # R2 score for each property
    r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]

    # MSE scaled (per property)
    mse_scaled = [mean_squared_error(Y_test_scaled[:, i], Y_pred_scaled[:, i]) for i in range(Y_test.shape[1])]
    # MSE original (per property)
    mse_original = [mean_squared_error(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]

    # MSE globale (scaled + original)
    mse_global_scaled = mean_squared_error(Y_test_scaled, Y_pred_scaled)
    mse_global_original = mean_squared_error(Y_test, Y_pred)

    # Store results
    all_r2_scores.append(r2_scores)
    all_mse_scaled.append(mse_scaled)
    all_mse_original.append(mse_original)
    all_mse_global_scaled.append(mse_global_scaled)
    all_mse_global_original.append(mse_global_original)
    all_times.append(training_time)

# ============== Final Statistics ==============
all_r2_scores = np.array(all_r2_scores)
all_mse_scaled = np.array(all_mse_scaled)
all_mse_original = np.array(all_mse_original)

mean_r2 = np.mean(all_r2_scores, axis=0)
std_r2 = np.std(all_r2_scores, axis=0)

mean_mse_scaled = np.mean(all_mse_scaled, axis=0)
std_mse_scaled = np.std(all_mse_scaled, axis=0)

mean_mse_original = np.mean(all_mse_original, axis=0)
std_mse_original = np.std(all_mse_original, axis=0)

# Global MSE
mean_mse_global_scaled = np.mean(all_mse_global_scaled)
std_mse_global_scaled = np.std(all_mse_global_scaled)

mean_mse_global_original = np.mean(all_mse_global_original)
std_mse_global_original = np.std(all_mse_global_original)

mean_time = np.mean(all_times)
std_time = np.std(all_times)

property_names = [
    'μ (D)', 'α (a₀³)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)', 'ε_gap (Ha)',
    '⟨r²⟩ (a₀²)', 'zpve (Ha)', 'U₀ (Ha)', 'U (Ha)', 'H (Ha)',
    'G (Ha)', 'Cᵥ (cal/mol·K)'
]

# ============== Save Results ==============
info_file = os.path.join(results_dir, "results_summary.txt")
with open(info_file, "w", encoding="utf-8") as f:
    f.write("Experiment Information\n")
    f.write("======================\n")
    f.write(f"Info type: {info}\n")
    f.write(f"Activation function: {activation_function}\n")
    f.write(f"Hidden layers: {hidden_layers}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Patience: {patience}\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")

    total_params = net.count_params()
    f.write(f"Total parameters: {total_params}\n\n")

    f.write("Average Metrics (± std)\n")
    f.write("=======================\n")
    for name, r2m, r2s, msem_s, mses_s, msem_o, mses_o in zip(
        property_names, mean_r2, std_r2,
        mean_mse_scaled, std_mse_scaled,
        mean_mse_original, std_mse_original
    ):
        f.write(
            f"{name:15s}: "
            f"R2 = {r2m:.4f} ± {r2s:.4f}, "
            f"MSE (scaled) = {msem_s:.6f} ± {mses_s:.6f}, "
            f"MSE (original) = {msem_o:.6f} ± {mses_o:.6f}\n"
        )

    f.write("\nGlobal Metrics on Test Set\n")
    f.write("==========================\n")
    f.write(f"MSE (scaled, all properties)   = {mean_mse_global_scaled:.6f} ± {std_mse_global_scaled:.6f}\n")
    f.write(f"MSE (original, all properties) = {mean_mse_global_original:.6f} ± {std_mse_global_original:.6f}\n")

    f.write(f"\nAverage training time: {mean_time:.2f}s ± {std_time:.2f}s\n")

print(f"\nResults saved in: {info_file}")
