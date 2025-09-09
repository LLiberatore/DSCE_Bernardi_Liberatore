import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import os
import json
import time
from Utils import extract_history, save_experiment_info
from sklearn.metrics import r2_score
from Plot import plot_training_curves, plot_parity_plots

epochs = 2000
patience = int(0.2 * epochs)   # 5% of total epochs
activation_function = "tanh"    # "relu", "selu", "tanh"
info = "FR"                     # "atom", "FG", "FR"

hidden_layers = [54]
Load_model = False
model_path = "trained_model.keras"
history_path = "training_history.json"

# ------------ Path Management ---------------
num_layers = len(hidden_layers)
num_neurons = "_".join(map(str, hidden_layers))
subfolder = f"{num_layers}layers_{num_neurons}neurons"

base_dir = "saved_models"
model_dir = os.path.join(base_dir, info, activation_function, subfolder)
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.keras")
history_path = os.path.join(model_dir, "training_history.json")

# ------------ Data Loading and Preprocessing ---------------
# Load preprocessed data
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 17  # after filtering, 22 functional groups remain

if info == "atom":
    X = X[:, :num_atoms]
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups]
elif info == "FR":
    X = X  # keep all columns
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42)

# Scale inputs
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale outputs (trainin only)
scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)   

# ------------ Model Definition, Training, and Evaluation ---------------
if Load_model and os.path.exists(model_path):
    print(f"------- [INFO] Loading pre-trained model from {model_path} -------")
    net = tf.keras.models.load_model(model_path)
    
    if os.path.exists(history_path):   # Load training history
        with open(history_path, "r") as f: 
            history_dict = json.load(f)
            MSE_training_history, MSE_val_history = extract_history(history_dict)
else:
    print("------- [INFO] Training model from scratch -------")
    # Define network
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for units in hidden_layers:
        net.add(tf.keras.layers.Dense(units, activation=activation_function))
    net.add(tf.keras.layers.Dense(Y_labels.shape[1], activation="linear"))
    
    # Compile
    net.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',           # monitor validation loss
        patience = patience,            # stop if no improvement after 20% of total epochs
        restore_best_weights = True     # restore weights from best epoch
    )
    
    # Train
    start_time = time.time()
    history = net.fit(X_train_scaled, Y_train_scaled, validation_split = 0.2, epochs=epochs, batch_size=32, verbose=1, callbacks=[early_stop])
    training_time = time.time() - start_time
    
    # Extract and save training history to JSON
    MSE_training_history, MSE_val_history = extract_history(history.history)
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    
    # Save model after training
    print("------- [INFO] Trained model saved -------")
    net.save(model_path)

# Predict (rescale predictions)
Y_pred_scaled = net.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# r2 score for each property
r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]

# Property names + u.m.
property_names = [
    'μ (D)',              # dipole moment
    'α (a₀³)',            # isotropic polarizability
    'ε_HOMO (Ha)',        # energy of HOMO
    'ε_LUMO (Ha)',        # energy of LUMO
    'ε_gap (Ha)',         # HOMO–LUMO gap
    '⟨r²⟩ (a₀²)',          # electronic spatial extent
    'zpve (Ha)',          # zero-point vibrational energy
    'U₀ (Ha)',            # internal energy at 0 K
    'U (Ha)',             # internal energy at 298.15 K
    'H (Ha)',             # enthalpy at 298.15 K
    'G (Ha)',             # free energy at 298.15 K
    'Cᵥ (cal/mol·K)'      # heat capacity at 298.15 K
]

# ------------- Plot Section -------------
plots_base_dir = "Plots"
plots_dir = os.path.join(plots_base_dir, info, activation_function, subfolder)
os.makedirs(plots_dir, exist_ok=True)

# Plot training curves
plot_training_curves(MSE_training_history, MSE_val_history, plots_dir)

# Parity plot 4x3 grid for all the molecules
plot_parity_plots(Y_test, Y_pred, r2_scores, property_names, plots_dir)

# ---------------- Save experiment info and metrics -------------------------
save_experiment_info(plots_dir, info, activation_function, hidden_layers, epochs, patience, X_train, X_test, property_names, r2_scores, Y_test, Y_pred, Y_test_scaled, Y_pred_scaled, net, training_time)
