import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import os
import json
import time
from sklearn.metrics import r2_score

# ================= Configuration =================
epochs = 2000
patience = int(0.05 * epochs)
activation_function = "relu"
info = "FR"   # "atom", "FG", "FR"
SEEDS = [0, 1,2 ,3 ,4 ]   

architectures = {
    2500:  {1: [27],   2: [22, 22],   3: [19, 19, 19]},
    5000:  {1: [54],   2: [38, 38],   3: [32, 32, 32]},
    7500:  {1: [81],   2: [51, 51],   3: [42, 42, 42]},
    10000: {1: [107],  2: [63, 63],   3: [51, 51, 51]},
    12500: {1: [134],  2: [74, 74],   3: [59, 59, 59]},
}


# ================= Data Loading =================
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 22  # after filtering

if info == "atom":
    X = X[:, :num_atoms]
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups]
elif info == "FR":
    X = X
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)   

# ================= Loop over architectures =================
results_summary = []

for target_params, layers_dict in architectures.items():
    for num_layers, hidden_layers in layers_dict.items():
        print(f"\n===== Training Target={target_params}, Layers={num_layers}, Arch={hidden_layers} =====")

        run_r2 = []
        run_mse_scaled = []
        run_times = []

        for seed in SEEDS:
            tf.keras.utils.set_random_seed(seed)

            # Build model
            net = tf.keras.models.Sequential()
            net.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
            for units in hidden_layers:
                net.add(tf.keras.layers.Dense(units, activation=activation_function))
            net.add(tf.keras.layers.Dense(Y_labels.shape[1], activation="linear"))

            net.compile(optimizer='adam', loss='mse')

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )

            start_time = time.time()
            history = net.fit(
                X_train_scaled, Y_train_scaled,
                validation_split=0.2,
                epochs=epochs,
                batch_size=32,
                verbose=1,
                callbacks=[early_stop]
            )
            training_time = time.time() - start_time

            # Predictions (scaled)
            Y_pred_scaled = net.predict(X_test_scaled, verbose=0)
            Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

            # Metrics
            r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]
            mse_test_scaled = np.mean((Y_test_scaled - Y_pred_scaled)**2)

            run_r2.append(np.mean(r2_scores))
            run_mse_scaled.append(mse_test_scaled)
            run_times.append(training_time)

        # Save average results
        avg_r2, std_r2 = np.mean(run_r2), np.std(run_r2)
        avg_mse, std_mse = np.mean(run_mse_scaled), np.std(run_mse_scaled)
        avg_time, std_time = np.mean(run_times), np.std(run_times)

        results_summary.append({
            "TargetParams": int(target_params),
            "NumLayers": int(num_layers),
            "HiddenLayers": [int(x) for x in hidden_layers],
            "ParamsActual": int(target_params),
            "R2_mean": float(avg_r2), "R2_std": float(std_r2),
            "MSE_scaled_mean": float(avg_mse), "MSE_scaled_std": float(std_mse),
            "Time_mean": float(avg_time), "Time_std": float(std_time)
        })

        # Save to file
        out_dir = os.path.join("Results_multi", info, activation_function, f"{target_params}_params", f"{num_layers}layers")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(json.dumps(results_summary[-1], indent=4))

# Save global summary
df = pd.DataFrame(results_summary)
df.to_csv("results_all_architectures.csv", index=False)

# ================= Final Plot: Scaled MSE vs Parameters =================
plt.figure(figsize=(8,6))
colors = {1:'tab:blue', 2:'tab:orange', 3:'tab:green'}

for num_layers in [1,2,3]:
    subset = df[df["NumLayers"]==num_layers]
    subset = subset.sort_values("TargetParams")
    plt.errorbar(subset["TargetParams"], subset["MSE_scaled_mean"], 
                 yerr=subset["MSE_scaled_std"],
                 marker='o', linestyle='-', color=colors[num_layers], 
                 label=f"{num_layers} layers")

plt.xlabel("Number of parameters")
plt.ylabel("Test MSE")
plt.title("MSE vs. Model Complexity")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save plot ---
out_dir = "Results_multi"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "MSE_vs_Params_scaled.png"), dpi=300)
plt.show()

# ================= Final Global Summary =================
out_dir = "Results_multi"
os.makedirs(out_dir, exist_ok=True)

summary_file = os.path.join(out_dir, "summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("==== GLOBAL SUMMARY OF ALL ARCHITECTURES ====\n\n")
    for res in results_summary:
        f.write(f"Target parameters : {res['TargetParams']}\n")
        f.write(f"Actual parameters : {res['ParamsActual']}\n")
        f.write(f"Num layers        : {res['NumLayers']}\n")
        f.write(f"Hidden layers     : {res['HiddenLayers']}\n")
        f.write(f"R² mean ± std     : {res['R2_mean']:.4f} ± {res['R2_std']:.4f}\n")
        f.write(f"MSE_test_scaled   : {res['MSE_scaled_mean']:.6e} ± {res['MSE_scaled_std']:.6e}\n")
        f.write(f"Training time [s] : {res['Time_mean']:.2f} ± {res['Time_std']:.2f}\n")
        f.write("-"*50 + "\n")

print(f"[INFO] Global summary saved to {summary_file}")
