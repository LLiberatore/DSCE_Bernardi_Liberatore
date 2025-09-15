# Random Forest Regression on Molecular Properties (with k-fold cross-validation)
import os
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from Plots import plot_parity_plots, plot_rf_r2_vs_time

import matplotlib.pyplot as plt

# ================= Configuration =================
info = "FR"                # "atom", "FG", "FR"
n_estimators_list = [100]   # puoi provare più valori
k = 5

cv = KFold(n_splits=k, shuffle=True, random_state=42)

# ------------- Path Management -------------
model_dir = os.path.join("Results", "RF", info)  # folder for model results
os.makedirs(model_dir, exist_ok=True)
plots_dir = os.path.join("Plots", "ML", info)   # folder for plots
os.makedirs(plots_dir, exist_ok=True)

# ================= Data Loading =================
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 20  # after filtering, 20 functional groups remain

# input selection
if info == "atom":
    X = X[:, :num_atoms]                   # Select atomic features
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups] # Select atomic and functional group features
elif info == "FR":
    X = X                                  # Select all features
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# Train/test split (per parity plots)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42)
property_names = ['μ (D)', 'α (a₀³)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)','ε_gap (Ha)', '⟨r²⟩ (a₀²)', 'zpve (Ha)', 'U₀ (Ha)','U (Ha)', 'H (Ha)', 'G (Ha)', 'Cᵥ (cal/mol·K)']

# ================= Loop over n_estimators =================
results = []
for n_estimators in n_estimators_list:
    print(f"\n[INFO] Training Random Forest with n_estimators = {n_estimators}")

    subfolder = f"n_estimators_{n_estimators}"
    start_time = time.time()
    r2_scores_mean = []
    r2_scores_std = []
    Y_pred = np.zeros_like(Y_test)

    # Loop over properties
    for i in range(Y_labels.shape[1]):
        model = RandomForestRegressor(n_estimators,random_state=42,n_jobs=-1)

        # Cross-validation R2
        scores = cross_val_score(model, X, Y_labels[:, i], cv=cv, scoring="r2")
        r2_scores_mean.append(np.mean(scores))
        r2_scores_std.append(np.std(scores))

        # Train once on train/test split for parity plots
        model.fit(X_train, Y_train[:, i])
        Y_pred[:, i] = model.predict(X_test)
        
    training_time = time.time() - start_time

    # Global R² on test set (macro-average over all outputs)
    r2_global = r2_score(Y_test, Y_pred, multioutput='uniform_average')

    # Plotting
    plot_parity_plots(Y_test, Y_pred, r2_scores_mean, property_names, plots_dir)

    # Save results
    results.append({
        "n_estimators": n_estimators,
        "R2_global": r2_global,
        "Training_time": training_time,
        "R2_means": r2_scores_mean,
        "R2_stds": r2_scores_std
    })

# ================= Save summary =================i
model_dir = os.path.join("Results", "RF", info)  # folder for model results
os.makedirs(model_dir, exist_ok=True)           

# File di riepilogo
summary_file = os.path.join(model_dir, "results_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Random Forest Results Summary\n")
    f.write("============================\n\n")
    for res in results:
        f.write(f"n_estimators = {res['n_estimators']}\n")
        f.write(f"  R2_global      = {res['R2_global']:.4f}\n")
        f.write(f"  Training time  = {res['Training_time']:.2f} s\n")
        f.write("----------------------------\n")

print(f"\n[INFO] Summary saved to {summary_file}")

# ================= Global Plot (R2 vs time) =================
n_estimators = [res["n_estimators"] for res in results]
R2_global = [res["R2_global"] for res in results]
Training_time = [res["Training_time"] for res in results]

plots_dir = os.path.join(plots_dir, info)
plot_rf_r2_vs_time(n_estimators, R2_global, Training_time, plots_dir)
