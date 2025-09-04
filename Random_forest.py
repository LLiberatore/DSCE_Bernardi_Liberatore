"""
Multi-Target Molecular Property Regression with Random Forest + Parity Plots

- Carica X_features.npy (n_samples, n_features) e Y_labels.npy (n_samples, n_targets)
- Train/test split
- RandomForestRegressor multi-output
- Metriche: MSE/MAE per target, R^2 pesato complessivo
- Parity plots 4x3 con linea 1:1 e R^2 per pannello
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

start = timer()

# ------------ Caricamento dati ------------
X = np.load('X_features.npy')      # shape: (n_samples, n_features)
Y = np.load('Y_labels.npy')        # shape: (n_samples,) o (n_samples, n_targets)

# Forza Y a 2D se è 1D
if Y.ndim == 1:
    Y = Y.reshape(-1, 1)

# ------------ Train / Test split ------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ------------ Modello ------------
model = RandomForestRegressor(
    n_estimators=500,     # aumenta per più accuratezza (più lento)
    random_state=42,
    n_jobs=-1,
    oob_score=False
)
model.fit(X_train, Y_train)

# Predizioni
Y_pred = model.predict(X_test)

# ------------ Metriche ------------
# MSE/MAE per target; R^2 pesato complessivo
mse_per_target = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
mae_per_target = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')
r2_weighted = r2_score(Y_test, Y_pred, multioutput='variance_weighted')

print("=== Metrics on Test Set ===")
for i, (mse_i, mae_i) in enumerate(zip(np.atleast_1d(mse_per_target), np.atleast_1d(mae_per_target))):
    print(f"Target {i:02d} -> MSE: {mse_i:.6f} | MAE: {mae_i:.6f}")
print(f"R^2 (variance-weighted overall): {r2_weighted:.4f}")

print(f"Elapsed: {timer() - start:.2f}s")

# ------------ Parity Plots 4x3 ------------
property_names = [
    'μ (D)',              # dipole moment
    'α (a₀³)',            # isotropic polarizability
    'ε_HOMO (Ha)',        # energy of HOMO
    'ε_LUMO (Ha)',        # energy of LUMO
    'ε_gap (Ha)',         # HOMO–LUMO gap
    '⟨r²⟩ (a₀²)',         # electronic spatial extent
    'zpve (Ha)',          # zero-point vibrational energy
    'U₀ (Ha)',            # internal energy at 0 K
    'U (Ha)',             # internal energy at 298.15 K
    'H (Ha)',             # enthalpy at 298.15 K
    'G (Ha)',             # free energy at 298.15 K
    'Cᵥ (cal/mol·K)'      # heat capacity at 298.15 K
]

n_targets = Y_pred.shape[1]
names = property_names[:n_targets] if n_targets <= len(property_names) else [f'y{i}' for i in range(n_targets)]

save_dir = "Plots"
os.makedirs(save_dir, exist_ok=True)

rows, cols = 4, 3
fig, axes = plt.subplots(rows, cols, figsize=(16, 13))
axes = axes.ravel()
fig.suptitle('Parity Plots for Test Molecules', fontsize=18)

for i in range(min(n_targets, rows*cols)):
    ax = axes[i]
    x = Y_test[:, i]
    y = Y_pred[:, i]

    # Limiti comuni e linea 1:1
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    ax.plot([mn, mx], [mn, mx], '--', linewidth=1.0)

    # Scatter
    ax.scatter(x, y, alpha=0.5, s=20, edgecolors='k', linewidth=0.3)

    # R^2 per pannello
    try:
        r2 = r2_score(x, y)
        ax.set_title(f"{names[i]} — $R^2={r2:.3f}$", fontsize=12)
    except Exception:
        ax.set_title(f"{names[i]}", fontsize=12)

    ax.set_xlabel('True', fontsize=10)
    ax.set_ylabel('Predicted', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True)

# Rimuove assi extra se meno di 12 target
for j in range(n_targets, rows*cols):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])

png_path = os.path.join(save_dir, "parity_plots.png")
pdf_path = os.path.join(save_dir, "parity_plots.pdf")
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path, format='pdf')
plt.show()

print("Saved parity plots to:", png_path, "and", pdf_path)
