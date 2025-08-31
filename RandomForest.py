# Random Forest Regression on Molecular Properties + Parity Plots

import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


start = timer()


# Dataset loading
X = np.load('X_features.npy')
Y_labels = np.load('Y_labels.npy')  

# Split train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, random_state=0)


# Random Forest
n_estimators = 100  # Number of tress in the forest
model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=-1)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# Metrics
# mse = mean_squared_error(Y_test, Y_pred)
# mae = mean_absolute_error(Y_test, Y_pred)
# r2 = r2_score(Y_test, Y_pred)

end = timer()

# print(f"MSE: {mse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"R2 score: {r2:.4f}")
print(f"Tempo di esecuzione: {end - start:.2f} secondi")


# Parity Plots
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


# Plots
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
fig.suptitle(f'Parity Plots for Test Molecules n_estimators = {n_estimators}', fontsize=18)

for i, ax in enumerate(axes.ravel()):
    ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.4,
               edgecolor='k', linewidth=0.3, s=20)
    ax.plot([Y_test[:, i].min(), Y_test[:, i].max()],
            [Y_test[:, i].min(), Y_test[:, i].max()],
            'r--', linewidth=1.2)
    ax.set_title(property_names[i], fontsize=13)
    ax.set_xlabel('True', fontsize=10)
    ax.set_ylabel('Predicted', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
