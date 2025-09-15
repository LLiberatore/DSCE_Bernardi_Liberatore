import os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from Plots import plot_parity_plots   # usa la stessa funzione già definita in Plots.py
from Utils import save_experiment_info

# ================= Configuration =================
info = "FR"   # "atom", "FG", "FR"
model_choice = "ridge"   # "linear", "ridge", "lasso"

# ================= Path Management =================
base_dir = "saved_models"
model_dir = os.path.join(base_dir, "ML", info, model_choice)
os.makedirs(model_dir, exist_ok=True)

# ================= Data Loading =================
X = np.load("X_features_filtered.npy")
Y_labels = np.load("Y_labels.npy")

num_atoms = 5
num_func_groups = 20  # dopo filtraggio
# input selection
if info == "atom":
    X = X[:, :num_atoms]
elif info == "FG":
    X = X[:, :num_atoms + num_func_groups]
elif info == "FR":
    X = X
else:
    raise ValueError("info must be 'atom', 'FG', or 'FR'")

# ================= Split dataset =================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42)

# ================= Normalization =================
# Scale inputs
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale outputs
scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# ================= Model Selection =================
if model_choice == "linear":
    model = LinearRegression()
elif model_choice == "ridge":
    model = Ridge(alpha=5)
elif model_choice == "lasso":
    model = Lasso(alpha=0.0001, max_iter=5000)
else:
    raise ValueError("Invalid model_choice")

# ================= Training and Prediction =================
start = timer()
model.fit(X_train_scaled, Y_train_scaled)
Y_pred_scaled = model.predict(X_test_scaled)
training_time = timer() - start

# Inverse-transform predictions to original scale
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# ================= Metrics =================
mse = mean_squared_error(Y_test, Y_pred)
r2_scores = [r2_score(Y_test[:, i], Y_pred[:, i]) for i in range(Y_test.shape[1])]
print(f"[INFO] Model: {model_choice}")
print(f"MSE: {mse:.6f}")
print(f"Training time: {training_time:.2f} s")

# ================= Property Names =================
property_names = [
    'μ (D)', 'α (a₀³)', 'ε_HOMO (Ha)', 'ε_LUMO (Ha)',
    'ε_gap (Ha)', '⟨r²⟩ (a₀²)', 'zpve (Ha)', 'U₀ (Ha)',
    'U (Ha)', 'H (Ha)', 'G (Ha)', 'Cᵥ (cal/mol·K)'
]

# ================= Plots =================
plots_dir = os.path.join("Plots", "ML", info, model_choice)
os.makedirs(plots_dir, exist_ok=True)

# --- Parity plots ---
plot_parity_plots(Y_test, Y_pred, r2_scores, property_names, plots_dir)

# --- Coefficient plots (solo barplot, tutte le proprietà in 4x3) ---
coef = model.coef_  # shape (n_outputs, n_features)

fig, axes = plt.subplots(4, 3, figsize=(18, 14))
for i, ax in enumerate(axes.ravel()):
    if i < coef.shape[0]:
        ax.bar(range(coef.shape[1]), coef[i])
        ax.set_title(property_names[i], fontsize=12)
        ax.set_xlabel("Variable index", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(plots_dir, f"coefficients_grid_{model_choice}.png"), dpi=300)
plt.show()
plt.close(fig)

print(f"[INFO] Coefficient grid plot saved in {plots_dir}")

# ================= Save Experiment Info =================
save_experiment_info(
    plots_dir, info, model_choice, "N/A", 0, 0,
    X_train, X_test, property_names, r2_scores,
    Y_test, Y_pred, Y_test_scaled, Y_pred_scaled, model, training_time
)
