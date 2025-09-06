import matplotlib.pyplot as plt
from Utils import save_plot
from rdkit.Chem import Draw
from rdkit import Chem
import os

def plot_frequency(freq_values, label, saving_name, x_labels=None):
    save_dir = os.path.join("Plots", "Dataset_visualization")
    fig = plt.figure(figsize=(12, 5))

    threshold = 100  # fixed threshold
    x_vals = list(range(len(freq_values)))
    
    # Assign colors: red if below threshold, green if above or equal
    colors = ["green" if val >= threshold else "red" for val in freq_values]
    plt.bar(x_vals, freq_values, color=colors)
    
    # Plot threshold line
    plt.axhline(threshold, color='black', linestyle='--', label=f"Threshold = {threshold}")
    plt.legend(fontsize=16)

    if x_labels is not None:
        plt.xticks(x_vals, x_labels, fontsize=14, rotation=90)
    else:
        plt.xticks(fontsize=14)

    plt.xlabel(f"{label} ID", fontsize=20)
    plt.ylabel("Frequency (occurrences)", fontsize=20)
    plt.yscale("log")
    plt.yticks(fontsize=20)
    plt.tight_layout()
    save_plot(fig, saving_name, save_dir)
    plt.show()

def plot_functional_groups(fparams, fids, freq_func_groups=None, sort_by_freq=False, save_path="functional_groups.png"):
                                     
    fg_mols = []
    fg_labels = []

    # Build (fid, freq) list
    if freq_func_groups is not None:
        func_with_freq = [(fid, freq_func_groups[fid]) for fid in fids]
        if sort_by_freq:
            func_with_freq = sorted(func_with_freq, key=lambda x: x[1], reverse=True)
    else:
        func_with_freq = [(fid, None) for fid in fids]

    for fid, freq in func_with_freq:
        fg = fparams.GetFuncGroup(fid)
        label = fg.GetProp("_Name") if fg.HasProp("_Name") else f"FG-{fid}"
        smarts = Chem.MolToSmarts(fg)
        mol = Chem.MolFromSmarts(smarts)
        if mol:
            fg_mols.append(mol)
            if freq is not None:
                fg_labels.append(f"#{fid+1} {label} ({int(freq)})")
            else:
                fg_labels.append(f"#{fid+1} {label}")

    img = Draw.MolsToGridImage(
        fg_mols,
        molsPerRow=5,
        subImgSize=(250, 250),
        legends=fg_labels
    )
    img.save(save_path + ".png")
    img.save(save_path + ".pdf")
    print(f"Saved: {save_path}.png and {save_path}.pdf")

import matplotlib.pyplot as plt
import os

def plot_training_curves(MSE_training_history, MSE_val_history, plots_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(MSE_training_history, label='Training Loss')
    plt.plot(MSE_val_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation Loss')
    plt.legend()
    # Save plots
    loss_png = os.path.join(plots_dir, "training_loss.png")
    loss_pdf = os.path.join(plots_dir, "training_loss.pdf")
    plt.savefig(loss_png, dpi=300)
    plt.savefig(loss_pdf, format="pdf")
    plt.show()
    print(f"[INFO] Training curves saved in {plots_dir}")

def plot_parity_plots(Y_test, Y_pred, r2_scores, property_names, plots_dir):
    fig, axes = plt.subplots(4, 3, figsize=(16, 13))
    fig.suptitle('Parity Plots for Test Molecules', fontsize=18)
    for i, ax in enumerate(axes.ravel()):
        ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.4, edgecolor='k', linewidth=0.3, s=20)
        ax.plot([Y_test[:, i].min(), Y_test[:, i].max()],
                [Y_test[:, i].min(), Y_test[:, i].max()],
                'r--', linewidth=1.2)
        ax.set_title(property_names[i], fontsize=13)
        ax.text(0.05, 0.90, f"$R^2$ = {r2_scores[i]:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    parity_png = os.path.join(plots_dir, "parity_plots.png")
    parity_pdf = os.path.join(plots_dir, "parity_plots.pdf")
    plt.savefig(parity_png, dpi=300)
    plt.savefig(parity_pdf, format="pdf")
    plt.show()
    print(f"[INFO] Parity plots saved in {plots_dir}")