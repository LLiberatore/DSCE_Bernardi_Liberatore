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
