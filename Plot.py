import matplotlib.pyplot as plt
from Utils import save_plot
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
