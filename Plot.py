import matplotlib.pyplot as plt

def plot_frequency(freq_values, label, threshold=None):
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(freq_values)), freq_values)
    if threshold is not None:
        plt.axhline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
        plt.legend()
    plt.xlabel(f"{label} ID")
    plt.ylabel("Frequency (occurrences)")
    plt.title(f"{label} Frequencies in Dataset")
    plt.tight_layout()
    plt.show()
