from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils import build_fragment_catalog, atom_count_from_smiles, count_functional_groups,  binary_fingerprint_from_smiles

# Load dataset 
df = pd.read_pickle("qm9_preprocessed.pkl")
df = df.head(10000)  
smiles_list = df['smiles'].tolist()

# Build fragment catalog 
fcat, fparams = build_fragment_catalog(smiles_list)
num_fr = fcat.GetNumEntries()
print(f"\nNumber of fragments in catalog: {num_fr}")

# Input Matrix Construction
features = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    atom_counts = atom_count_from_smiles(smi)                  # Count atoms (C, H, O, N, F)
    func_group_counts = count_functional_groups(mol, fparams)  # Count functional groups
    fragment_fp = binary_fingerprint_from_smiles(smi, fcat)    # Generate binary fragment fingerprint

    input_vector = atom_counts + func_group_counts + fragment_fp # Concatenate all feature types into a single input vector
    features.append(input_vector)

X = np.array(features) # Convert feature matrix to numpy array

# Labels Matrix Construction
target_cols = [col for col in df.columns if col not in  ['smiles', 'tag', 'index']]
Y_labels = df[target_cols].to_numpy(dtype=np.float32) # Convert to numpy array

# Save features and labels
np.save("X_features.npy", X)
np.save("Y_labels.npy", Y_labels)


# Frequency analysis
num_atoms = 5  # C, H, O, N, F
num_func_groups = fparams.GetNumFuncGroups()
num_fragments = fcat.GetNumEntries()

# Slice functional groups and fragments
X_func_groups = X[:, num_atoms : num_atoms + num_func_groups]
X_fragments = X[:, num_atoms + num_func_groups:]

# Absolute frequency of appearance of each functional group and fragment
freq_func_groups = X_func_groups.sum(axis=0)
freq_fragments = X_fragments.sum(axis=0)

threshold = 100 # treshold for filtering

# Functional groups absolute frequency plot
plt.figure(figsize=(12,5))
plt.bar(range(num_func_groups), freq_func_groups)
plt.axhline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
plt.xlabel("Functional Group ID")
plt.ylabel("Frequency (occurrences)")
plt.title("Functional Group Frequencies in Dataset")
plt.legend()
plt.tight_layout()
plt.show()

# Fragments absolute frequency plot
plt.figure(figsize=(12,5))
plt.bar(range(num_fragments), freq_fragments)
plt.axhline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
plt.xlabel("Fragment ID")
plt.ylabel("Frequency (occurrences)")
plt.title("Fragment Frequencies in Dataset")
plt.legend()
plt.tight_layout()
plt.show()

# Filter functional groups index and fragments based on frequency threshold
keep_func_ids = [i for i, f in enumerate(freq_func_groups) if f > threshold] # enumerate() to get both index and element
keep_frag_ids = [i for i, f in enumerate(freq_fragments) if f > threshold]

# Ricostruisci nuova X
X_filtered = np.concatenate([
    X[:, :num_atoms],                   # Atoms
    X_func_groups[:, keep_func_ids],    # Functional groups filtered
    X_fragments[:, keep_frag_ids]       # Fragments filtered
], axis=1)                              # to concatenate along columns

print(f"Original X shape: {X.shape}")
print(f"Filtered X shape: {X_filtered.shape}")

# Save the filtered version
np.save("X_features_filtered.npy", X_filtered)