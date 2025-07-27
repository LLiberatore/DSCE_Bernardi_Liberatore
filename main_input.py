from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from Utils import build_fragment_catalog, atom_count_from_smiles, count_functional_groups,  binary_fingerprint_from_smiles

# Load dataset 
df = pd.read_pickle("qm9_preprocessed.pkl")
df = df.head(10000)  # Limit to 10000 molecules
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

