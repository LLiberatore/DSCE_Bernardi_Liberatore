import pandas as pd
from Utils import build_fragment_catalog

# Load dataset 
df = pd.read_pickle("qm9_preprocessed.pkl")
df = df.head(100)  # Limit to 100 molecules
smiles_list = df['smiles'].tolist()

# Build fragment catalog 
fcat, fparams = build_fragment_catalog(smiles_list)

print(f"\nNumber of fragments in catalog: {fcat.GetNumEntries()}")
