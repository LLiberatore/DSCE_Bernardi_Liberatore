from rdkit import Chem
from rdkit.Chem import FragmentCatalog, RDConfig, Draw
from collections import defaultdict
import os

# 1. Load RDKit functional groups definition file
fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')

# 2. Create fragment parameters (fragments between 1 and 6 atoms)
fparams = FragmentCatalog.FragCatParams(1, 1, fName)
print(f"Number of functional groups in the catalog: {fparams.GetNumFuncGroups()}")

# 3. Initialize fragment catalog and generator
fcat = FragmentCatalog.FragCatalog(fparams)
fcgen = FragmentCatalog.FragCatGenerator()

# 4. Define the molecule via SMILES
smiles = 'OCC(NC1CC1)CCC'
mol = Chem.MolFromSmiles(smiles)

# 5. Count atoms C, H, O, N, F in the whole molecule
# Add explicit hydrogens
mol_with_H = Chem.AddHs(mol)

# Count atoms C, H, O, N, F
atom_counts = defaultdict(int)
for atom in mol_with_H.GetAtoms():
    symbol = atom.GetSymbol()
    if symbol in {'C', 'H', 'O', 'N', 'F'}:
        atom_counts[symbol] += 1

# Print result
print("\nAtom counts in the full molecule (with explicit H):")
for symbol in ['C', 'H', 'O', 'N', 'F']:
    print(f"  {symbol}: {atom_counts[symbol]}")

# 6. Add fragments to the catalog
num_entries = fcgen.AddFragsFromMol(mol, fcat)
print(f"\nNumber of fragments identified: {num_entries}")

# 7. For each fragment, print description and functional group info
for i in range(fcat.GetNumEntries()):
    desc = fcat.GetEntryDescription(i)
    print(f"\nFragment {i}: {desc}")
    
    fg_ids = list(fcat.GetEntryFuncGroupIds(i))
    print(f"  Functional group IDs: {fg_ids}")
    
    for fid in fg_ids:
        name = fparams.GetFuncGroup(fid).GetProp('_Name')
        smarts = Chem.MolToSmarts(fparams.GetFuncGroup(fid))
        print(f"    ID {fid}: {name} (SMARTS: {smarts})")

# 8. Show the molecule
img = Draw.MolToImage(mol, size=(300, 300))
img.show()
