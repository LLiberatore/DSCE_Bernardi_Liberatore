import pandas as pd
from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
from collections import defaultdict
import os
import pandas as pd


def build_fragment_catalog(smiles_list, fg_filename='CustomFunctionalGroups.txt'):
    
    # Convert SMILES to RDKit Mol objects
    ms = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]

    # Create fragment catalog parameters and objects
    fg_file_path = os.path.join(os.path.dirname(__file__), fg_filename)
    fparams = FragmentCatalog.FragCatParams(1, 1, fg_file_path)
    fcat = FragmentCatalog.FragCatalog(fparams)
    fcgen = FragmentCatalog.FragCatGenerator()

    # Add fragments from molecules to catalog
    for m in ms:
        fcgen.AddFragsFromMol(m, fcat)

    return fcat, fparams

def atom_count_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    mol_with_H = Chem.AddHs(mol)
    atom_counts = defaultdict(int) # to initialize missing keys

    for atom in mol_with_H.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in {'C', 'H', 'O', 'N', 'F'}:
            atom_counts[symbol] += 1

    return [atom_counts['C'], atom_counts['H'], atom_counts['O'], atom_counts['N'], atom_counts['F']]

