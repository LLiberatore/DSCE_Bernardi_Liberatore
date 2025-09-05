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

def count_functional_groups(mol, fparams):
    
    fg_counts = defaultdict(int)
    seen_matches = set()

    for fid in range(fparams.GetNumFuncGroups()):
        fg = fparams.GetFuncGroup(fid)
        smarts = Chem.MolToSmarts(fg)
        smarts_mol = Chem.MolFromSmarts(smarts)

        matches = mol.GetSubstructMatches(smarts_mol)
        for match in matches:
            atom_set = frozenset(match) # convert match to immutable set
            key = (fid, atom_set)
            if key not in seen_matches:
                seen_matches.add(key)
                fg_counts[fid] += 1

    # Convert to ordered list (vector)
    return [fg_counts[i] for i in range(fparams.GetNumFuncGroups())]

def binary_fingerprint_from_smiles(smiles, fcat):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    fpgen = FragmentCatalog.FragFPGenerator()
    fp = fpgen.GetFPForMol(mol, fcat)
    n_bits = fp.GetNumBits()
    return [1 if fp.GetBit(i) else 0 for i in range(n_bits)]

# NN training functions

def extract_history(history_dict):
    mse_train = history_dict.get('loss', [])
    mse_val = history_dict.get('val_loss', [])
    return mse_train, mse_val


def save_plot(fig, filename_base, save_dir):
    """Save a matplotlib figure in both PNG and PDF format."""
    os.makedirs(save_dir, exist_ok=True) # to create directory if it doesn't exist
    fig.savefig(os.path.join(save_dir, filename_base + ".png"))
    fig.savefig(os.path.join(save_dir, filename_base + ".pdf"))