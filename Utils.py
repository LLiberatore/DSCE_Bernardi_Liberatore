import pandas as pd
from rdkit import Chem
from rdkit.Chem import FragmentCatalog
from rdkit import RDConfig
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

