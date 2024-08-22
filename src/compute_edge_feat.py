import torch
from Bio.PDB import PDBParser
import numpy as np
def compute_distance_matrix(pdb_file, chain_id):
        # Parse the PDB file
        parser = PDBParser()
        structure = parser.get_structure('pdb_structure', pdb_file)
        
        # Extract coordinates for the specified chain and CA atoms
        chain_coords = []
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.get_id()[0] == ' ' and residue.get_id()[2] == ' ':
                            try:
                                ca_atom = residue['CA']
                                chain_coords.append(ca_atom.get_coord())
                            except KeyError:
                                pass
        chain_coords = np.array(chain_coords)
    
        # Compute pairwise distances
        num_atoms = len(chain_coords)
        distance_matrix = torch.zeros(num_atoms, num_atoms)
        for i in range(num_atoms):
            for j in range(num_atoms):
                distance_matrix[i, j] = torch.norm(torch.tensor(chain_coords[i] - chain_coords[j]))
        
        return distance_matrix