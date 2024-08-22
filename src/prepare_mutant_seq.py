from Bio import PDB

from residue_constants import restypes_with_x
import torch

# Dictionary to map 3-letter amino acid codes to 1-letter codes
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'XLE': 'J',
    'XAA': 'X', 'MSE': 'M'  # Including some common cases and unknown residues
}

def read_pdb_chain(pdb_file, chain_id):
    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)
    # Parse the structure
    structure = parser.get_structure('structure', pdb_file)
    
    # Initialize the dictionary
    residue_dict = {}
    
    # Loop through all residues in the specified chain
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    # Only include standard amino acids
                    if residue.get_resname() in three_to_one:
                        # Residue id is a tuple: (' ', resseq, icode), we're interested in resseq
                        residue_index = residue.get_id()[1]
                        residue_code = three_to_one[residue.get_resname()]
                        residue_dict[residue_index] = residue_code
    
    return residue_dict

# Example usage:
# pdb_dict = read_pdb_chain('example.pdb', 'A')
# print(pdb_dict)

def prepare_mutant_seq(wild_pdb_path, mutation_info, chain_id):
    wild_residue_dict = read_pdb_chain(pdb_file=wild_pdb_path,chain_id=chain_id)
    wild_residue, pos, mutant_residue = mutation_info.split('_')
    sorted_dict = dict(sorted(wild_residue_dict.items()))

    mut_seq = ''

    for key, value in sorted_dict.items():
        if key != pos:
            mut_seq += value
        else:
            mut_seq += mutant_residue
    return mut_seq

def prepare_wild_seq(wild_pdb_path, mutation_info, chain_id):
    wild_residue_dict = read_pdb_chain(pdb_file=wild_pdb_path,chain_id=chain_id)
    wild_residue, pos, mutant_residue = mutation_info.split('_')
    sorted_dict = dict(sorted(wild_residue_dict.items()))

    wild_seq = ''

    for key, value in sorted_dict.items():
        wild_seq += value
    return wild_seq





def seq_one_hot(wild_seq,mut_seq):
        lst = []
        for i,residue in enumerate(wild_seq):
            one_hot_vector_lst = [0] * len(restypes_with_x)
            if wild_seq[i] == mut_seq[i]:
                residue_idx = restypes_with_x.index(wild_seq[i])
                one_hot_vector_lst[residue_idx] = 1
            else:
                wild_residue_idx = restypes_with_x.index(wild_seq[i])
                mut_residue_idx = restypes_with_x.index(mut_seq[i])
                one_hot_vector_lst[wild_residue_idx] = -1
                one_hot_vector_lst[mut_residue_idx] = 1
            lst.append(one_hot_vector_lst)

        return torch.tensor(lst)

def get_aatype(seq):
        lst = []
        for res in seq:
            lst.append(int(restypes_with_x.index(res)))
        return torch.tensor(lst).long()


