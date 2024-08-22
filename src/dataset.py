import os
import torch
from torch.utils.data import Dataset
# from openfold.utils.rigid_utils import Rigid
from biopandas.pdb import PandasPdb
import pandas as pd
from protein import from_pdb_string
import pickle
import random
from residue_constants import restype_3to1, restypes, atom_types, restype_1to3, restypes_with_x
from collections import defaultdict
from Bio.PDB import PDBParser, PPBuilder
import math
import numpy as np
import subprocess
from Bio.PDB import PDBParser, Selection, NeighborSearch
from sklearn.metrics import mean_squared_error
from Bio import PDB
# from edge_feature_calculation import extract_features

# from openfold.data.data_transforms import atom37_to_frames, atom37_to_torsion_angles, make_atom14_positions,make_atom14_masks
# from openfold.utils.loss import (
#     torsion_angle_loss,
#     compute_fape,
#     between_residue_bond_loss,
#     between_residue_clash_loss,
#     find_structural_violations,
#     compute_renamed_ground_truth,
#     masked_msa_loss,
#     distogram_loss,
#     experimentally_resolved_loss,
#     violation_loss,
#     fape_loss,
#     lddt_loss,
#     supervised_chi_loss,
#     backbone_loss,
#     sidechain_loss,
#     tm_loss,
#     compute_plddt,
#     compute_tm,
#     chain_center_of_mass_loss
# )

from Bio.PDB import PDBParser, PDBIO

esm_feature_path = 'esm_feature_dict_15b_new.pkl'
esm_feature_path_2023 = 'esm_feature_dict_15b_new_2023.pkl'
class dataset(Dataset):
    def __init__(self,csv_file,cluster_dict,test=False,pdb_file_dir='All_PDB',esm_feature_path = esm_feature_path) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.cluster_dict = cluster_dict
        self.pdb_dir = pdb_file_dir
        self.esm_feature_path = esm_feature_path
        self.esm_feature_dict = self.read_pickle_file(self.esm_feature_path)
        # self.esm_featue_dict = self.read_pickle_file('')
        self.test = test
    
    def __len__(self):
        return len(self.df)
    def read_cluster_dict(self):
        with open(self.cluster_dict,'rb') as f:
            cluster = pickle.load(f)
        key_list = []

        for key,value in cluster.items():
            key_list.append(key)
        return cluster, key_list
    def get_random_integer(self,min_value, max_value):
    
        return random.randint(min_value, max_value)
    
    def read_file_as_string(self,file_path):
   
        with open(file_path, 'r') as file:
            return file.read()
    def pdb_to_tensor(self,pdb_file, chain_id):
        """
        Reads a PDB file and returns the coordinates of the atoms for a specified chain as a PyTorch tensor.

        Args:
            pdb_file (str): Path to the PDB file.
            chain_id (str): The ID of the chain to extract coordinates from.

        Returns:
            torch.Tensor: A tensor containing the coordinates of the atoms in the specified chain.
        """
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_file)

        # Extract coordinates from the specified chain
        coords = []
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())

        # Convert to a PyTorch tensor
        coords_tensor = torch.tensor(coords)

        return coords_tensor
    def read_pdb_and_extract_coordinates(self,pdb_file, chain_id):
        """
        Reads a PDB file and extracts the coordinates for atoms N, CA, and C for a specified chain as PyTorch tensors.

        Parameters:
        - pdb_file: Path to the PDB file.
        - chain_id: The chain identifier to filter atoms by.

        Returns:
        - n_coords: Tensor of coordinates for atom N.
        - ca_coords: Tensor of coordinates for atom CA.
        - c_coords: Tensor of coordinates for atom C.
        """
        n_coords = []
        ca_coords = []
        c_coords = []

        # pdb_file_str = self.read_file_as_string(file_path=pdb_file)
        # pdb_file_obj = from_pdb_string(pdb_file_str,chain_id=chain_id)

        # atom_positions = pdb_file_obj.atom_positions

        # pdb_tensor = self.pdb_to_tensor(pdb_file=pdb_file,chain_id=chain_id)
        # pdb_tensor_mean = self.subtract_mean_dim(torch.tensor(atom_positions),dim=(-3,-2))

        with open(pdb_file, 'r') as file:
            for line in file:
                if line.startswith('ATOM') and line[21] == chain_id:
                    atom_type = line[12:16].strip()
                    alt_loc = line[16].strip()
                    if alt_loc not in ('', 'A'):  # Skip alternate locations other than 'A' or blank
                        continue
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    if atom_type == 'N':
                        n_coords.append([x, y, z])
                    elif atom_type == 'CA':
                        ca_coords.append([x, y, z])
                    elif atom_type == 'C':
                        c_coords.append([x, y, z])

        return torch.tensor(n_coords), torch.tensor(ca_coords), torch.tensor(c_coords)
    

    def get_sequence(self,pdb_file,chain_id):
        pdb = PandasPdb().read_pdb(path=pdb_file)
        seq3_amino = pdb.df['ATOM'][
            (pdb.df['ATOM']['chain_id'] == chain_id) & (pdb.df['ATOM']['atom_name'] == 'CA') & (
                pdb.df['ATOM']['alt_loc'].isin(['A', '']))]['residue_name'].to_numpy()
        # print(len(seq3_amino))
        sequence = ''
        for amino_acid in seq3_amino:
            if amino_acid in restype_3to1:
                sequence += restype_3to1[amino_acid]
            else:
                sequence += 'X'
        return sequence
    
    def get_mut_seq(self,seq,mapping,wild_res,pos,mut_res):
        mut_seq = ''
        # for idx, res in enumerate(seq):
        #     if idx == int(mapping[int(pos)+1]) - 1 and res == wild_res:
        #         mut_seq += mut_res
        #     else:
        #         mut_seq += res
        for idx,res in enumerate(seq):
            if idx == mapping.index(int(pos)) and res == wild_res:
                mut_seq += mut_res
            else:
                mut_seq += res
        return mut_seq
    def get_residue_mapping(self,pdb_file):
        with open(pdb_file, 'r') as f:
            pdb_lines = f.readlines()
        new_residue_number = 0
        old_residue_new_residue_mapping = defaultdict(int)
        for i in range(len(pdb_lines)):
            if pdb_lines[i].startswith('ATOM') and 'CA' in pdb_lines[i]:
                # Extract the original residue number
                original_residue_number = int(pdb_lines[i][22:26].strip())
                # print(original_residue_number)
                if original_residue_number not in old_residue_new_residue_mapping:
                    new_residue_number += 1
                    old_residue_new_residue_mapping[original_residue_number] = new_residue_number
        return old_residue_new_residue_mapping
    def read_pickle_file(self,file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    def compute_distance_matrix(self,pdb_file, chain_id):
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
    def get_aatype(self,seq):
        lst = []
        for res in seq:
            lst.append(int(restypes_with_x.index(res)))
        return torch.tensor(lst).long()
    
    def calculate_torsion_angles(self,pdb_file, chain_id):
        """
        Calculates the phi and psi torsion angles for each amino acid residue in a protein structure for a specific chain,
        and encodes them as sine and cosine values.

        Parameters:
        - pdb_file: Path to the PDB file.
        - chain_id: The chain identifier to filter atoms by.

        Returns:
        - torsion_angles: A PyTorch tensor of size (N, 7, 2) where N is the number of amino acids in the chain,
                        7 represents the seven torsion angles (phi, psi, omega, chi1, chi2, chi3, chi4),
                        and 2 represents the sine and cosine encoding of the angle.
        - torsion_angles_sym: A PyTorch tensor of the same size representing the 180-degree rotation symmetry.
        """
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]  # Assuming only one model in the PDB file

        torsion_angles = []

        for chain in model:
            if chain.id == chain_id:
                polypeptides = PPBuilder().build_peptides(chain)
                for poly in polypeptides:
                    phi_psi = poly.get_phi_psi_list()
                    for res_index, angles in enumerate(phi_psi):
                        # Initialize all angles to (None, None)
                        angles_list = [(None, None)] * 7
                        angles_list[0] = (angles[0], None) if angles[0] is not None else (None, None)
                        angles_list[1] = (angles[1], None) if angles[1] is not None else (None, None)
                        # Add additional torsion angles here if needed
                        torsion_angles.append(angles_list)

        # Encode angles as sine and cosine, and create the symmetry tensor
        torsion_angles_encoded = []
        torsion_angles_sym_encoded = []

        for residue in torsion_angles:
            encoded_residue = []
            encoded_residue_sym = []
            for angle_pair in residue:
                angle, _ = angle_pair
                if angle is not None:
                    sin_val = math.sin(angle)
                    cos_val = math.cos(angle)
                    encoded_residue.append([sin_val, cos_val])
                    # For 180-degree rotation symmetry, add pi to the angle
                    sin_val_sym = math.sin(angle + math.pi)
                    cos_val_sym = math.cos(angle + math.pi)
                    encoded_residue_sym.append([sin_val_sym, cos_val_sym])
                else:
                    encoded_residue.append([float('nan'), float('nan')])
                    encoded_residue_sym.append([float('nan'), float('nan')])
            torsion_angles_encoded.append(encoded_residue)
            torsion_angles_sym_encoded.append(encoded_residue_sym)

        # Convert to PyTorch tensors
        torsion_angles_tensor = torch.tensor(torsion_angles_encoded)
        torsion_angles_sym_tensor = torch.tensor(torsion_angles_sym_encoded)

        return torsion_angles_tensor, torsion_angles_sym_tensor
    def numeric_to_letter(self,variable):
        """
        Checks if a variable is numeric and converts it to a corresponding letter.
        A for 1, B for 2, and so on.

        Parameters:
        - variable: The variable to check and convert.

        Returns:
        - A letter corresponding to the numeric value, or None if the variable is not numeric.
        """
        # Check if the variable is an int, float, or a string that represents a number
        if isinstance(variable, (int, float)) or (isinstance(variable, str) and variable.isdigit()):
            numeric_value = int(variable)
            if 1 <= numeric_value <= 26:
                return chr(64 + numeric_value)  # Convert to uppercase letter
            else:
                return "Value out of range (1-26)"
        else:
            return False
    def remove_hetatm(self,path,temp_dir,chain):
        name = path.split('/')[-1]
        temp_path = os.path.join(temp_dir,name)
        if self.numeric_to_letter(chain) == False:
            subprocess.run('pdb_selchain -{2} {0} | pdb_delhetatm  | pdb_delinsertion | pdb_reres > {1}'.format(path,temp_path,chain),shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:

            subprocess.run('pdb_selchain -{2} {0} | pdb_rplchain -{2}:{3} | pdb_delhetatm  | pdb_delinsertion | pdb_reres > {1}'.format(path,temp_path,chain,self.numeric_to_letter(chain)),shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run('pdb_rplchain -{0}:{1} {2} > {2}'.format(chain,self.numeric_to_letter(chain),temp_path),shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return temp_path
    
    
    def seq_one_hot(self,wild_seq,mut_seq):
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


        

    def esm_feature(self,sequence):
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        return token_representations[:,1:batch_lens-1]
    def get_ri(self,seq):
        lst = []
        for i in range(len(seq)):
            lst.append(int(i+1))
        return torch.tensor(lst).long()

    def subtract_mean_dim(self,tensor,dim=-2):
        """
        Subtracts the mean of a tensor along the specified dimension from the tensor itself.

        Args:
            tensor (torch.Tensor): A tensor of size (1 x N x 3).

        Returns:
            torch.Tensor: The tensor after subtracting the mean along the specified dimension.
        """
        # Calculate the mean along the specified dimension
        # mean = tensor.mean(dim=dim, keepdim=True)

        mean = torch.mean(input=tensor,dim=dim,keepdim=False)

        # Subtract the mean from the tensor
        result = tensor - mean

        return mean

    def center_pdb(self,input_pdb, output_pdb):
        parser = PDBParser()
        structure = parser.get_structure('input_structure', input_pdb)

        # Calculate the mean coordinates
        atom_coords = [atom.get_coord() for atom in structure.get_atoms()]
        mean_coords = np.mean(atom_coords, axis=0)

        # Subtract the mean coordinates from each atom's coordinates
        for atom in structure.get_atoms():
            atom.set_coord(atom.get_coord() - mean_coords)

        # Write the centered structure to a new PDB file
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb)
    
    def find_close_residues(self,pdb_file, chain_id, residue_pos, threshold=10.0):
        """Find residues within a certain distance of a specified residue in a PDB file.
        
        Args:
            pdb_file (str): Path to the PDB file.
            chain_id (str): Chain identifier.
            residue_pos (int): Position of the target residue.
            threshold (float): Distance threshold in angstroms.
            
        Returns:
            torch.Tensor: Tensor of size N where 1 indicates residues within the threshold distance, 0 otherwise.
        """
        # Parse the PDB file and get the specified chain
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        chain = None
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    break
            if chain is not None:
                break
        if chain is None or chain.id != chain_id:
            raise ValueError(f"Chain {chain_id} not found in the PDB file.")

        # Get coordinates of the target residue
        target_residue = None
        for residue in chain:
            if residue.id[1] == residue_pos:
                target_residue = residue
                break
        if target_residue is None:
            raise ValueError(f"Residue {residue_pos} not found in chain {chain_id}.")
        
        target_coords = np.array([atom.coord for atom in target_residue]).mean(axis=0)
        
        # Calculate distances and create the output tensor
        distances = []
        for residue in chain:
            coords = np.array([atom.coord for atom in residue]).mean(axis=0)
            distance = np.linalg.norm(coords - target_coords)
            distances.append(distance)
        
        distances = np.array(distances)
        within_distance = distances <= threshold
        tensor = torch.tensor(within_distance, dtype=torch.float32)
        
        return tensor
    def merge_pdb_files(self,pdb_a_path, pdb_b_path, output_path, chain_id):
        # Initialize dictionaries to store PDB data
        pdb_a_data = {}
        pdb_b_data = {}

        # Parse PDB A
        with open(pdb_a_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM") and line[21].strip() == chain_id:
                    key = (line[21].strip(), int(line[22:26].strip()), line[12:16].strip())
                    pdb_a_data[key] = {
                        'line': line[:54],  # Everything up to the occupancy and temperature factor
                    }

        # Parse PDB B
        with open(pdb_b_path, 'r') as file:
            for line in file:
                if line.startswith("ATOM") and line[21].strip() == chain_id:
                    key = (line[21].strip(), int(line[22:26].strip()), line[12:16].strip())
                    if key in pdb_a_data:  # Only store B data if it matches an A key
                        pdb_b_data[key] = {
                            'occupancy': line[54:60],
                            'temp_factor': line[60:66]
                        }

        # Write the new PDB file
        with open(output_path, 'w') as output_file:
            for key, a_entry in pdb_a_data.items():
                if key in pdb_b_data:
                    b_entry = pdb_b_data[key]
                    new_line = f"{a_entry['line']}{b_entry['occupancy']}{b_entry['temp_factor']}\n"
                else:
                    # Default values if no match found
                    new_line = f"{a_entry['line']}  1.00 50.00\n"
                output_file.write(new_line)

        return output_path
    
    def get_residue_indexes(self,pdb_file, chain_id):
        # Create a PDB parser object
        parser = PDB.PDBParser(QUIET=True)
        
        # Parse the PDB file into a structure object
        structure = parser.get_structure("protein", pdb_file)
        
        # Initialize an empty list to store residue indexes
        residue_indexes = []
        
        # Iterate over all models, chains, and residues to find the specified chain and its residues
        for model in structure:
            for chain in model:
                if chain.id == chain_id:  # Check if it's the specified chain
                    for residue in chain:
                        if residue.id[0] == ' ':  # Check if the residue is a standard amino acid
                        # Add the residue index to the list
                            residue_indexes.append(residue.id[1])
                    
                    # Break the chain loop since we found and processed the specified chain
                    break

        return residue_indexes
    def rename_residue(self,pdb_file, chain_id, residue_index, new_residue_name, output_file):
        # Create a PDB parser object
        parser = PDB.PDBParser(QUIET=True)
        
        # Parse the PDB file into a structure object
        structure = parser.get_structure("protein", pdb_file)
        
        # Define a PDB io object for writing the modified structure
        io = PDB.PDBIO()

        # Iterate over all models and chains to find the specified chain
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    # Once we find the correct chain, iterate over its residues
                    for residue in chain:
                        if residue.id[1] == residue_index:  # Check if it's the specified residue
                            # Modify the residue name
                            residue.resname = new_residue_name
                            break  # Exit after modifying the residue
                    break  # Exit after processing the specified chain

        # Save the modified structure to a new PDB file
        io.set_structure(structure)
        io.save(output_file)
    def scwrl_pdb(self,pdb_file,chain_id,pos,mut_res):
        if not os.path.exists('renamed_pdb_dir'):
            os.mkdir('renamed_pdb_dir')
        renamed_path = os.path.join('renamed_pdb_dir',pdb_file.split('/')[-1])
        self.rename_residue(pdb_file,chain_id,pos,mut_res,renamed_path)
        if not os.path.exists('scwrl_pdbs'):
            os.mkdir('scwrl_pdbs')
        scwrl_path = os.path.join('scwrl_pdbs',pdb_file.split('/')[-1])

        subprocess.run('Scwrl4 -h -i {0} -o {1}'.format(renamed_path,scwrl_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return scwrl_path
    def do_tmalign(self,wild_path,mutant_path,wild_chain, mutant_chain):
        wild_name = wild_path.split('/')[-1]
        mutant_name = mutant_path.split('/')[-1]
        tmalign_path = 'Tmalign_temp_dir'
        if not os.path.exists(tmalign_path):
            os.mkdir(tmalign_path)
        # subprocess.run('pdb_selaltloc {0} > {0}'.format(wild_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # subprocess.run('pdb_selaltloc {0} > {0}'.format(mutant_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        subprocess.run('./TMalign {0} {1} -o {2}/TM.sup'.format(wild_path,mutant_path,tmalign_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run('pdb_keepcoord {0}/TM.sup_all_atm > {0}/TM.sup_all_atm_coord'.format(tmalign_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        subprocess.run('pdb_selchain -A {0}/TM.sup_all_atm_coord | pdb_rplchain -A:{2} > {0}/{1}'.format(tmalign_path,wild_name,wild_chain),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run('pdb_selchain -B {0}/TM.sup_all_atm_coord | pdb_rplchain -B:{2} > {0}/{1}'.format(tmalign_path,mutant_name,mutant_chain),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        new_wild_path = os.path.join(tmalign_path,wild_name)
        new_mutant_path = os.path.join(tmalign_path,mutant_name)
        new_new_wild_path = self.merge_pdb_files(pdb_a_path=new_wild_path,pdb_b_path=wild_path,output_path=new_wild_path,chain_id=wild_chain)
        new_new_mutant_path = self.merge_pdb_files(pdb_a_path=new_mutant_path,pdb_b_path=mutant_path,output_path=new_mutant_path,chain_id=mutant_chain)
        subprocess.run('mv {0}/TM.sup_all_atm_coord {0}/TM_sup_all_atm_coord.pdb'.format(tmalign_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

        return new_new_wild_path, new_new_mutant_path, tmalign_path

    
    def find_engineered_mutation(self,pdb_filename):
        # Open the PDB file in read mode
        with open(pdb_filename, 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Check if the line starts with 'SEQADV' and has 'ENGINEERED MUTATION' in it
                if line.startswith('SEQADV') and 'ENGINEERED MUTATION' in line:
                    return line  # Return the line if it meets the criteria

        return None  # Return None if no such line is found
    def read_pdb_and_convert_to_numpy(self,pdb_filename, atom_types, chain_id):
        """
        Reads a PDB file and converts the coordinates to a numpy array of shape (1, N, 37, 3) for a specific chain.
        
        Parameters:
            pdb_filename (str): The path to the PDB file.
            atom_types (list): A list of atom types to include in the order specified.
            chain_id (str): The specific chain ID to filter the atoms by.

        Returns:
            np.array: A numpy array of shape (1, N, 37, 3) with coordinates, filled with zeros where atoms are missing.
        """
        # Initialize a dictionary to hold residue data
        residue_dict = {}
        
        # Read the PDB file
        with open(pdb_filename, 'r') as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    current_chain_id = line[21].strip()  # Chain identifier is at column 22 (0-based indexing 21)
                    if current_chain_id == chain_id:
                        atom_type = line[12:16].strip()
                        res_seq = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        
                        if atom_type in atom_types:
                            if res_seq not in residue_dict:
                                residue_dict[res_seq] = {}
                            residue_dict[res_seq][atom_type] = np.array([x, y, z])

        # Sort the residues by their sequence number
        sorted_residues = sorted(residue_dict.keys())

        # Create the numpy array
        N = len(sorted_residues)
        coords_array = np.zeros((1, N, 37, 3), dtype=float)
        
        for i, res_seq in enumerate(sorted_residues):
            for j, atom_type in enumerate(atom_types):
                if atom_type in residue_dict[res_seq]:
                    coords_array[0, i, j, :] = residue_dict[res_seq][atom_type]
        
        return coords_array

    def __getitem__(self, index) :
        data = self.df.iloc[index]
       
        

        wild = data['Wild_Protein']
        mutation_info = data['Mutation']
        Variant = data['Variant']
        wild_chain = wild.split('_')[-1]
        mutant = Variant

        # print(wild,Variant)
        wild_path = os.path.join(self.pdb_dir,wild.split('_')[0] + '.pdb')
        if not os.path.exists('temp_pdb_dir'):
            os.mkdir('temp_pdb_dir')
        wild_path = self.remove_hetatm(path = wild_path, temp_dir= 'temp_pdb_dir',chain=wild_chain)
        if not os.path.exists('temp_pdb_dir_mean'):
            os.mkdir('temp_pdb_dir_mean')
        
        wild_path_mean = os.path.join('temp_pdb_dir_mean',wild.split('_')[0] + '.pdb')
        self.center_pdb(wild_path,wild_path_mean)
        if self.numeric_to_letter(variable=wild_chain) != False:
            wild_chain = self.numeric_to_letter(wild_chain)
        # print(wild_path)
        # N_wild, CA_wild, C_wild = self.read_pdb_and_extract_coordinates(pdb_file=wild_path,chain_id=wild_chain)
        N_wild, CA_wild, C_wild = self.read_pdb_and_extract_coordinates(pdb_file=wild_path_mean,chain_id=wild_chain)

        if len(N_wild) == len(CA_wild) and len(N_wild) == len(C_wild):
            missing_wild = False
        else:
            missing_wild = True
        
        seq = self.get_sequence(pdb_file=wild_path,chain_id=wild_chain)
        old_to_new_mapping = self.get_residue_mapping(pdb_file=wild_path)
        wild_residue_index = self.get_residue_indexes(pdb_file=wild_path,chain_id=wild_chain)
        
        # mut_seq = self.get_mut_seq(seq=seq,mapping=old_to_new_mapping,wild_res=wild_res,pos=pos,mut_res=mut_res)
        # mut_seq = self.get_mut_seq(seq=seq,mapping=wild_residue_index,wild_res=wild_res,pos=pos,mut_res=mut_res)
        # close_residue_list = self.find_close_residues(pdb_file=wild_path_mean,chain_id=wild_chain,residue_index=int(pos) + 1,distance_threshold=10.0)
        # close_residue_list = self.find_close_residues(pdb_file=wild_path_mean,chain_id=wild_chain,residue_index=wild_residue_index.index(int(pos) + 1),distance_threshold=10.0)


        ###ground_truth

        mutant_path = os.path.join(self.pdb_dir,mutant.split('_')[0].lower() + '.pdb')
        mutant_chain = mutant.split('_')[-1]
        mutant_residue_indexes = self.get_residue_indexes(pdb_file=mutant_path,chain_id=mutant_chain)
        if not os.path.exists('temp_pdb_dir'):
            os.mkdir('temp_pdb_dir')
        mutant_path = self.remove_hetatm(path=mutant_path,temp_dir='temp_pdb_dir',chain=mutant_chain)
        if not os.path.exists('temp_pdb_dir_mean'):
            os.mkdir('temp_pdb_dir_mean')
        mutant_path_mean = os.path.join('temp_pdb_dir_mean',mutant.split('_')[0].lower() + '.pdb')
        self.center_pdb(input_pdb=mutant_path,output_pdb=mutant_path_mean)
        # mutant_residue_indexes = self.get_residue_indexes(pdb_file=mutant_path,chain_id=mutant_chain)
        wild_res, pos, mut_res = mutation_info.split('_')
        pos = self.find_engineered_mutation(pdb_filename=mutant_path).split()[4]
        

        try:
            actual_pos = int(wild_residue_index[mutant_residue_indexes.index(int(pos))])
        except:
            print('Some indexing error in {0}_{1}_{2}'.format(wild,mutation_info,Variant))
            print(wild_residue_index, mutant_residue_indexes)
            
        try:
            mut_seq = self.get_mut_seq(seq=seq,mapping=wild_residue_index,wild_res=wild_res,pos=actual_pos,mut_res=mut_res)
        except:
            print('conflict in {0}_{1}_{2}'.format(wild,mutation_info,Variant))
            
        # new_to_old_mapping = reverse_dict(old_to_new_mapping)
        wild_res, pos, mut_res = mutation_info.split('_')
        pos = self.find_engineered_mutation(pdb_filename=mutant_path).split()[4]
        try:
            close_residue_list = self.find_close_residues(pdb_file=wild_path,chain_id=wild_chain,residue_pos=actual_pos, threshold=10.0)
        except:
            print(wild + '_' + mutation_info + '_' + Variant,pos,actual_pos)


        if self.numeric_to_letter(variable=mutant_chain) != False:
            mutant_chain = self.numeric_to_letter(mutant_chain)
        N_mutant, CA_mutant, C_mutant = self.read_pdb_and_extract_coordinates(pdb_file=mutant_path_mean,chain_id=mutant_chain)

        new_wild_path, new_mutant_path, tmalign_path = self.do_tmalign(wild_path=wild_path_mean,mutant_path=mutant_path_mean,wild_chain=wild_chain,mutant_chain=mutant_chain)
        

        if len(N_mutant) == len(CA_mutant) and len(N_mutant) == len(C_mutant):
            missing_mutant = False
        else:
            missing_mutant = True
        missing_atom = missing_wild or missing_mutant
        if wild.split('_')[0].lower() == Variant.split('_')[0].lower():
            missing_atom = True
        
        # print(wild, mutant)
        # print(wild_chain, mutant_chain)
        if self.test == False:
            superimposed_path = os.path.join(tmalign_path,'TM_sup_all_atm_coord.pdb')
            N_wild, CA_wild, C_wild = self.read_pdb_and_extract_coordinates(pdb_file=new_wild_path,chain_id=wild_chain)
            N_mutant, CA_mutant, C_mutant = self.read_pdb_and_extract_coordinates(pdb_file=new_mutant_path,chain_id=mutant_chain)
        
        # r = Rigid.from_3_points(N_wild,CA_wild,C_wild)
        # r_mutant = Rigid.from_3_points(N_mutant,CA_mutant,C_mutant)

        # print(mean_squared_error(N_mutant,N_wild,squared=False))
        # print(mean_squared_error(CA_mutant,CA_wild,squared=False))
        # print(mean_squared_error(C_mutant,C_wild,squared=False))

        
        a_gt, a_gt_alt = self.calculate_torsion_angles(pdb_file=mutant_path,chain_id=mutant_chain)
        try:
            s = torch.squeeze(self.esm_feature_dict[mut_seq]).cpu()
            s = s[1:-1,:].to(torch.float32)
        except KeyError:
            # mut_seq = self.get_sequence(pdb_file=mutant_path_mean,chain_id=mutant_chain)
            # # s = torch.squeeze(self.esm_feature(sequence=mut_seq))
            # s = torch.squeeze(self.esm_feature_dict[mut_seq]).cpu()
            # s = s[1:-1,:].to(torch.float32)
            print(wild+'_' + mutation_info + '_' + Variant)
            print(seq)
            print(wild_residue_index)
            print(mut_seq)
            print(mutant_residue_indexes)
            
        
        # print(s.size(),len(mut_seq),mut_seq)
        ri = self.get_ri(mut_seq)

        z = self.compute_distance_matrix(pdb_file=wild_path,chain_id=wild_chain)
        z = torch.unsqueeze(z,dim=-1)
        aatype = self.get_aatype(seq=mut_seq)

        # mutant_pdb_str = self.read_file_as_string(file_path=mutant_path_mean)
        if self.test == False:
            print(new_wild_path)
            input_pdb = self.scwrl_pdb(pdb_file=new_wild_path,chain_id=wild_chain,pos=int(actual_pos),mut_res=restype_1to3[mut_res])
            # edge_concatenated_features, edge_index_tensor, res_id_tensor = extract_features(input_pdb, wild_chain, 50)
            input_pdb_str = self.read_file_as_string(file_path=input_pdb)
            input_protein_obj = from_pdb_string(pdb_str=input_pdb_str,chain_id=wild_chain)
            mutant_pdb_str = self.read_file_as_string(file_path=new_mutant_path)
            # print(mutant_pdb_str)
            mutant_protein_obj = from_pdb_string(pdb_str=mutant_pdb_str,chain_id=mutant_chain)
            mutant_positions = mutant_protein_obj.atom_positions
        else:
            input_pdb = self.scwrl_pdb(pdb_file=wild_path_mean,chain_id=wild_chain,pos=int(actual_pos),mut_res=restype_1to3[mut_res])
            # edge_concatenated_features, edge_index_tensor, res_id_tensor = extract_features(input_pdb, wild_chain, 50)

            input_pdb_str = self.read_file_as_string(file_path=input_pdb)
            input_protein_obj = from_pdb_string(pdb_str=input_pdb_str,chain_id=wild_chain)
            mutant_pdb_str = self.read_file_as_string(file_path=mutant_path_mean)
            # print(mutant_pdb_str)
            mutant_protein_obj = from_pdb_string(pdb_str=mutant_pdb_str,chain_id=mutant_chain)
            mutant_positions = mutant_protein_obj.atom_positions
        # mutant_positions = self.read_pdb_and_convert_to_numpy(pdb_filename=superimposed_path,atom_types=atom_types,chain_id='B')
        # mutant_dict = {'all_atom_positions': torch.tensor(mutant_protein_obj.atom_positions) - self.subtract_mean_dim(torch.tensor(mutant_protein_obj.atom_positions),dim=(-2,-3)), 'all_atom_positions_without_mean': torch.tensor(mutant_protein_obj.atom_positions),'aatype': torch.tensor(mutant_protein_obj.aatype), 'all_atom_mask': torch.tensor(mutant_protein_obj.atom_mask)}
        mutant_dict = {'all_atom_positions': torch.tensor(mutant_protein_obj.atom_positions) , 'all_atom_positions_without_mean': torch.tensor(mutant_protein_obj.atom_positions),'aatype': torch.tensor(mutant_protein_obj.aatype), 'all_atom_mask': torch.tensor(mutant_protein_obj.atom_mask)}
        # mutant_dict = {'all_atom_positions': torch.tensor(mutant_positions) , 'all_atom_positions_without_mean': torch.tensor(mutant_positions),'aatype': torch.tensor(mutant_protein_obj.aatype), 'all_atom_mask': torch.tensor(mutant_protein_obj.atom_mask)}

        # mutant_dict = atom37_to_torsion_angles(protein=mutant_dict)
        # # mutant_dict = atom37_to_frames(protein=mutant_dict)
        # mutant_dict = make_atom14_masks(protein=mutant_dict)
        # mutant_dict = make_atom14_positions(protein=mutant_dict)

        # mutant_dict = get_torsion_angle()
        # print(mutant_path)




        
        # r_rot = r.get_rots().get_rot_mats()
        # r_tran = r.get_trans()

        # r_mutant_rot = r_mutant.get_rots().get_rot_mats()
        # r_mutant_tran = r_mutant.get_trans()

        one_hot = self.seq_one_hot(wild_seq=seq,mut_seq=mut_seq)
        # print('{0}'.format(wild+'_' + mutation_info + '_' + Variant))
        # print(len(seq),len(mut_seq))
        s_cat = torch.cat((s,one_hot),dim=-1)
        # print(wild,mutant)
        subprocess.run('cd {0}; rm *'.format(tmalign_path),shell=True,stdout=subprocess.DEVNULL)
        # print(s_cat.size(),r_rot.size(),r_mutant_rot.size())
        # z = edge_concatenated_features
        # edges = edge_index_tensor
        if self.test == False:
            # return (s_cat, z, ri, aatype, mutant_dict, old_to_new_mapping, wild, Variant,wild_chain, close_residue_list,wild+'_' + mutation_info + '_' + Variant, torch.tensor(input_protein_obj.atom_positions),edges)
            return (s_cat, z, ri, aatype, mutant_dict, old_to_new_mapping, wild, Variant,wild_chain, close_residue_list,wild+'_' + mutation_info + '_' + Variant, torch.tensor(input_protein_obj.atom_positions))
        
        else:
            # return (s_cat, z, ri, aatype, mutant_dict, old_to_new_mapping, wild, Variant,wild_chain, close_residue_list,wild+'_' + mutation_info + '_' + Variant, torch.tensor(input_protein_obj.atom_positions),edges)
            return (s_cat, z, ri, aatype, mutant_dict, old_to_new_mapping, wild, Variant,wild_chain, close_residue_list,wild+'_' + mutation_info + '_' + Variant, torch.tensor(input_protein_obj.atom_positions))
        
           

train_ds = dataset(csv_file='MutData2022_train_sampled.csv',cluster_dict='MutData2022_train_cluster_dict.pkl')
validation_ds = dataset(csv_file='MutData2022_validation_sampled.csv',cluster_dict='MutData2022_validation_cluster_dict.pkl')
test_ds = dataset(csv_file='MutData2022_test_sampled.csv',cluster_dict='MutData2022_test_cluster_dict.pkl',test=True)
test_ds_2023 = dataset(csv_file='MutData2023_test_sampled.csv',cluster_dict='MutData2023_cluster_dict',esm_feature_path=esm_feature_path_2023,test=True)

# from network import SM
# from tqdm import tqdm
# for i in tqdm(range(validation_ds.__len__())):
    
        
#     s, z, ri, aatype, r_rot,r_tran , r_mutant_rot, r_mutant_tran, mutant_dict,_,_,_,_,close_residues, name, input_structure = validation_ds.__getitem__(i)
#     if s.size()[0] != input_structure.size()[0] or mutant_dict['all_atom_positions'].size()[0] != input_structure.size()[0]:
#         print(name, s.size(), mutant_dict['all_atom_positions'].size(), input_structure.size())
        
#     else:
#         # print(name, s.size(), mutant_dict['all_atom_positions'].size(), input_structure.size())
#         pass
    

        
    
    
    


# for i in range(train_ds.__len__()):
#     s, z, r, aatype, a_gt, a_gt_alt, r_mutant, mutant_dict = train_ds.__getitem__(i)
#     # print(s.shape[:-1])
    
#     out = SM({"single": s, "pair": z,"rigid": r}, aatype)
#     # print(out['positions'][-1])
#     # print(out['positions'][0].size())
#     pred_positions = torch.reshape(out['positions'][0],shape=(1,-1,3))
#     # print(torch.unsqueeze(mutant_dict['all_atom_positions'],dim=0).size())
#     target_positions = torch.unsqueeze(mutant_dict['all_atom_positions'],dim=0)
#     # print(torch.unsqueeze(mutant_dict['atom14_gt_positions'],dim=0).size())
#     target_positions = torch.reshape(torch.unsqueeze(mutant_dict['atom14_gt_positions'],dim=0),shape=(1,-1,3))

#     # frame_mask_size = r.to_tensor_7().size()
#     # print(r_mutant.to_tensor_7().size())
#     frame_mask = torch.ones(size= (s.shape[:-1]))
#     position_mask = torch.ones(size= (target_positions.shape[:-1]))
#     # print(frame_mask.size(),position_mask.size())

#     target_angles = torch.unsqueeze(mutant_dict['torsion_angles_sin_cos'],dim=0)
#     target_angles_alt = torch.unsqueeze(mutant_dict['alt_torsion_angles_sin_cos'],dim=0)

    
    
    
#     fapeloss = compute_fape(pred_frames=r,target_frames=r_mutant,frames_mask=frame_mask, pred_positions=pred_positions,target_positions=target_positions,positions_mask=position_mask,length_scale=10)
#     torsion_loss = torsion_angle_loss(a=out['angles'][0],a_gt=target_angles,a_alt_gt=target_angles_alt)
#     print(fapeloss)
#     print(torsion_loss)
#     break
    
    
