import subprocess
import os
import torch
import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO
from protein import from_pdb_string
from residue_constants import restype_1to3

def numeric_to_letter(variable):
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
        


def remove_hetatm(path,temp_dir,chain):
        name = path.split('/')[-1]
        temp_path = os.path.join(temp_dir,name)
        if numeric_to_letter(chain) == False:
            subprocess.run('pdb_selchain -{2} {0} | pdb_delhetatm  | pdb_delinsertion | pdb_reres > {1}'.format(path,temp_path,chain),shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:

            subprocess.run('pdb_selchain -{2} {0} | pdb_rplchain -{2}:{3} | pdb_delhetatm  | pdb_delinsertion | pdb_reres > {1}'.format(path,temp_path,chain,numeric_to_letter(chain)),shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # subprocess.run('pdb_rplchain -{0}:{1} {2} > {2}'.format(chain,self.numeric_to_letter(chain),temp_path),shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return temp_path

def center_pdb(input_pdb, output_pdb):
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


def rename_residue(pdb_file, chain_id, residue_index, new_residue_name, output_file):
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
def scwrl_pdb(pdb_file,chain_id,pos,mut_res):
    if not os.path.exists('renamed_pdb_dir'):
        os.mkdir('renamed_pdb_dir')
    renamed_path = os.path.join('renamed_pdb_dir',pdb_file.split('/')[-1])
    rename_residue(pdb_file,chain_id,pos,mut_res,renamed_path)
    if not os.path.exists('scwrl_pdbs'):
        os.mkdir('scwrl_pdbs')
    scwrl_path = os.path.join('scwrl_pdbs',pdb_file.split('/')[-1])

    subprocess.run('Scwrl4 -h -i {0} -o {1}'.format(renamed_path,scwrl_path),shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return scwrl_path
def read_file_as_string(file_path):
   
        with open(file_path, 'r') as file:
            return file.read()
def prepare_input_pdb(wild_path,actual_pos,mut_res,wild_chain):
    wild = wild_path.split('/')[-1]
    if not os.path.exists('temp_pdb_dir'):
        os.mkdir('temp_pdb_dir')
    wild_path = remove_hetatm(path = wild_path, temp_dir= 'temp_pdb_dir',chain=wild_chain)
    if not os.path.exists('temp_pdb_dir_mean'):
        os.mkdir('temp_pdb_dir_mean')
    
    wild_path_mean = os.path.join('temp_pdb_dir_mean',wild.split('_')[0] + '.pdb')
    center_pdb(wild_path,wild_path_mean)
    if numeric_to_letter(variable=wild_chain) != False:
        wild_chain = numeric_to_letter(wild_chain)

    
    input_pdb = scwrl_pdb(pdb_file=wild_path_mean,chain_id=wild_chain,pos=int(actual_pos),mut_res=restype_1to3[mut_res])
            # edge_concatenated_features, edge_index_tensor, res_id_tensor = extract_features(input_pdb, wild_chain, 50)

    input_pdb_str = read_file_as_string(file_path=input_pdb)
    input_protein_obj = from_pdb_string(pdb_str=input_pdb_str,chain_id=wild_chain)

    return torch.tensor(input_protein_obj.atom_positions)