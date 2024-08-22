import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

# from network import CombinedModel
from egnn_model import egnn
from dataset import train_ds, validation_ds, test_ds, test_ds_2023, test_ds_2024
import warnings

import os

from residue_constants import restype_3to1, restypes, atom_types, restype_1to3, restypes_with_x

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Check if CUDA is available
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Check your installation.")

# List all available GPUs
print("Available GPUs:", torch.cuda.device_count())

# Set the GPU ID
device_id = 1  # This is the second GPU
if device_id >= torch.cuda.device_count():
    raise Exception("Invalid device_id, out of range.")

# Set the device
device = torch.device(f"cuda:{device_id}")
print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")

warnings.filterwarnings('ignore')


def create_tensor_from_nonzero(B):
    A = torch.zeros_like(B)
    A = torch.where(B != 0, torch.tensor(1).to(device), A)
    return A



def mask_tensor_by_indices(input_tensor, indices):
    """
    Creates a mask tensor based on specified indices.
    
    Args:
    input_tensor (torch.Tensor): The input tensor of shape (1, N, 3).
    indices (list): List of 1-indexed positions to be marked.
    
    Returns:
    torch.Tensor: A tensor of shape (1, N, 3) where positions in `indices` are 1, others are 0.
    """
    # Convert 1-indexed to 0-indexed
    zero_based_indices = [i - 1 for i in indices]
    
    # Initialize a tensor of zeros with the same shape as input_tensor
    mask_tensor = torch.zeros_like(input_tensor)
    
    # Set positions to 1 based on indices
    for index in zero_based_indices:
        if 0 <= index < mask_tensor.size(1):  # Check if the index is within bounds
            mask_tensor[0, index, :] = 1
    
    return mask_tensor
    
    



def calculate_losses(output,input_str,close_indices,target):
    number_of_nodes = input_str.size()[1]
    out = torch.reshape(output,shape=(1,number_of_nodes,37,3))

    out = out + input_str

    rmse_loss = F.mse_loss(input=out.to(device),target=target.to(device),reduction='mean')

    return rmse_loss




def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
            # else:
            #     rows.append(i)
            #     cols.append(j)

    edges = [rows, cols]
    return edges
def extract_off_diagonal(tensor):
    """
    Extracts a sub-tensor from a given square tensor by removing the diagonal elements.
    Handles tensors with a singleton third dimension.

    Args:
    tensor (torch.Tensor): A square tensor of shape (N, N, 1).

    Returns:
    torch.Tensor: A tensor of shape (N, N-1, 1) with diagonal elements removed.
    """
    if tensor.dim() == 3 and tensor.size(2) == 1:
        tensor = tensor.squeeze(2)  # Remove the singleton dimension
    else:
        raise ValueError("Tensor must be of shape (N, N, 1)")

    N = tensor.size(0)
    # Create a mask that is True for all non-diagonal elements
    mask = ~torch.eye(N, dtype=bool, device=tensor.device)
    # Use the mask to select elements and reshape the result
    extracted = tensor[mask].view(N, N-1)
    # Optionally add the singleton dimension back if needed
    return extracted.unsqueeze(2)
def get_edges_batch(n_nodes, edge_attr,batch_size=1):
    edges = get_edges(n_nodes)
    # edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]).to(device), torch.LongTensor(edges[1]).to(device)]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
def generate_mask(input_tensor):
    """
    Generates a mask tensor with 1s where the input tensor has non-zero values
    and 0s otherwise.

    Args:
    input_tensor (torch.Tensor): The input tensor of shape (N, 37, 3).

    Returns:
    torch.Tensor: A mask tensor of shape (N, 37, 3) with 1s and 0s.
    """
    # Ensure the tensor is not zero
    mask = input_tensor.ne(0)
    
    # Convert boolean mask to integer (1 and 0)
    mask = mask.int()
    
    return mask
def tensor_to_pdb(coords_tensor, res_tensor, filename):
    assert coords_tensor.shape[0] == 1 and coords_tensor.shape[3] == 3
    # assert res_tensor.shape[0] == 1 and res_tensor.shape[2] == 21

    coords_tensor = coords_tensor.squeeze(0).cpu().numpy()
    res_tensor = res_tensor.squeeze(0).cpu().numpy()

    with open(filename, 'w') as pdb_file:
        atom_index = 1
        for res_index in range(coords_tensor.shape[0]):
            # Get residue type from one-hot encoding
            # res_type_idx = res_tensor[res_index].argmax()
            res_type_idx = res_tensor[res_index]
            res_type = restypes_with_x[res_type_idx]
            
            for atom_type_idx in range(coords_tensor.shape[1]):
                atom_type = atom_types[atom_type_idx]
                x, y, z = coords_tensor[res_index, atom_type_idx]

                if not (x == 0 and y == 0 and z == 0):  # Skip zero coordinates
                    pdb_file.write(
                        f"ATOM  {atom_index:5d} {atom_type:>4} {restype_1to3[res_type]} A{res_index+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           \n"
                    )
                    atom_index += 1
def load_model_state_dict(model, state_dict_path):
    """
    Loads a state dictionary into a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to load the state dictionary into.
        state_dict_path (str): Path to the file containing the state dictionary.
    """
    # Load the state dictionary from the file
    state_dict = torch.load(state_dict_path)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
# wandb.init(project='PreMut_Structure_Module_EGNN')
model = egnn.to(device)

load_model_state_dict(model=model,state_dict_path='/bml/sajid/bmlfast_backup_sajid/sajid/PreMut_V2/PreMut_EGNN_ESM_15B_new_sampling_proper.pth')


optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
# validation_loader = DataLoader(validation_ds, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_ds,batch_size=1, shuffle=False)

test_loader = DataLoader(test_ds_2023,batch_size=1,shuffle=False)

# test_loader = DataLoader(test_ds_tanner,batch_size=1,shuffle=False)
best_val_loss = float('inf')

val_loss = 0
# save_dir = '/bml/sajid/bmlfast_backup_sajid/sajid/PreMut_V2/PreMut_EGNN_Predicts_MutData2023_more_edge_feat_proper'
# save_dir = '/bml/sajid/bmlfast_backup_sajid/sajid/PreMut_V2/PreMut_EGNN_Predicts_MutData2023_weighted_proper'
# save_dir = '/bml/sajid/bmlfast_backup_sajid/sajid/PreMut_V2/PreMut_EGNN_Predicts_Tanner'
save_dir = '/bml/sajid/bmlfast_backup_sajid/sajid/PreMut_V2/PreMut_EGNN_Predicts_MutData2022'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model.eval()
with torch.no_grad():
        for val_data in tqdm(test_loader):
            
            s, z, ri, aatype, mutant_dict,_,_,_,_,close_residues, name, input_str = val_data
            # s, z, ri, aatype,  mutant_dict,_,_,_,_,close_residues, name, input_str, edges = val_data

            # if s.size()[1] != input_str.size()[1] or mutant_dict['all_atom_positions'].size()[1] != input_str.size()[1]:
            #     continue
            s = s.to(device)
            z = z.to(device)
            ri = ri.to(device)
            # close_residues = close_residues.to(device)
            aatype = aatype.to(device)
            input_str = input_str.to(device)
            print(aatype.size())

            # r = Rigid(Rotation(rot_mats=r_rot), r_tran)
            # r_mutant = Rigid(Rotation(rot_mats=r_mutant_rot),r_mutant_tran)

            # r = r.cuda(device)
            # r_mutant = r_mutant.cuda(device)
            # out = SM({"single": s, "pair": z,"rigid": r}, aatype)
            # out = model(s, ri, r, aatype)
            edges, edge_attr = get_edges_batch(n_nodes=z.size()[1],edge_attr=z,batch_size= 1)
            # edges = [torch.LongTensor(edges[0]).to(device), torch.LongTensor(edges[1]).to(device)]

            if s.size()[1] != input_str.size()[1]:
                s = s[:,:-1,:]
                edges,edge_attr = get_edges_batch(n_nodes=s.size()[1],edge_attr=z[:,:-1,:-1,:],batch_size=1)
            print(s.size(),input_str.size(),edges[0].size(),edges[1].size(),edge_attr.size())
            
            out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_str[:,:,1,:].float(),dim=0),edges,torch.reshape(extract_off_diagonal(torch.squeeze(edge_attr.float(),dim=0)),shape=(-1,1)))
            # out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_str[:,:,1,:].float(),dim=0),edges,torch.reshape(torch.squeeze(z.float(),dim=0),shape=(-1,49)))

            # pred_positions = torch.reshape(out['positions'][-1],shape=(1,-1,3))
            # frame_mask = torch.ones(size= (s.shape[:-1])).to(device)
            
            # target_positions = torch.reshape(mutant_dict['atom14_gt_positions'],shape=(1,-1,3)).to(device)
            # position_mask = torch.ones(size= (target_positions.shape[:-1])).to(device)
            # fapeloss = compute_fape(pred_frames=r,target_frames=r_mutant,frames_mask=frame_mask,pred_positions=pred_positions,target_positions=target_positions,positions_mask=position_mask,length_scale=10)
            # torsionangleloss = torsion_angle_loss(a=out['angles'][-1],a_gt=mutant_dict['torsion_angles_sin_cos'].to(device),a_alt_gt=mutant_dict['alt_torsion_angles_sin_cos'].to(device))
            # loss = calculate_losses(output=out,input_str=input_str,close_indices=close_residues,target=mutant_dict['all_atom_positions'])
            mask = generate_mask(input_str)
            out = torch.reshape(out,shape=(1,-1,37,3))
            if out.size()[1] != input_str.size()[1]:
                out = out[:,:-1,:,:]
            output = torch.reshape(out,shape=(1,-1,37,3)) + input_str
            output = output * mask
            tensor_to_pdb(coords_tensor=output,res_tensor=aatype,filename=os.path.join(save_dir,name[0] + '.pdb'))

            # loss = fapeloss + torsionangleloss
            # loss = 0.5 * aux_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s) + 0.5 * main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s) 
            # aux = aux_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s, input_str=input_str)
            # main = main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s, input_str=input_str)
            # close_loss = close_main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s,close_indices=close_residues, input_str=input_str)
            # loss = 0.33 * aux + 0.33 * main + 0.33 * close_loss
            # loss = 0.5 * aux + 0.5 * main



            # val_loss += loss.item()

# val_loss /= len(test_loader)

# print(val_loss)
