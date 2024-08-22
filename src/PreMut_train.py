import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

# from network import CombinedModel
from egnn_model import egnn
from dataset import train_ds, validation_ds, test_ds
import warnings

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Check if CUDA is available
if not torch.cuda.is_available():
    raise Exception("CUDA is not available. Check your installation.")

# List all available GPUs
print("Available GPUs:", torch.cuda.device_count())

# Set the GPU ID
device_id = 2  # This is the second GPU
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
    
    


def expand_tensor(tensor, new_dim_1=37, new_dim_2=3):
    """Expand a 1 x N tensor to a 1 x N x new_dim_1 x new_dim_2 tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of size 1 x N with values either 1 or 0.
        new_dim_1 (int): The first new dimension size (default is 37).
        new_dim_2 (int): The second new dimension size (default is 3).
        
    Returns:
        torch.Tensor: Expanded tensor of size 1 x N x new_dim_1 x new_dim_2.
    """
    # Ensure the input tensor is of the correct size and shape
    assert tensor.dim() == 2 and tensor.size(0) == 1, "Input tensor must be of size 1 x N"
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.int32, "Tensor values must be either 1 or 0"

    # Expand the tensor
    expanded_tensor = tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, new_dim_1, new_dim_2)
    
    return expanded_tensor
def calculate_losses(output,input_str,close_indices,target):
    number_of_nodes = input_str.size()[1]
    out = torch.reshape(output,shape=(1,number_of_nodes,37,3))
    # mask = generate_mask(input_str)
    # out = out * mask
    out = out + input_str

    rmse_loss = F.mse_loss(input=out.to(device),target=target.to(device),reduction='mean')

    return rmse_loss


def calculate_losses_close(output,input_str,close_indices,target,close_residues):
    number_of_nodes = input_str.size()[1]
    out = torch.reshape(output,shape=(1,number_of_nodes,37,3))
    # mask = generate_mask(input_str)
    # out = out * mask
    out = out + input_str
    out = out * expand_tensor(tensor=close_residues)

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
model = egnn.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
validation_loader = DataLoader(validation_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds,batch_size=1, shuffle=False)
best_val_loss = float('inf')

num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0
    for data in tqdm(train_loader):
        
        optimizer.zero_grad()
        # s, z, ri, aatype,  mutant_dict,_,_,_,_,close_residues,name,input_str,edges = data
        s, z, ri, aatype,  mutant_dict,_,_,_,_,close_residues,name,input_str = data

        if s.size()[1] != input_str.size()[1] or mutant_dict['all_atom_positions'].size()[1] != input_str.size()[1]:
            continue
        s = s.to(device)
        z = z.to(device)
        ri = ri.to(device)
        aatype = aatype.to(device)
        input_str = input_str.to(device)
        close_residues = close_residues.to(device)
        
        # edges = [torch.LongTensor(edges[0]).to(device), torch.LongTensor(edges[1]).to(device)]
        

        # r = Rigid(Rotation(rot_mats=r_rot), r_tran)
        # r_mutant = Rigid(Rotation(rot_mats=r_mutant_rot),r_mutant_tran)

        # r = r.cuda(device)
        # r_mutant = r_mutant.cuda(device)

        # out = SM({"single": s, "pair": z,"rigid": r}, aatype)
        # out = model(s, ri, r, aatype)
        ####This is only for the new protattention####
        # coordinates = torch.reshape(input=input_str,shape=(1,-1,37*3))
        # one_hot_seq = s[:,:,5120:] 
        # embeddings = s[:,:,:5120] 
        
        
        ######
        edges, edge_attr = get_edges_batch(n_nodes=z.size()[1],edge_attr=z,batch_size= 1)
        # out = model(coordinates, one_hot_seq, embeddings, edges,edge_attr)
        out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_str[:,:,1,:].float(),dim=0),edges,torch.reshape(extract_off_diagonal(torch.squeeze(edge_attr.float(),dim=0)),shape=(-1,1)))
        # out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_str[:,:,1,:].float(),dim=0),edges,torch.reshape(torch.squeeze(z.float(),dim=0),shape=(-1,49)))

        
        

        # pred_positions = torch.reshape(out['positions'][-1],shape=(1,-1,3))
        # frame_mask = torch.ones(size= (s.shape[:-1])).to(device)
        
        # target_positions = torch.reshape(mutant_dict['atom14_gt_positions'],shape=(1,-1,3)).to(device)
        # position_mask = torch.ones(size= (target_positions.shape[:-1])).to(device)
        # fapeloss = compute_fape(pred_frames=r,target_frames=r_mutant,frames_mask=frame_mask,pred_positions=pred_positions,target_positions=target_positions,positions_mask=position_mask,length_scale=10)
        # torsionangleloss = torsion_angle_loss(a=out['angles'][-1],a_gt=mutant_dict['torsion_angles_sin_cos'].to(device),a_alt_gt=mutant_dict['alt_torsion_angles_sin_cos'].to(device))

        # loss = fapeloss + torsionangleloss
        # loss = 0.5 * aux_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s) + 0.5 * main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s) 
        main_loss = calculate_losses(output=out,input_str=input_str,close_indices=close_residues,target=mutant_dict['all_atom_positions'])
        # close_loss = calculate_losses_close(output=out,input_str=input_str,close_indices=close_residues,target=mutant_dict['all_atom_positions'],close_residues=close_residues)
        # loss = 0.5 * main_loss + 0.5 * close_loss
        loss = main_loss.float()
        # try:
        #     aux = aux_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s,input_str=input_str)
        #     main = main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s, input_str=input_str)
        #     close_loss = close_main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s,close_indices=close_residues, input_str=input_str)
        #     loss = 0.33 * aux + 0.33 * main + 0.33 * close_loss
        # except:
        #     print(name)
        #     break
        # loss = 0.5 * aux + 0.5 * main
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # wandb.log({'Train Loss Step': loss.item()})

       






        

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data in tqdm(validation_loader):
            
            # s, z, ri, aatype,  mutant_dict,_,_,_,_,close_residues, _, input_str, edges = val_data
            s, z, ri, aatype,  mutant_dict,_,_,_,_,close_residues, _, input_str = val_data

            if s.size()[1] != input_str.size()[1] or mutant_dict['all_atom_positions'].size()[1] != input_str.size()[1]:
                continue
            s = s.to(device)
            z = z.to(device)
            ri = ri.to(device)
            close_residues = close_residues.to(device)
            aatype = aatype.to(device)
            input_str = input_str.to(device)

            # r = Rigid(Rotation(rot_mats=r_rot), r_tran)
            # r_mutant = Rigid(Rotation(rot_mats=r_mutant_rot),r_mutant_tran)

            # r = r.cuda(device)
            # r_mutant = r_mutant.cuda(device)
            # out = SM({"single": s, "pair": z,"rigid": r}, aatype)
            # out = model(s, ri, r, aatype)
            edges, edge_attr = get_edges_batch(n_nodes=z.size()[1],edge_attr=z,batch_size= 1)
            # edges = [torch.LongTensor(edges[0]).to(device), torch.LongTensor(edges[1]).to(device)]

            ####This is only for the new protattention####
            # coordinates = torch.reshape(input=input_str,shape=(1,-1,37*3))
            # one_hot_seq = s[:,:,5120:] 
            # embeddings = s[:,:,:5120] 
            
            
            ######
            # edges, edge_attr = get_edges_batch(n_nodes=z.size()[1],edge_attr=z,batch_size= 1)
            # out = model(coordinates, one_hot_seq, embeddings, edges,edge_attr)
            out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_str[:,:,1,:].float(),dim=0),edges,torch.reshape(extract_off_diagonal(torch.squeeze(edge_attr.float(),dim=0)),shape=(-1,1)))
            # out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_str[:,:,1,:].float(),dim=0),edges,torch.reshape(torch.squeeze(z.float(),dim=0),shape=(-1,49)))

            # pred_positions = torch.reshape(out['positions'][-1],shape=(1,-1,3))
            # frame_mask = torch.ones(size= (s.shape[:-1])).to(device)
            
            # target_positions = torch.reshape(mutant_dict['atom14_gt_positions'],shape=(1,-1,3)).to(device)
            # position_mask = torch.ones(size= (target_positions.shape[:-1])).to(device)
            # fapeloss = compute_fape(pred_frames=r,target_frames=r_mutant,frames_mask=frame_mask,pred_positions=pred_positions,target_positions=target_positions,positions_mask=position_mask,length_scale=10)
            # torsionangleloss = torsion_angle_loss(a=out['angles'][-1],a_gt=mutant_dict['torsion_angles_sin_cos'].to(device),a_alt_gt=mutant_dict['alt_torsion_angles_sin_cos'].to(device))
            main_loss = calculate_losses(output=out,input_str=input_str,close_indices=close_residues,target=mutant_dict['all_atom_positions'])
            # close_loss = calculate_losses_close(output=out,input_str=input_str,close_indices=close_residues,target=mutant_dict['all_atom_positions'],close_residues=close_residues)
            # loss = 0.5 * main_loss + 0.5 * close_loss
            loss = main_loss.float()


            # loss = fapeloss + torsionangleloss
            # loss = 0.5 * aux_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s) + 0.5 * main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s) 
            # aux = aux_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s, input_str=input_str)
            # main = main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s, input_str=input_str)
            # close_loss = close_main_fape_loss(out_dict=out,mutant_dict=mutant_dict,aatype=aatype,r_mutant=r_mutant,s=s,close_indices=close_residues, input_str=input_str)
            # loss = 0.33 * aux + 0.33 * main + 0.33 * close_loss
            # loss = 0.5 * aux + 0.5 * main



            val_loss += loss.item()
            # wandb.log({'Validation Loss Step': loss.item()})

    # Calculate average losses
    train_loss /= len(train_loader)
    val_loss /= len(validation_loader)



    # Log losses to Weights & Biases
    # wandb.log({'Train Loss': train_loss, 'Validation Loss': val_loss})
    print(f'Epoch {epoch+1}: Train Loss = {train_loss}, Validation Loss = {val_loss}')
        
    # Save model if validation loss has decreased
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss} --> {val_loss}). Saving model ...')
        torch.save(model.state_dict(), 'PreMut_weights.pth')
        best_val_loss = val_loss


# export PATH=/bml/sajid/scwrl4:$PATH
