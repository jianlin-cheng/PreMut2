import parser
import argparse
from prepare_input_pdb import prepare_input_pdb
from prepare_mutant_seq import prepare_mutant_seq, prepare_wild_seq, seq_one_hot, get_aatype
from esm_embedding_generate import generate_embedding
from esm_embedding_generate import model as esm_model
from egnn_model import egnn
import warnings
import torch
from compute_edge_feat import compute_distance_matrix
import os
from residue_constants import restypes_with_x, atom_types, restype_1to3
parser = argparse.ArgumentParser(description='Prediction using PreMut')

parser.add_argument('wild_pdb_path',help='path to the wild pdb file')
parser.add_argument('mutation_info',help='the wild residue, the residue index (0 indexed) and the mutant residue. e.g: C_144_A')
parser.add_argument('chain_id',help='The chain id of the pdb file')
parser.add_argument('output_dir',help='Directory to save the output')
parser.add_argument('name',help='Name of the predicted file')
args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
wild_residue, pos, mutation_residue = args.mutation_info.split('_')
chain_id = args.chain_id

input_obj = prepare_input_pdb(wild_path=args.wild_pdb_path,actual_pos=int(pos),mut_res=mutation_residue,wild_chain=chain_id)
input_obj = torch.unsqueeze(input_obj,dim=0)
# print(input_obj)
mutant_sequence = prepare_mutant_seq(wild_pdb_path=args.wild_pdb_path,mutation_info=args.mutation_info,chain_id=args.chain_id)
wild_sequence = prepare_wild_seq(wild_pdb_path=args.wild_pdb_path,mutation_info=args.mutation_info,chain_id=args.chain_id)
one_hot_seq = seq_one_hot(wild_seq=wild_sequence,mut_seq=mutant_sequence)
esm_embeddings = generate_embedding(seq=mutant_sequence,model=esm_model)


# print(esm_embeddings)
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
# Set the GPU ID
device_id = 0  # This is the second GPU
if device_id >= torch.cuda.device_count():
    raise Exception("Invalid device_id, out of range.")
device = torch.device(f"cuda:{device_id}")
print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")

warnings.filterwarnings('ignore')
model = egnn.to(device)

load_model_state_dict(model=model,state_dict_path='PreMut_EGNN_ESM_15B_new_sampling_proper.pth')

z = compute_distance_matrix(pdb_file=args.wild_pdb_path,chain_id=args.chain_id)
z = torch.unsqueeze(z,dim=-1)
z = torch.unsqueeze(z,dim=0)
z = z.to(device)
one_hot_seq = torch.unsqueeze(one_hot_seq,dim=0)
one_hot_seq = one_hot_seq.to(device)
esm_embeddings = esm_embeddings.to(device)
input_obj = input_obj.to(device)
aatype = get_aatype(mutant_sequence)
print(esm_embeddings.size())
print(one_hot_seq.size())

print(z.size())
print(input_obj.size())

s = torch.cat((esm_embeddings,one_hot_seq),dim=-1)
print(s.size())

edges, edge_attr = get_edges_batch(n_nodes=z.size()[1],edge_attr=z,batch_size= 1)
with torch.no_grad():
    out, _ = model(torch.squeeze(s.float(),dim=0),torch.squeeze(input_obj[:,:,1,:].float(),dim=0),edges,torch.reshape(extract_off_diagonal(torch.squeeze(edge_attr.float(),dim=0)),shape=(-1,1)))


print(out.size())

mask = generate_mask(input_obj)
out = torch.reshape(out,shape=(1,-1,37,3))
if out.size()[1] != input_obj.size()[1]:
    out = out[:,:-1,:,:]
output = torch.reshape(out,shape=(1,-1,37,3)) + input_obj
output = output * mask
tensor_to_pdb(coords_tensor=output,res_tensor=aatype,filename=os.path.join(args.output_dir,args.name + '.pdb'))
