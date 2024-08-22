from egnn_clean import EGNN
import torch
import torch.nn as nn
import torch.nn.functional as F

egnn = EGNN(in_node_nf=5120 + 21,hidden_nf=128, out_node_nf=3 * 37,in_edge_nf=1,n_layers=4,attention=True)


