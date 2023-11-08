from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F

'''
此模型邻居矩阵采用sparse tensor的形式，可以大大减少计算量，
如果不使用sparse tensor形式传递，将adj_t替换成edge_index
'''
# class GCN(torch.nn.Module):
#     def __init__(self
#                  , in_channels
#                  , hidden_channels
#                  , out_channels
#                  , num_layers
#                  , dropout
#                  , batchnorm=True):
#         super(GCN, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
#         self.batchnorm = batchnorm
#         if self.batchnorm:
#             self.bns = torch.nn.ModuleList()
#             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(
#                 GCNConv(hidden_channels, hidden_channels, cached=True))
#             if self.batchnorm: 
#                 self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
#         self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         if self.batchnorm:
#             for bn in self.bns:
#                 bn.reset_parameters()

#     def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index)
#             if self.batchnorm: 
#                 x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x.log_softmax(dim=-1)

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

class ImprovedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, similarity_threshold=0.5):
        super(ImprovedGCNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.similarity_threshold = similarity_threshold
        self.weight = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.weight)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Calculate the similarity between node features
        similarity = torch.cosine_similarity(x_i, x_j, dim=1)

        # Filter out neighbors with low similarity
        mask = (similarity > self.similarity_threshold).view(-1, 1)
        x_j = x_j * mask

        return x_j

    def update(self, aggr_out):
        return self.weight(aggr_out)
    


class SMART(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, similarity_threshold=0.5, num_layers=2):
        super(SMART, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ImprovedGCNConv(input_dim, hidden_dim ,similarity_threshold))
        for _ in range(1, num_layers - 1):
            self.convs.append(ImprovedGCNConv(hidden_dim, hidden_dim, similarity_threshold))
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(ImprovedGCNConv(hidden_dim, output_dim, similarity_threshold))
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.convs[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
    

