from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# '''
# 此模型邻居矩阵采用sparse tensor的形式，可以大大减少计算量，
# 如果不使用sparse tensor形式传递，将adj_t替换成edge_index
# '''
# # class GCN(torch.nn.Module):
# #     def __init__(self
# #                  , in_channels
# #                  , hidden_channels
# #                  , out_channels
# #                  , num_layers
# #                  , dropout
# #                  , batchnorm=True):
# #         super(GCN, self).__init__()

# #         self.convs = torch.nn.ModuleList()
# #         self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
# #         self.batchnorm = batchnorm
# #         if self.batchnorm:
# #             self.bns = torch.nn.ModuleList()
# #             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
# #         for _ in range(num_layers - 2):
# #             self.convs.append(
# #                 GCNConv(hidden_channels, hidden_channels, cached=True))
# #             if self.batchnorm: 
# #                 self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
# #         self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

# #         self.dropout = dropout
# #         # self.linear = nn.Linear(hidden_channels, hidden_channels)

# #     def reset_parameters(self):
# #         for conv in self.convs:
# #             conv.reset_parameters()
# #         if self.batchnorm:
# #             for bn in self.bns:
# #                 bn.reset_parameters()

# #     def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
# #         for i, conv in enumerate(self.convs[:-1]):
# #             x = conv(x, edge_index)
# #             if self.batchnorm: 
# #                 x = self.bns[i](x)
# #             x = F.relu(x)
# #             x = F.dropout(x, p=self.dropout, training=self.training)
# #         x = self.convs[-1](x, edge_index)
# #         return x.log_softmax(dim=-1)


# import torch.nn as nn
# import torch
# import math
# import numpy as np
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter

# class GraphConvolution(nn.Module):

#     def __init__(self, in_features, out_features, residual=False, variant=False):
#         super(GraphConvolution, self).__init__() 
#         self.variant = variant
#         if self.variant:
#             self.in_features = 2*in_features 
#         else:
#             self.in_features = in_features

#         self.out_features = out_features
#         self.residual = residual
#         self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.out_features)
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj, h0, lamda, alpha, l):
#         theta = math.log(lamda/l+1)
#         hi = torch.spmm(adj, input)
#         if self.variant:
#             support = torch.cat([hi,h0],1)
#             r = (1-alpha)*hi+alpha*h0
#         else:
#             support = (1-alpha)*hi+alpha*h0
#             r = support
#         output = theta*torch.mm(support, self.weight)+(1-theta)*r
#         if self.residual:
#             output = output+input
#         return output

# class GCNII(nn.Module):
#     def __init__(self, nfeat, num_layers, hidden_channels, num_classes, dropout, lamda, alpha, variant):
#         super(GCNII, self).__init__()
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             self.convs.append(GraphConvolution(hidden_channels, hidden_channels,variant=variant))
#         self.fcs = nn.ModuleList()
#         self.fcs.append(nn.Linear(nfeat, hidden_channels))
#         self.fcs.append(nn.Linear(hidden_channels, num_classes))
#         self.params1 = list(self.convs.parameters())
#         self.params2 = list(self.fcs.parameters())
#         self.act_fn = nn.ReLU()
#         self.dropout = dropout
#         self.alpha = alpha
#         self.lamda = lamda

#     def forward(self, x, adj):
#         _layers = []
#         x = F.dropout(x, self.dropout, training=self.training)
#         layer_inner = self.act_fn(self.fcs[0](x))
#         _layers.append(layer_inner)
#         for i,con in enumerate(self.convs):
#             layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#             layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
#         layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
#         layer_inner = self.fcs[-1](layer_inner)
#         return F.log_softmax(layer_inner, dim=1)


# class GCNII_model(torch.nn.Module):
#     def __init__(self):
#         super(GCNII_model, self).__init__()
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(torch.nn.Linear(dataset.num_features, hidden_dim))
#         for _ in range(nlayer):
#             self.convs.append(GConv(hidden_dim, hidden_dim))
#         self.convs.append(torch.nn.Linear(hidden_dim,dataset.num_classes))
#         self.reg_params = list(self.convs[1:-1].parameters())
#         self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

#     def forward(self):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         _hidden = []
#         x = F.dropout(x, dropout ,training=self.training)
#         x = F.relu(self.convs[0](x))
#         _hidden.append(x)
#         for i,con in enumerate(self.convs[1:-1]):
#             x = F.dropout(x, dropout ,training=self.training)
#             beta = math.log(lamda/(i+1)+1)
#             x = F.relu(con(x, edge_index,alpha, _hidden[0],beta,edge_weight))
#         x = F.dropout(x, dropout ,training=self.training)
#         x = self.convs[-1](x)
#         return F.log_softmax(x, dim=1)