# dataset name: DGraphFin

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from utils.tricks import Missvalues
from utils.tricks import Background
from utils.tricks import Structure
# from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2,RGCN
from logger import Logger

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.sparse as sp
import random 
import time
import argparse
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric as tg
import torch_geometric.transforms as T

from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
import time
import math

eval_metric = 'auc'

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def eval_rocauc(y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''
        # print(y_true)
        # print(torch.unique(y_true))
        # print(y_pred)
        if y_pred.shape[1] == 2:
            auc = roc_auc_score(y_true, y_pred[:, 1])
            ap = average_precision_score(y_true, y_pred[:, 1])
        else:
            onehot_code = np.eye(y_pred.shape[1])
            y_true_onehot = onehot_code[y_true]
            auc = roc_auc_score(y_true_onehot, y_pred)
            ap = average_precision_score(y_true_onehot, y_pred)

        return auc, ap

from typing import Tuple, Optional, Union
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class GCNIIdenseConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 **kwargs):

        super(GCNIIdenseConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, alpha, h0, beta,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        
        
        support = (1-beta)*(1-alpha)*x + beta*torch.matmul(x, self.weight1)
        initial = (1-beta)*(alpha)*h0 + beta*torch.matmul(h0, self.weight2)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight,
                             size=None)+initial
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

GConv = GCNIIdenseConv

class GCNII_model(torch.nn.Module):
    def __init__(self):
        super(GCNII_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(dataset.num_features, hidden_dim))
        for _ in range(nlayer):
            self.convs.append(GConv(hidden_dim, hidden_dim))
        self.convs.append(torch.nn.Linear(hidden_dim,nlabels))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, dropout ,training=self.training)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, edge_index,alpha, _hidden[0],beta,edge_weight))
        x = F.dropout(x, dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     output = model(data.x,adj)
#     acc_train = accuracy(output[train_idx], data.y[train_idx])
#     loss_train = F.nll_loss(output[train_idx], data.y[train_idx])
#     loss_train.backward()
#     optimizer.step()
#     return loss_train.item(),acc_train.item()


# def validate():
#     model.eval()
#     with torch.no_grad():
#         output = model(data.x,adj)
#         loss_val = F.nll_loss(output[val_idx], data.y[val_idx])
#         # acc_val = accuracy(output[val_idx], labels[val_idx].to(device))

#         auc_val, ap_val = eval_rocauc(output[val_idx], data.y[val_idx]) 

#         return loss_val.item(), auc_val.item(), ap_val.item()

# @torch.no_grad()
# def test():
#     model.load_state_dict(torch.load('gcnii.pth'))
#     model.eval()
#     with torch.no_grad():
#         output = model(data.x, adj)
#         loss_test = F.nll_loss(output[test_idx], data.y[test_idx])

#         auc_test, ap_test = eval_rocauc(output[test_idx], data.y[test_idx]) 

#         # acc_test = accuracy(output[test_idx], labels[test_idx].to(device))
#         return loss_test.item(), auc_test.item(), ap_test.item()
        
def train():
    model.train()
    optimizer.zero_grad()
    loss_train = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


@torch.no_grad()
def test():
    model.eval()
    logits = model()
    loss_val = F.nll_loss(logits[data.valid_mask], data.y[data.valid_mask]).item()
    for _, mask in data('test_mask'):
        pred = logits[mask]
        auc_test, ap_test = eval_rocauc(data.y[data.test_mask], pred) 
        # accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss_val, auc_test, ap_test


# def sys_normalized_adjacency(adj):
#    adj = sp.coo_matrix(adj)
#    adj = adj + sp.eye(adj.shape[0])
#    row_sum = np.array(adj.sum(1))
#    row_sum=(row_sum==0)*1+row_sum
#    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
#    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


parser = argparse.ArgumentParser(description='gnn_models')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='DGraphFin')
# parser.add_argument('--log_steps', type=int, default=10)
# # parser.add_argument('--model', type=str, default='mlp')
# parser.add_argument('--use_embeddings', action='store_true')
# # parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--MV_trick', type=str, default='null')
parser.add_argument('--BN_trick', type=str, default='null')
parser.add_argument('--BN_ratio', type=float, default=0.1)
parser.add_argument('--Structure', type=str, default='original')
# # parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
# # parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
# # parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
# # # Training settings
# # parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
# parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
# parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
# parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
# parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--patience', type=int, default=100, help='Patience')
# # parser.add_argument('--data', default='cora', help='dateset')
# # parser.add_argument('--dev', type=int, default=0, help='device id')
# parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
# parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
# parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
# parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
args = parser.parse_args()
print(args)

# no_conv = False
# if args.model in ['mlp']: no_conv = True        

###################hyperparameters
nlayer = args.layer
dropout = 0.6
alpha = 0.1
lamda = 0.5
hidden_dim = 64
weight_decay1 = 0.01
weight_decay2 = 5e-4
lr = 0.01
patience = 100
#####################

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
# device = torch.device('cpu')
dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if args.dataset in ['DGraphFin']: nlabels = 2
    
data = dataset[0]
data.edge_index = data.adj_t
data.adj_t = torch.cat([data.edge_index.coo()[0].view(1,-1),data.edge_index.coo()[1].view(1,-1)],dim=0)
data.edge_index = data.adj_t
structure = Structure(args.Structure)
data = structure.process(data)
# data.adj_t = data.edge_index
# data.adj_t = tg.utils.to_undirected(data.adj_t)
# data.edge_index = data.adj_t

if args.dataset in ['DGraphFin']:
    x = data.x
    x = (x-x.mean(0))/x.std(0)
    data.x = x
if data.y.dim()==2:
    data.y = data.y.squeeze(1)        


print(data)
print(data.train_mask.sum())
print(data.valid_mask.sum())
print(data.test_mask.sum())

missvalues = Missvalues(args.MV_trick)
data = missvalues.process(data)

#print(data.edge_index)

# data.edge_index = data.adj_t
BN = Background(args.BN_trick)
data = BN.process(data,args.BN_ratio)

split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}

fold = args.fold
if split_idx['train'].dim()>1 and split_idx['train'].shape[1] >1:
    kfolds = True
    print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
    split_idx['train'] = split_idx['train'][:, fold]
    split_idx['valid'] = split_idx['valid'][:, fold]
    split_idx['test'] = split_idx['test'][:, fold]
else:
    kfolds = False

split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}

data = data.to(device)
train_idx = split_idx['train'].to(device)
val_idx = split_idx['valid'].to(device)
test_idx = split_idx['test'].to(device)
    


# adj = nx.adjacency_matrix(nx.from_dict_of_lists(data))
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# adj = sys_normalized_adjacency(adj)
# adj = sparse_mx_to_torch_sparse_tensor(adj)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, data = GCNII_model().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=weight_decay1),
    dict(params=model.non_reg_params, weight_decay=weight_decay2)
], lr=lr)

print(f'Model initialized')
print(model)

result_dir = prepare_folder(args.dataset, model)
print('result_dir:', result_dir)

evaluator = Evaluator(eval_metric)
logger = Logger(args.runs, args)
weight = torch.tensor([1,50]).to(device).float()

for run in range(args.runs):
    import gc
    gc.collect()
    # print(sum(p.numel() for p in model.parameters()))

    # best_val_loss = 9999999
    # test_acc = 0
    # bad_counter = 0
    # best_epoch = 0

    # model.reset_parameters()
    # optimizer = torch.optim.Adam([
    #                     {'params':model.params1,'weight_decay':args.wd1},
    #                     {'params':model.params2,'weight_decay':args.wd2},
    #                     ],lr=args.lr)
    best_valid = 0
    min_valid_loss = 1e8
    best_out = None
    
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    auc = 0
    for epoch in range(1, args.epochs+1):
        
        loss_tra = train()
        loss_val, auc_test_tmp, ap_test_tmp = test()
        if loss_val < min_valid_loss:
            best_val_loss = loss_val
            test_auc = auc_test_tmp
            test_ap = ap_test_tmp
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter+=1
        if epoch%20 == 0: 
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test auc: {:.4f}, Test ap: {:.4f}'
            print(log.format(epoch, loss_tra, loss_val, test_auc, test_ap))
        if bad_counter == patience:
            break
    log = 'best Epoch: {:03d}, Val loss: {:.4f}, Test auc: {:.4f}, Test ap: {:.4f}'
    print(log.format(best_epoch, best_val_loss, test_auc, test_ap))


