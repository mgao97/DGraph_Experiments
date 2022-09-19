import torch
import numpy as np
import torch_geometric as tg

datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'

def build_tg_data(is_undirected=True,datapath=None):
    origin_data = np.load(datapath)
    data = tg.data.Data()
    data.x = torch.tensor(origin_data['x']).float()
    data.y = torch.tensor(origin_data['y']).long()
    data.edge_index = torch.tensor(origin_data['edge_index']).long().T
    data.train_mask = torch.tensor(origin_data['train_mask']).long()
    data.val_mask = torch.tensor(origin_data['valid_mask']).long()
    data.test_mask = torch.tensor(origin_data['test_mask']).long()
    data.edge_time = torch.tensor(origin_data['edge_timestamp']).long()
    if(is_undirected):
        data.edge_index = tg.utils.to_undirected(data.edge_index)
    return data
