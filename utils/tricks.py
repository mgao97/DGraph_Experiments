import torch
import torch_geometric as tg
import copy
import numpy as np
class Structure:
    def __init__(self,trick):
        if trick not in ['original', 'knn', 'random']:
            raise ValueError('trick should be original, knn, or random')
        self.trick = trick
    def process(self,data):
        if self.trick == 'original':
            return data
        if self.trick == 'knn':
            return self._KNNGraph(data)
        if self.trick == 'random':
            return self._RandomGraph(data)
    def _KNNGraph(self,data,num=4300999):
        num=4300999
        node_num = data.x.shape[0]
        L = (torch.rand(num*50)*node_num).long().view(-1,1)
        R = (torch.rand(num*50)*node_num).long().view(-1,1)
        flag = (L.view(-1)!=R.view(-1))
        L = L[flag,:]
        R = R[flag,:]
        edge_index = torch.cat((L,R),dim=1).T
        x = data.x/data.x.norm(dim=1).view(-1,1)
        L = x[L.view(-1)]
        R = x[R.view(-1)]
        score = (L*R).sum(dim=1)
        score_a= score[score>0.9]
        edge_index = edge_index[:,score>0.9]
        score = score_a.numpy()
        index = np.argsort(-score)
        index = torch.tensor(index).long()
        edge_index = edge_index[:,index[:4300999]]
        data.edge_index = edge_index
        return data
    def _RandomGraph(self,data,num=4300999):
        node_num = data.x.shape[0]
        L = (torch.rand(num*2)*node_num).long().view(-1,1)
        R = (torch.rand(num*2)*node_num).long().view(-1,1)
        flag = (L.view(-1)!=R.view(-1))
        L = L[flag,:]
        R = R[flag,:]
        L = L[:num,:]
        R = R[:num,:]
        edge_index = torch.cat((L,R),dim=1).T
        data.edge_index = edge_index
        return data

class Missvalues:
    def __init__(self,trick):
        if trick not in ['null', 'default', 'trickA', 'trickB', 'trickC']:
            raise ValueError('trick should be null, default, trickA, trickB or trickC')
        self.trick = trick
        
    def process(self, data):
        if self.trick == 'null':
            return self._null(data)
        elif self.trick == 'default':
            return self._default(data)
        elif self.trick == 'trickA':
            return self._trickA(data)
        elif self.trick == 'trickB':
            return self._trickB(data)
        elif self.trick == 'trickC':
            return self._trickC(data)
    def _null(self, data):
        return data
    
    def _default(self, data):
        x = torch.cat([data.x,data.x],dim=1)
        data.x = x
        return data

    def _trickA(self, data):
        x = torch.cat([data.x,(data.x==-1).long()],dim=1)
        data.x = x
        return data

    def _trickB(self, data):
        x = data.x
        x[x==-1]+=1
        x = torch.cat([x,(data.x==-1).long()],dim=1)
        data.x = x
        return data

    def _trickC(self, data):
        x = data.x
        x[x==-1]+=1
        x = torch.cat([x,(data.x==-1).long()],dim=1)
        data.x = x
        return data

class Background:
    def __init__(self, trick,):
        if trick not in ['null', 'remove', 'flag', 'hetro']:
            raise ValueError('trick should be null, remove, flag, hetro')
        self.trick = trick
    def process(self, data, ratio = 0.5):
        
        if self.trick == 'null':
            return self._null(data)
        
        elif self.trick == 'remove':
            return self._remove(data, ratio)
        
        elif self.trick == 'flag':
            return self._flag(data)
        
        elif self.trick == 'hetro':
            return self._trans2hetro(data)
        
    def _null(self, data):
        return data
 
    def _remove(self, data,ratio=0.5):
        def build_new_mask(tmask,pure_idx):
            mask = torch.zeros(3700550)
            mask[tmask.long()]=1
            mask = mask[pure_idx]
            ids = torch.arange(len(mask))
            ids = ids[mask.bool()]
            return ids
        pure_idx = torch.arange(data.x.shape[0])[data.y<=1]
        pure_idx2 = torch.arange(data.x.shape[0])[data.y>1]
        randn = torch.rand(len(pure_idx2))
        pure_idx2 = pure_idx2[randn<=ratio]
        pure_idx = torch.cat([pure_idx,pure_idx2],dim=0)
        print(pure_idx)
        print(data.edge_index)
        pure_edge_index = tg.utils.subgraph(pure_idx,data.edge_index,relabel_nodes=True)[0]
        pure_data = copy.deepcopy(data)
        pure_data.x = data.x[pure_idx]
        pure_data.edge_index = pure_edge_index
        pure_data.y = data.y[pure_idx]
        pure_data.train_mask = build_new_mask(data.train_mask,pure_idx)
        pure_data.valid_mask = build_new_mask(data.valid_mask,pure_idx)
        pure_data.test_mask = build_new_mask(data.test_mask,pure_idx)
        print(pure_data.train_mask.max())
        return pure_data

    def _flag(self, data):
        flag=(data.y<=1).float().view(-1,1)
        flag1 = (data.y>1).float().view(-1,1)
        data.x = torch.cat([flag,flag1,data.x],dim=1)
        return data
    
    def _trans2hetro(self, data):
        l = data.y[data.edge_index[0,:]]
        r = data.y[data.edge_index[1,:]]
        edge_type = torch.zeros(data.edge_index.shape[1])
        edge_type[(l<=1) & (r<=1)]=0
        edge_type[(l<=1) & (r>1)]=1
        edge_type[(l>1) & (r<=1)]=2
        edge_type[(l>1) & (r>1)]=3
        data.edge_type=edge_type.long()
        data.edge_type=edge_type.long()
        return data