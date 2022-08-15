import matplotlib 
matplotlib.use('Agg')

import sys
sys.path.append("..")
from load_dataset import build_tg_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch_geometric as tg
from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor, matmul
if __name__ == "__main__":

    class MyConv(MessagePassing):

        def __init__(self,**kwargs):
            kwargs.setdefault('aggr', 'add')
            super().__init__(**kwargs)

        def reset_parameters(self):
            reset(self.nn)
            self.eps.data.fill_(self.initial_eps)
        def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                    size: Size = None) -> Tensor:
            if isinstance(x, Tensor):
                x: OptPairTensor = (x, x)

            # propagate_type: (x: OptPairTensor)
            out = self.propagate(edge_index, x=x, size=size)
            return out


        def message(self, x_j: Tensor) -> Tensor:
            return x_j

        def message_and_aggregate(self, adj_t: SparseTensor,
                                  x: OptPairTensor) -> Tensor:
            adj_t = adj_t.set_value(None, layout=None)
            return matmul(adj_t, x[0], reduce=self.aggr)

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}(nn={self.nn})'

    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=True)


    conv=MyConv()

    flag = (data.y[data.edge_index[0,:]]<5)&(data.y[data.edge_index[1,:]]>=2)
    flag1 = (data.y[data.edge_index[0,:]]>=2)&(data.y[data.edge_index[1,:]]<5)
    #lag = (flag|flag1)
    edge_index = data.edge_index[:,:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,2]+=(data.y>1)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y1 = conv(y,edge_index[:,flag])
    y11 = conv(y,edge_index[:,flag][[1,0],:])
    y2 = conv(y1,edge_index[:,flag1])-y*y11.sum(dim=1).view(-1,1)

    ans=0
    for i in range(3):
        ans += ((y2[data.y==i,i]).sum()*1.0/(y2[data.y==i].sum()+1e-5))-(data.y==i).sum()*1.0/len(data.y)
        print(ans)
    print('XBX h',ans/2)

    flag = (data.y[data.edge_index[0,:]]<5)&(data.y[data.edge_index[1,:]]<=1)
    flag1 = (data.y[data.edge_index[0,:]]<=1)&(data.y[data.edge_index[1,:]]<5)
    #lag = (flag|flag1)
    edge_index = data.edge_index[:,:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,2]+=(data.y>1)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y1 = conv(y,edge_index[:,flag])
    y11 = conv(y,edge_index[:,flag][[1,0],:])
    y2 = conv(y1,edge_index[:,flag1])-y*y11.sum(dim=1).view(-1,1)

    ans=0
    for i in range(3):
        ans += ((y2[data.y==i,i]).sum()*1.0/(y2[data.y==i].sum()+1e-5))-(data.y==i).sum()*1.0/len(data.y)
        print(ans)
    print('XTX h',ans/2)

    edge_index = data.edge_index[:,:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,2]+=(data.y>1)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y1 = conv(y,edge_index)
    y2=y1

    ans=0
    for i in range(3):
        ans += ((y2[data.y==i,i]).sum()*1.0/(y2[data.y==i].sum()+1e-5))-(data.y==i).sum()*1.0/len(data.y)
        print(ans)
    print('XX h',ans/2)


    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2

    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=45)
    plt.yticks([0,0.2,0.4,0.6],fontsize=45)

    ax = plt.scatter(x=["$\\times$B$\\times$",'$\\times$T$\\times$','$\\times\\times$'], y=[0.2225,0.1460,0.1285], color='black',marker='+',edgecolors='black',s=500,linewidths=30)
    ax = plt.scatter(x=["$\\times$B$\\times$",'$\\times$T$\\timse$','$\\times\\times$'], y=[0.2225,0.1460,0.1285], color='black',marker='.',edgecolors='black',s=200,linewidths=None)

    ax = plt.scatter(x=['$\\times\\times$'], y=[0.011], color='r',marker='x',edgecolors='r',s=500,linewidths=30)
    ax = plt.scatter(x=['$\\times\\times$'], y=[0.011], color='r',marker='.',edgecolors='r',s=100,linewidths=1)

    ax = plt.scatter(x=['$\\times\\times$'], y=[0.416], color='r',marker='x',edgecolors='r',s=500,linewidths=30)
    ax = plt.scatter(x=['$\\times\\times$'], y=[0.416], color='r',marker='.',edgecolors='r',s=100,linewidths=1)

    plt.xlabel('Edge type',fontsize=50)
    plt.ylabel('$h$',fontsize=50)
    plt.ylim(0,0.5)
    plt.xlim(-0.5,2.5)
    plt.tight_layout()
    plt.savefig('./figure8.pdf',bbox_inches='tight', format='pdf')

#plt.show()