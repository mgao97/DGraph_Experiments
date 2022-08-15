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
    data = build_tg_data(is_undirected=False)


    conv=MyConv()

    flag = (data.y[data.edge_index[0,:]]<2)&(data.y[data.edge_index[1,:]]<2)
    #flag1 = (data.y[data.edge_index[0,:]]==0)&(data.y[data.edge_index[1,:]]<2)
    #lag = (flag|flag1)
    edge_index = data.edge_index[:,:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y[:,2]+=(data.y>1)
    y1 = conv(y,edge_index[:,:])
    #y2 = conv(y1,edge_index[:,flag1])
    #count_edge_homo(y.T.matmul(y1))
    in_ratio=(y1[:,2]/(y1.sum(dim=1)+1e-4))


    flag = (data.y[data.edge_index[0,:]]<2)&(data.y[data.edge_index[1,:]]<2)
    #flag1 = (data.y[data.edge_index[0,:]]==0)&(data.y[data.edge_index[1,:]]<2)
    #lag = (flag|flag1)
    edge_index = data.edge_index[:,:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y[:,2]+=(data.y>1)
    y1 = conv(y,edge_index[[1,0],:])
    #y2 = conv(y1,edge_index[:,flag1])
    #count_edge_homo(y.T.matmul(y1))
    out_ratio=(y1[:,2]/(y1.sum(dim=1)+1e-4))


    y = list(in_ratio[data.y==1].numpy())+list(in_ratio[data.y==0].numpy())+list(out_ratio[data.y==1].numpy())+list(out_ratio[data.y==0].numpy())
    x = ['In-Neighbors']*((data.y<=1).sum())+['Out-Neighbors']*((data.y<=1).sum())
    label = ['Fraudsters']*((data.y==1).sum())+['Normal users']*((data.y==0).sum())+['Fraudsters']*((data.y==1).sum())+['Normal users']*((data.y==0).sum())
    plot_data=pd.DataFrame()
    plot_data['y']=y
    plot_data['x']=x
    plot_data['label']=label

    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2

    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=45)
    plt.yticks([0,0.5,1.0,1.5,2.0,2.5],fontsize=45)

    ax = sns.barplot(x="x", y="y", hue="label",
                        data=plot_data, palette=['r','b'],capsize=0.02,errwidth=1.5,linewidth=1.0,edgecolor=".2")

    #plt.ylim([0, 10])
    plt.xlabel(' ',fontsize=50)
    plt.ylabel('Ratio of BN',fontsize=50)

    plt.ylim(0,0.8)
    plt.legend(loc = 'best',fontsize=40,markerscale=2)
    plt.tight_layout()
    plt.savefig('./figure7.pdf',bbox_inches='tight',  format='pdf')

#plt.show()