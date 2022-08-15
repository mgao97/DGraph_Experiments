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

from torch_geometric.nn.conv import MessagePassing
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor, matmul

import torch


def main():
    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=False,)
    class MyConv(MessagePassing):

        def __init__(self,**kwargs):
            kwargs.setdefault('aggr', 'mean')
            super().__init__(**kwargs)

        def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                    size: Size = None) -> Tensor:
            out = self.propagate(edge_index, x=x, size=size)
            return out

        def message(self, x_i,x_j: Tensor) -> Tensor:
            p = (x_i*x_j).sum(dim=1).view(-1,1)/(torch.norm(x_i,dim=1).view(-1,1)*torch.norm(x_j,dim=1).view(-1,1)+1e-5)

            return p

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}(nn={self.nn})'
    conv=MyConv()
    data = build_tg_data(is_undirected=False)
    edge_index = data.edge_index[[0,1],:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,2]+=(data.y>1)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y1 = conv(data.x,edge_index)
    edge_index = data.edge_index[[1,0],:]
    y = torch.zeros(data.x.shape[0],3)
    y[:,2]+=(data.y>1)
    y[:,1]+=(data.y==1)
    y[:,0]+=(data.y==0)
    y2 = conv(data.x,edge_index)



    plotdata = pd.DataFrame()
    plotdata['y'] = list(y1[data.y==1].view(-1).numpy())+list(y1[data.y==0].view(-1).numpy())+list(y2[data.y==1].view(-1).numpy())+list(y2[data.y==0].view(-1).numpy())
    plotdata['label']=['Fraudsters']*len(y1[data.y==1])+['Normal users']*len(y1[data.y==0])+['Fraudsters']*len(y1[data.y==1])+['Normal users']*len(y1[data.y==0])
    plotdata['x'] = ['In-neighbors']*(data.y<=1).sum()+['Out-neighbors']*(data.y<=1).sum()
    #plotdata = plotdata.sample(1000000)

    #plt.rcParams['font.sans-serif'] = ['Times New Roman']

    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2

    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=45)
    plt.yticks([0,0.2,0.4,0.6,0.9],fontsize=45)
    #ax = sns.barplot(data=plotdata,y=
    #            'Ratio of missing values',x='x',hue='label',palette=['r','b'])

    ax = sns.barplot(data=plotdata,y=
                'y',x='x',hue='label', palette=['r','b'],capsize=0.02,errwidth=1.5,linewidth=1.0,edgecolor=".2")
    ax.legend(loc="upper right",fontsize=45)
    #plt.ylim([0, 10])
    plt.xlabel(' ',fontsize=50)
    plt.ylabel('Avg cosine similarity',fontsize=50)
    plt.legend(loc = 'best',fontsize=40,markerscale=2)
    plt.ylim(0,0.6)
    #plt.xlim(-0.1,1)
    plt.tight_layout()
    plt.savefig('./figure2.pdf',bbox_inches='tight', format='pdf')
if __name__ == "__main__":
    main()
#plt.show()
#ax.legend(loc="upper right",fontsize=32)