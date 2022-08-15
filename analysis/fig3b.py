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
if __name__ == "__main__":

    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=True)
    #tsne = TSNE()
    #Y = tsne.fit_transform(data.x[20000:120000])
    pure_idx = torch.arange(data.x.shape[0])[data.y<=1]

    pure_edge_index = tg.utils.subgraph(pure_idx,data.edge_index,relabel_nodes=True)[0]
    pure_data = tg.data.Data()
    pure_data.x = data.x[pure_idx]
    pure_data.edge_index = pure_edge_index

    netgraph = tg.utils.to_networkx(data,to_undirected=True)
    new_netgraph = tg.utils.to_networkx(pure_data,to_undirected=True)
    connected_components = [len(c) for c in sorted(nx.connected_components(new_netgraph), key=len, reverse=True)]
    connected_components = np.array(connected_components)
    print(connected_components.sum())
    data_list = []
    x = []
    for i in range(1,557):
        num = (connected_components==i).sum()
        print(num)
        if(num>0):
            data_list.append(num)
            x.append(i)
    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2



    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1,1,1)
    s1 = plt.scatter(y=data_list,x=x,color='w',marker='o',edgecolors='r',s=90,linewidths=3)
    s2 = plt.scatter(y=[1],x=[3700550],color='r',marker='*',edgecolors='r',s=120,linewidths=3)

    #s1 = plt.scatter(x=Y[:,0][label<2],y=Y[:,1][label<2],color='r',marker='o',alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    #s2 = plt.scatter(x=Y[:,0][label>1],y=Y[:,1][label>1],color='b',marker='+',alpha=0.9)

    plt.xticks([1e0,1e3,1e6],fontsize=45)
    plt.yticks([1e0,1e2,1e4,1e6],fontsize=45)
    #plt.legend(fontsize=32)


    plt.legend((s1,s2),('w/o BN','original graph'),handletextpad=0 ,loc = 'best',fontsize=40,markerscale=2)
    plt.xlabel('Size of components',fontsize=50)
    plt.ylabel('Count',fontsize=50)
    plt.ylim(0,1e7)
    #plt.xlim(-0.1,1)
    plt.tight_layout()
    plt.savefig('./figure6.pdf',bbox_inches='tight', format='pdf')

#plt.show()