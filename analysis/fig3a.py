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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
if __name__ == "__main__":

    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=True)
    tsne = TSNE()
    Y = tsne.fit_transform(data.x[20000:120000])
   # Y = np.load('./tsne.npy')
    label = data.y[20000:120000]

    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2



    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    s1 = plt.scatter(x=Y[:,0][label<2][:5000],y=Y[:,1][label<2][:5000],color='b',marker='.',alpha=0.2,linewidths=5.2,edgecolors=None)

    s2 = plt.scatter(x=Y[:,0][label>1][:5000],y=Y[:,1][label>1][:5000],color='r',marker='.',alpha=0.2,linewidths=5.2,edgecolors=None)



    #plt.legend(fontsize=32)


    plt.xlabel('$x$',fontsize=50)
    plt.ylabel('$y$',fontsize=50)
    plt.legend((s1,s2),('Other nodes','Background nodes') ,loc = 'upper left',fontsize=40,markerscale=12,handletextpad=0)
    #plt.ylim(0,6)
    #plt.xlim(-0.1,1)
    plt.tight_layout()
    plt.savefig('./figure5.pdf',bbox_inches='tight', format='pdf')

#plt.show()
