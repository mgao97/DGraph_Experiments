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

if __name__ == "__main__":
    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=False)

    x = (data.edge_time.max()-data.edge_time.min())/30
    edge_time = ((data.edge_time-data.edge_time.min())/x).long()

    du_data = pd.DataFrame(data.edge_index.T.numpy())
    du_data['time']= edge_time.view(-1).numpy()
    #data = pd.DataFrame()
    x = []
    hue = []
    y = []
    for i in range(2,6):
        ids = du_data.groupby(0).count().index.values
        degree = du_data.groupby(0).count().values
        ids2 = ids[degree[:,0]==i]
        values = du_data.groupby(0).max()['time'].values-du_data.groupby(0).min()['time'].values
        values = values[degree[:,0]==i]
        label = data.y[ids2].numpy()
        y = y+list(values[label==1]/(i-1))+list(values[label==0]/(i-1))
        x = x+[i]*len(values[label==1])+[i]*len(values[label==0])
        hue = hue+['Fraudsters']*len(values[label==1])+['Normal users']*len(values[label==0])
        print(values[label==1].mean()/(i-1))
        print(values[label==0].mean()/(i-1))
    plot_data = pd.DataFrame()
    plot_data['x']=x
    plot_data['y']=y
    plot_data['label']=hue


    #plt.rcParams['font.sans-serif'] = ['Times New Roman']

    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    #pic_id = 2

    plt.figure(figsize=(10, 8))
    plt.xticks([2,3,4,5],fontsize=45)
    plt.yticks([0,2,4,6],fontsize=45)
    ax = sns.lineplot(data=plot_data,x='x',y='y',hue='label',style="label",palette=['r','b'],sizes=[12,23],markersize='15',markers=['o']*2,markeredgecolor=None,legend='full')
    #for h in ax.legend_.legendHandles: 
    #    h.set_marker('o')
    #ax.legend(loc="upper right",fontsize=32,markerscale=32)

    #plt.ylim([0, 10])
    plt.xlabel('Deg.',fontsize=50)
    plt.ylabel('Gap of each edges',fontsize=50)
    plt.legend(loc = 'best',fontsize=40,markerscale=2)
    plt.ylim(0,6)
    #plt.xlim(-0.1,1)
    plt.tight_layout()
    plt.savefig('./figure4.pdf',bbox_inches='tight', format='pdf')

#plt.show()