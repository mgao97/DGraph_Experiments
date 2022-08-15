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
def main():
    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=False,)


    degree = pd.DataFrame(data.edge_index.T.numpy()).groupby(0).count().values
    ids = pd.DataFrame(data.edge_index.T.numpy()).groupby(0).count().index.values
    key = {}
    for i in range(data.x.shape[0]):
        key[i]=0
    for i in range(len(ids)):
        key[ids[i]]=degree[i][0]
    all_degree = np.array(list(key.values()))
    all_out_degree = all_degree


    degree = pd.DataFrame(data.edge_index.T.numpy()).groupby(1).count().values
    ids = pd.DataFrame(data.edge_index.T.numpy()).groupby(1).count().index.values
    key = {}
    for i in range(data.x.shape[0]):
        key[i]=0
    for i in range(len(ids)):
        key[ids[i]]=degree[i][0]
    all_degree = np.array(list(key.values()))
    all_in_degree = all_degree

    ab_in_d = all_in_degree[data.y==1]
    ab_out_d = all_out_degree[data.y==1]
    normal_in_d = all_in_degree[data.y==0]
    normal_out_d = all_out_degree[data.y==0]

    plot_data = pd.DataFrame()
    plot_data['y']=list(ab_in_d)+list(normal_in_d)+list(ab_out_d)+list(normal_out_d)
    plot_data['x']=['In deg.']*(len(ab_in_d)+len(normal_in_d))+['Out deg.']*(len(ab_out_d)+len(normal_out_d))
    plot_data['label']=['Fraudsters']*len(ab_in_d)+['Normal users']*len(normal_in_d)+['Fraudsters']*len(ab_out_d)+['Normal users']*len(normal_out_d)

    #plt.rcParams['font.sans-serif'] = ['Times New Roman']

    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2

    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=45)
    plt.yticks([0,0.5,1.0,1.5,2.0,2.5],fontsize=45)
    ax = sns.barplot(x="x", y="y", hue="label", data=plot_data, palette=['r','b'],capsize=0.02,errwidth=1.5,linewidth=1.0,edgecolor=".2")
    #plt.ylim([0, 10])
    plt.xlabel(' ',fontsize=50)
    plt.ylabel('Avg deg.',fontsize=50)

    plt.ylim(0,2.5)
    plt.legend(loc = 'best',fontsize=40,markerscale=2)
    plt.tight_layout()
    plt.savefig('./figure1.pdf',bbox_inches='tight',  format='pdf')

if __name__ == "__main__":
    main()

#plt.show()