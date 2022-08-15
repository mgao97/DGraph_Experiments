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
def main():
    datapath = '../dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=False,)


    MVR=(data.x==-1).sum(dim=1)*1.0
    MVR=MVR.float()
    plotdata = pd.DataFrame()
    plotdata['Ratio of missing values'] = list(MVR[data.y==1].numpy()/17)+list(MVR[data.y==0].numpy()/17)
    plotdata['label']=['Fraudsters']*len(MVR[data.y==1])+['Normal users']*len(MVR[data.y==0])
    #plotdata = plotdata.sample(1000000)
    sns.set_color_codes("pastel")
    plt.rc('font', family='Times New Roman')

    pic_id = 2

    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    ax = sns.kdeplot(data=plotdata[plotdata['label']=='Fraudsters'],x=
                'Ratio of missing values',multiple="layer",common_norm=False,common_grid=False,fill=True,color='r',legend=False,label='Fraudsters')
    ax = sns.kdeplot(data=plotdata[plotdata['label']=='Normal users'],x=
                'Ratio of missing values',multiple="layer",common_norm=False,common_grid=False,fill=True,color='b',legend=False,label='Normal users')

    #ax.legend(loc="upper right",fontsize=45)
    #plt.ylim([0, 10])
    plt.xlabel('Ratio of missing values',fontsize=50)
    plt.ylabel('Density',fontsize=50)
    plt.legend(loc = 'best',fontsize=40,markerscale=2)
    plt.ylim(0,10)
    plt.xlim(-0.1,1)
    plt.tight_layout()
    plt.savefig('./figure3.pdf',bbox_inches='tight',  format='pdf')
if __name__ == "__main__":
    main()

#

#plt.show()