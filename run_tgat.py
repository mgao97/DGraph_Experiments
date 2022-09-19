from analysis.load_dataset import build_tg_data
import numpy as np
import torch
import pandas as pd

import torch.nn.functional as F
from sklearn import metrics
from torch_geometric.nn import TransformerConv
def evaluate(y_truth,y_pred):
    auc = metrics.roc_auc_score(y_truth, y_pred, multi_class='ovo',labels=[0,1],average='macro')
    ap = metrics.average_precision_score(y_truth, y_pred, average='macro', pos_label=1, sample_weight=None)
    return ap,auc
def process_data(data,max_time_steps=32):
    data.edge_time = data.edge_time-data.edge_time.min() #process edge time
    data.edge_time = data.edge_time/data.edge_time.max()
    data.edge_time = (data.edge_time*max_time_steps).long()
    data.edge_time = data.edge_time.view(-1,1).float()
    
    
    edge = torch.cat([data.edge_index,data.edge_time.view(1,-1)],dim=0) #process node time
    degree = pd.DataFrame(edge.T.numpy()).groupby(0).min().values
    ids = pd.DataFrame(data.edge_index.T.numpy()).groupby(0).count().index.values 
    key = {}
    for i in range(data.x.shape[0]):
        key[i]=0
    for i in range(len(ids)):
        key[ids[i]]=degree[i][1]
    node_time = np.array(list(key.values()))
    data.node_time=torch.tensor(node_time)
    
    # trans to undirected graph
    data.edge_index = torch.cat((data.edge_index,data.edge_index[[1,0],:]),dim=1)
    data.edge_time = torch.cat((data.edge_time,data.edge_time),dim=0)

    return data

class TimeEncode(torch.nn.Module): 
    # https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)#

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)
    
class TGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.time_enc = TimeEncode(32)
        edge_dim =32
        self.lin = torch.nn.Linear(17,32)
        self.conv = TransformerConv(32, 32 // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        self.conv1 = TransformerConv(32, 32 // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        self.out = torch.nn.Linear(32,2)
    def forward(self, x, edge_index, t):
        rel_t = data.node_time[edge_index[0]].view(-1,1) - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        #edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        h1 = self.lin(x)
        h1 = F.relu(h1)
        #print(h1.shape)
        h1 = self.conv(h1, edge_index, rel_t_enc)
        #h1 = F.relu(h1)
        #h2 = self.conv1(h1, edge_index, rel_t_enc)
        out = self.out(h1)
        return F.log_softmax(out,dim=1)

    
    

if __name__ == "__main__":
    datapath = './dataset/DGraphFin/raw/dgraphfin.npz'
    origin_data = np.load(datapath)
    data = build_tg_data(is_undirected=False,datapath=datapath)
    data = process_data(data)
    
    device = torch.device('cuda:1')
    #model = GCN(data.x.shape[1],2)
    model = TGAT(in_channels=17, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0e-4)
    data = data.to(device)
    model = model.to(device)
    lossf = torch.nn.CrossEntropyLoss()#.cuda(1)
    loss=None
    val_acc = 0
    test = 0
    weight = torch.tensor([1,50]).to(device).float()
    duration = 0
    y_valid = data.y[data.val_mask].cpu()#.numpy()
    for i in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(x = data.x,edge_index = data.edge_index,t = data.edge_time)
        loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask],weight=weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(x = data.x,edge_index = data.edge_index,t = data.edge_time)
            val_ap , val_auc = evaluate(y_valid.numpy(),scores[data.val_mask,1].cpu().numpy())
            val=val_auc
            if(val>val_acc):
                y_true = data.y[data.test_mask].cpu().numpy()
                y_scores = scores[:,1][data.test_mask].cpu().numpy()
                ap,auc=evaluate(y_true,y_scores)
                print('best (epoch,val_ap,val_auc,test_ap,test_auc):',i,val_ap,val_auc,ap,auc)
                val_acc=val
                duration=0