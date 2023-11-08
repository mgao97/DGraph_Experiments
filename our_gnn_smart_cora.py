# dataset name: DGraphFin

# from utils import DGraphFin
from torch_geometric.datasets import Planetoid
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from utils.tricks import Missvalues
from utils.tricks import Background
from utils.tricks import Structure
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2,RGCN, SMART, GAT_NeighSampler
from logger import Logger
from torch_geometric.data import NeighborSampler
from tqdm import tqdm
import argparse

from scipy.sparse import coo_matrix

from utils.utils import load_data

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric as tg
import torch_geometric.transforms as T

from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
import time
eval_metric = 'auc'

mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gcn_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

sage_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gcn_smart_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':64
              , 'dropout':0
              , 'simi_threshold':0.5
              , 'batchnorm': False
              , 'l2':5e-7
}

gat_neighsampler_parameters = {'lr':0.003
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             , 'layer_heads':[4,1]
             }

def train(model, data, train_idx, optimizer, no_conv=False,is_rgcn=False):
    # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        if(is_rgcn):
            out = model(data.x, data.edge_index, data.edge_type)[train_idx]
        else:
            out = model(data.x, data.edge_index)[train_idx]
    loss = F.cross_entropy(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False,is_rgcn=True):
    # data.y is labels of shape (N, )
    model.eval()
    
    if no_conv:
        out = model(data.x)
    else:
        if(is_rgcn):
            out = model(data.x, data.edge_index, data.edge_type)
        else:
            out = model(data.x, data.edge_index)
        
    y_pred = out.exp()  # (N,num_classes)
    # print(y_pred)
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.cross_entropy(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])
            
    return eval_results, losses, y_pred


def batch_train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(data.x[n_id], adjs)
        loss = F.nll_loss(out, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def batch_test(layer_loader, model, data, split_idx, evaluator, device, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.inference(data.x, layer_loader, device)
#     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)   
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
            
    return eval_results, losses, y_pred       
            
def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='gcn_smart')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--MV_trick', type=str, default='null')
    parser.add_argument('--BN_trick', type=str, default='null')
    parser.add_argument('--BN_ratio', type=float, default=0.1)
    parser.add_argument('--Structure', type=str, default='original')
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
   # device = torch.device('cpu')
    dataset = Planetoid(root='data', name=args.dataset, transform=T.NormalizeFeatures())

    
    nlabels = dataset.num_classes
        
    data = dataset[0]
    # data.edge_index = data.adj_t
    # data.adj_t = torch.cat([data.edge_index.coo()[0].view(1,-1),data.edge_index.coo()[1].view(1,-1)],dim=0)
    # data.edge_index = data.adj_t
    # structure = Structure(args.Structure)
    # data = structure.process(data)
    # data.adj_t = data.edge_index
    # data.adj_t = tg.utils.to_undirected(data.adj_t)
    # data.edge_index = data.adj_t
    
    adj, features, idx_train, idx_val, idx_test, labels = load_data('cora')
    data.adj = adj
    
    # x = data.x
    # x = (x-x.mean(0))/x.std(0)
    # data.x = x
    if data.y.dim()==2:
        data.y = data.y.squeeze(1)        
    
    print(data)
    
    
    # missvalues = Missvalues(args.MV_trick)
    # data = missvalues.process(data)
    
    #print(data.edge_index)
    
    # data.edge_index = data.adj_t
    # BN = Background(args.BN_trick)
    # data = BN.process(data,args.BN_ratio)
    
    split_idx = {'train':data.train_mask, 'valid':data.val_mask, 'test':data.test_mask}

    fold = args.fold
    if split_idx['train'].dim()>1 and split_idx['train'].shape[1] >1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False

    split_idx = {'train':data.train_mask, 'valid':data.val_mask, 'test':data.test_mask}

    data = data.to(device)
    train_idx = split_idx['train'].to(device)
        
    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)

    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 5], batch_size=1024, shuffle=True, num_workers=12)
    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12)   
    
    is_rgcn=False
    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gcn':   
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GCN(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage':        
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = SAGE(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'rgcn':   
        para_dict = gcn_parameters
        model = RGCN(data.x.size(-1),16,2,4).to(device)
        is_rgcn=True
    if args.model == 'gat_neighsampler':   
        para_dict = gat_neighsampler_parameters
        model_para = gat_neighsampler_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GAT_NeighSampler(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)   
    if args.model == 'gcn_smart':
        para_dict = gcn_smart_parameters
        model_para = gcn_smart_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')  
        model = SMART(data.x.size(-1), hidden_dim = 64, output_dim = nlabels, dropout=0, similarity_threshold=0.5, num_layers=2).to(device)
    print(f'Model {args.model} initialized')

    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)
    # weight = torch.tensor([1,50]).to(device).float()
   
    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))

        # model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        min_valid_loss = 1e8
        best_out = None
        
        time_ls = []
        starttime = time.time()
        for epoch in range(1, args.epochs+1):
            starttime = time.time()
            loss = batch_train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv)
            
            endtime = time.time()
            time_ls.append(endtime-starttime)
            eval_results, losses, out = batch_test(layer_loader, model, data, split_idx, evaluator, device, no_conv)
            train_auc, valid_auc, test_auc = eval_results['train']['auc'], eval_results['valid']['auc'], eval_results['test']['auc']
            train_ap, valid_ap, test_ap = eval_results['train']['ap'], eval_results['valid']['ap'], eval_results['test']['ap']
                                                                                                 
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']
            #print(eval_results['train'])
#                 if valid_eval > best_valid:
#                     best_valid = valid_result
#                     best_out = out.cpu().exp()
            
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()
            
            
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train AUC: {train_auc:.3f} '
                          f'Train AP: {train_ap:.3f} '
                          f'Valid AUC: {valid_auc:.3f} '
                          f'Valid AP: {valid_ap:.3f} '
                          f'Test AUC: { test_auc:.3f} '
                          f'Test AP: { test_ap:.3f} '
                          f'Train time(s): {np.mean(time_ls):.3f}')
                
                
                time_ls = []
            logger.add_result(run, [train_auc, valid_auc, test_auc])

        logger.print_statistics(run)

    final_results = logger.print_statistics()
    print('final_results:', final_results)
    para_dict.update(final_results)
    pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir+'/cora_gat_results.csv')


if __name__ == "__main__":
    main()
