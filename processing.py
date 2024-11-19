import torch
import sys
import numpy as np
from torch_sparse import SparseTensor, mul,eye
from torch_sparse import sum as sparsesum
from dataset_loader import data_loader
from torch_sparse import matmul as torch_sparse_matmul
from torch_geometric.utils import *
from torch_sparse import fill_diag



def load_dataset(args):
    root = "./data/"
    dl = data_loader(root+args.dataset)

    #futures
    #use one-hot index vectors for nodes with no features (not used)
    feature_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            feature_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            feature_list.append(torch.FloatTensor(th))
    
    #labels and idx
    num_classes = dl.labels_test['num_classes']
    if args.dataset == "IMDB":
        labels = torch.FloatTensor(dl.labels_train['data']+dl.labels_test['data'])
    else:
        labels = torch.LongTensor(dl.labels_train['data']+dl.labels_test['data']).argmax(dim=1)

    val_ratio = 0.2 #split used in HGB 
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]

    adjs = []
    N = dl.nodes['total']

    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        row = v.row 
        col = v.col 
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=(N,N))
        adjs.append(adj)

    if args.dataset == "DBLP":
        adjs = [adjs[0], adjs[1], adjs[2], adjs[4], adjs[3], adjs[5]]
    
    adjs_new =[]
    for i in range(len(adjs)):
        if i%2==0:
            adjs_new.append(adjs[i])
    for i in range(len(adjs)):
        if i%2==0:
            adjs_new.append(adjs[i+1])

    if args.dataset == "ACM":
        row0, col0, _ = adjs_new[0].coo()
        row1, col1, _ = adjs_new[4].coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=(N,N))
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs_new[0]=PP

    for i in range(len(adjs_new)):
        adj = adjs_new[i]
        adj = adj.fill_value(1.)
        deg = sparsesum(adj, dim=1)
        deg_inv_sqrt = deg.pow_(-1.0)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) #D^-1A
        adjs_new[i] = adj

    # DBLP
    adjs={} 
    meta_paths = []
    if args.dataset=="DBLP":
        target = "A"
        target_max = 4056
        meta_paths =  ['APA','APTPA','APVPA']
        adjs["AP"]=adjs_new[0].to(args.device)
        adjs["PT"]=adjs_new[1].to(args.device)
        adjs["PV"]=adjs_new[2].to(args.device)
        adjs["PA"]=adjs_new[3].to(args.device)
        adjs["TP"]=adjs_new[4].to(args.device)
        adjs["VP"]=adjs_new[5].to(args.device)
    if args.dataset=="ACM":
        target = "P"
        target_max = 3024
        meta_paths =  ['PPP','PAP','PCP','PKP']
        adjs["PP"]=adjs_new[0].to(args.device)
        adjs["PA"]=adjs_new[1].to(args.device)
        adjs["PC"]=adjs_new[2].to(args.device)
        adjs["PK"]=adjs_new[3].to(args.device)
        # adjs["IP"]=adjs_new[4].to(args.device)
        adjs["AP"]=adjs_new[5].to(args.device)
        adjs["CP"]=adjs_new[6].to(args.device)
        adjs["KP"]=adjs_new[7].to(args.device)
    if args.dataset=="IMDB":
        target = "M"
        target_max = 4931
        adjs["MD"]=adjs_new[0].to(args.device)
        adjs["MA"]=adjs_new[1].to(args.device)
        adjs["MK"]=adjs_new[2].to(args.device)
        adjs["DM"]=adjs_new[3].to(args.device)
        adjs["AM"]=adjs_new[4].to(args.device)
        adjs["KM"]=adjs_new[5].to(args.device)
    if args.dataset=="AMiner":
        target = "P"
        target_max = 6563
        adjs["PA"]=adjs_new[0].to(args.device)
        adjs["PR"]=adjs_new[1].to(args.device)
        adjs["AP"]=adjs_new[2].to(args.device)
        adjs["RP"]=adjs_new[3].to(args.device)

    print(args.dataset)

    lis = []
    adjs_list = []

    if args.dataset=="DBLP":
        for meta_path in meta_paths:
            result = None
            for i in range(len(meta_path)-2):
                if i==0: 
                    result = torch_sparse_matmul(adjs[meta_path[i:i+2]],adjs[meta_path[i+1:i+3]])
                else:
                    result = torch_sparse_matmul(result,adjs[meta_path[i+1:i+3]])
            row, col, value = result.coo()
            mask = (row <= target_max) & (col <= target_max)
            filtered_row = row[mask]
            filtered_col = col[mask]
            filtered_value = value[mask]
            result = SparseTensor(row=filtered_row, col=filtered_col, value=filtered_value,
                    sparse_sizes=(target_max+1,target_max+1))
            adjs_list.append(result)
            lis.append(meta_path)
    else:
        for i in adjs.keys():
            for j in adjs.keys():
                if i[-1]==j[0] and i[0]==target and j[1]==target:
                    temp = torch_sparse_matmul(adjs[i],adjs[j])
                    row, col, value = temp.coo()
                    mask = (row <= target_max) & (col <= target_max)
                    filtered_row = row[mask]
                    filtered_col = col[mask]
                    filtered_value = value[mask]
                    temp = SparseTensor(row=filtered_row, col=filtered_col, value=filtered_value,
                    sparse_sizes=(target_max+1,target_max+1))
                    adjs_list.append(temp)
                    lis.append(i+j)

    lis_t={}
    
    return feature_list, adjs, lis, lis_t, labels, num_classes, train_idx, val_idx, test_idx,adjs_list


