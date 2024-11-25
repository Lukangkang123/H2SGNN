import argparse
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from processing import load_dataset
from utils import set_seed, evaluate, test
from model import H2SGNN


def training(args):
    set_seed(args.seed)
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    feature_list, adjs, lis, lis_t, labels, num_classes, train_idx, val_idx, test_idx,adjs_list = load_dataset(args)
    in_dim = feature_list[0].shape[1]
    feature_list = [feat.to(device) for feat in feature_list]
    labels = labels.to(device)
    num_relations = len(adjs)
    model = H2SGNN(in_dim, num_classes, lis, lis_t, args)
    model.to(device)
    
    optimizer = torch.optim.AdamW([
                { 'params': model.feat_project.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                { 'params': model.lin1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                { 'params': model.lin2.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                {'params': model.temp, 'weight_decay': args.temp_wd, 'lr': args.temp_lr},
                {'params': model.temp_list, 'weight_decay': args.temp_list_wd, 'lr': args.temp_list_lr},
                {'params': model.beta, 'weight_decay': args.beta_wd, 'lr': args.beta_lr}])
    if args.dataset =="IMDB":
        LOSS = torch.nn.BCEWithLogitsLoss()
    else:
        LOSS = F.cross_entropy

    print("start training...")
    best_epoch = 0
    result_micro=0
    result_macro=0
    best_val_loss=100000000
    
    num_target = feature_list[0].shape[0]
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = model(adjs, feature_list,adjs_list)

        logits = logits[:num_target]
        loss = LOSS(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        train_micro, train_macro, valid_micro, valid_macro, test_micro, test_macro, val_loss = test(model, adjs, feature_list, labels, train_idx, val_idx, test_idx, args.dataset,LOSS,adjs_list)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result_micro = test_micro
            result_macro = test_macro
            best_epoch = epoch
            cnt = 0
        else:
            cnt+=1

        if cnt==args.early_stopping:
            print("Early_stopping!")
            break

        if epoch%50==0:
            print("Epoch {:05d} | train_micro {:.4f} | Train Loss: {:.4f}| valid_micro {:.4f} | valid_loss {:.4f} |test_micro: {:.4f}" .
                format(epoch, train_micro, loss.item(),  valid_micro, val_loss, test_micro))
    
    print("Best epoch:{} | Test Micro: {:.4f} | Test Macro: {:.4f}".format(best_epoch, result_micro, result_macro))
   
    return result_micro, result_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H2SGNN')
    parser.add_argument("--dataset", type=str, default="ACM", help="The dataset used.")
    parser.add_argument("--seed", type=int, default=1,help="The seed used in the training.")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--run", type=int, default=5,help="The run times.")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early_stopping",type=int,default=100)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--input_drop", type=float, default=0)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1.0)
    parser.add_argument("--prop_lr", type=float, default=0.01)
    parser.add_argument("--prop_wd", type=float, default=1.0)
    parser.add_argument("--temp_lr", type=float, default=0.01)
    parser.add_argument("--temp_wd", type=float, default=1.0)
    parser.add_argument("--temp_list_lr", type=float, default=0.01)
    parser.add_argument("--temp_list_wd", type=float, default=1.0)
    parser.add_argument("--beta_lr", type=float, default=0.01)
    parser.add_argument("--beta_wd", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--alpha_list", type=float, default=0.1)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--b", type=float, default=0.1)
    parser.add_argument("--poly", type=str, default='GPR_legendre')  # GPR_GPR,GPR_legendre,GPR_jaco
    parser.add_argument("--u", type=float, default=0.5)
    args = parser.parse_args()

    with open('config.yaml', 'r') as c:
        config = yaml.safe_load(c)
         
    args.emb_dim = config[args.dataset]["emb_dim"]
    args.hidden = config[args.dataset]["hidden"]
    args.dropout = config[args.dataset]["dropout"]
    args.input_drop = config[args.dataset]["input_drop"]
    args.K = config[args.dataset]["K"]
    args.lr = config[args.dataset]["lr"]
    args.wd = config[args.dataset]["wd"]
    args.temp_list_lr = config[args.dataset]["temp_list_lr"]
    args.temp_list_wd = config[args.dataset]["temp_list_wd"]
    args.temp_lr = config[args.dataset]["temp_lr"]
    args.temp_wd = config[args.dataset]["temp_wd"]
    args.beta_lr = config[args.dataset]["beta_lr"]
    args.beta_wd =  config[args.dataset]["beta_wd"]
    args.alpha = config[args.dataset]["alpha"]
    args.alpha_list = config[args.dataset]["alpha_list"]
    args.a = config[args.dataset]["a"]
    args.b = config[args.dataset]["b"]
    args.poly = config[args.dataset]["poly"]
    print(args)


    results = []
    for i in range(1,args.run+1):
        args.seed = i
        result_micro, result_macro = training(args)
        results.append([result_micro,result_macro])

    micro_mean, macro_mean = np.mean(results, axis=0) * 100
    micro_std, macro_std = np.sqrt(np.var(results, axis=0)) * 100
    print(f'SHGCN on dataset {args.dataset}, in {args.run} repeated experiment:')
    print(f'Micro mean = {micro_mean:.4f} ± {micro_std:.4f}  \t  Macro mean =  {macro_mean:.4f} ± {macro_std:.4f}')




