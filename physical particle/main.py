from utils import get_data_loaders

import argparse
import sys
import os, random
import numpy as np
import torch
import copy
import yaml
import time

import torch.nn as nn
import torch.nn.functional as F

from parse import parser_add_main_args, parse_method
from models import GraphGNN
from eval import eval_recall, eval_rocauc, eval_f1, eval_accuracy, eval_batch
from logger import Logger

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

config_path=f'configs/{args.dataset}.yml'
with open(config_path,'r') as f:
    config=yaml.safe_load(f)
loaders, _, dataset=get_data_loaders(args.data_dir, args.dataset,args.batch_size,config,args.seed)

train_loader, valid_loader, test_loader=loaders['train'], loaders['valid'], loaders['test']

# sample=next(iter(train_loader))

d=dataset[0].x.shape[1]
c=1 # all are binary classification

print(f'# features: {d}, # classes: {c}')


gnn_node = parse_method(args,c,d,device)
model=GraphGNN(args.hidden_channels,c,gnn_node,args.graph_pooling).to(device)

criterion = nn.BCEWithLogitsLoss()

## Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
elif args.metric == 'recall':
    eval_func = eval_recall
elif args.metric == 'acc':
    eval_func = eval_accuracy
else:
    raise MetricError
# eval_func=eval_rocauc

logger = Logger(args.runs, args)
model.train()

start=time.time()
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        
        for batch in train_loader:
            batch=batch.to(device)
            out=model(batch.x, batch.edge_index, batch.n_nodes)
            if len(batch.y.shape)==1:
                batch.y=batch.y.unsqueeze(1)
            loss=criterion(out,batch.y.to(torch.float)) # to float?

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        train_res=eval_batch(model,train_loader,eval_func,device)
        valid_res=eval_batch(model,valid_loader,eval_func,device)
        test_res=eval_batch(model,test_loader,eval_func,device)
        result=[train_res,valid_res,test_res]
        logger.add_result(run, result)

        # print(1)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
out_folder='results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

train_time=time.time()-start

def make_print(method):
    print_str=''
    if method=='gcn':
        print_str+=f'method: {args.method} layers: {args.num_layers} hidden: {args.hidden_channels} lr: {args.lr} dropout: {args.dropout} decay: {args.weight_decay}\n'
    elif method=='difformer':
        print_str+=f'method: {args.method} hidden: {args.hidden_channels} layers: {args.num_layers} lr: {args.lr} decay: {args.weight_decay} dropout: {args.dropout} epochs: {args.epochs} kernel: {args.kernel} use_graph: {args.use_graph} graph_weight: {args.graph_weight} alpha: {args.alpha} metric: {args.metric}\n'
    elif method=='gat':
        print_str+=f'method: {args.method} hidden: {args.hidden_channels} lr: {args.lr} dropout: {args.dropout} decay: {args.weight_decay} heads: {args.gat_heads} layers: {args.num_layers}\n'
    elif method=='mlp':
        print_str+=f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr} dropout: {args.dropout} decay: {args.weight_decay} n_layers:{args.num_layers}\n'
    else:
        print_str+=f'method: {args.method} hidden: {args.hidden_channels} lr:{args.lr} k:{args.k}\n'
    return print_str

file_name=f'{args.dataset}_{args.method}.txt'
out_path=os.path.join(out_folder,file_name)
with open(out_path,'a+') as f:
    print_str=make_print(args.method)
    f.write(print_str)
    f.write(results)
    f.write(f'\ntime: {train_time}')
    f.write('\n\n')

