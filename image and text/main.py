import argparse
import os, random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter
from sklearn.neighbors import kneighbors_graph

from logger import Logger, save_result
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, to_sparse_tensor, \
    class_rand_splits ,adj_mul
from eval import evaluate
from parse import parse_method, parser_add_main_args

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name=args.dataset
if dataset_name.startswith('mini') or dataset_name.startswith('20news') or dataset_name in ['stl10','cifar10']:
    adj = kneighbors_graph(dataset.graph['node_feat'],n_neighbors=args.k, include_self=True)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    dataset.graph['edge_index'] = edge_index

split_idx_lst = [dataset.get_idx_split(split_type='class', train_prop=args.train_prop, valid_prop=args.valid_prop, label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]

n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

### Load method ###
model = parse_method(args, dataset, n, c, d, device)

### Loss function and metric ###
criterion = nn.NLLLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)


### Training loop ###
best_emb,best_model=None,None
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        if args.method == 'manireg':
            row, col = edge_index
            loss_reg = torch.mean(torch.square(out[row] - out[col]).sum(-1))

        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])

        if args.method == 'manireg':
            loss += args.manireg * loss_reg

        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float() / pred.shape[0])
    logger.print_statistics(run)


results = logger.print_statistics()

### Save results ###
if args.save_result:
    save_result(args, results)