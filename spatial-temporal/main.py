import argparse
import os
import random

import numpy as np
import torch
from torch_geometric.nn import knn_graph
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WikiMathsDatasetLoader, EnglandCovidDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from logger import Logger, save_result
from parse import parse_method, parser_add_main_args
from eval import evaluate

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Spatial Temporal Prediction')
parser_add_main_args(parser)
args = parser.parse_args()
if args.val_ratio == -1:
    args.val_ratio = args.train_ratio

print(args)
fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
if args.dataset.lower() == 'chickenpox':
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    n = 20
    d = 4
elif args.dataset.lower() == 'wikimath':
    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset(lags=14)
    n = 1068
    d = 14
elif args.dataset.lower() == 'twitter_rg':
    loader = TwitterTennisDatasetLoader()
    dataset = loader.get_dataset()
    n = 1000
    d = 16
elif args.dataset.lower() == 'twitter_uo':
    loader = TwitterTennisDatasetLoader(event_id="uo17")
    dataset = loader.get_dataset()
    n = 1000
    d = 16
elif args.dataset.lower() == 'covid':
    loader = EnglandCovidDatasetLoader()
    dataset = loader.get_dataset()
    n = 129
    d = 8
 
c = 1
train_dataset, val_test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)
val_dataset, test_dataset = temporal_signal_split(val_test_dataset, train_ratio=args.val_ratio/(1-args.train_ratio))
    
### Load method ###
model, suffix = parse_method(args, n, c, d, device)
if args.special_treat != 'None':
    args.method += '_' + args.special_treat

# logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
cost_list = []
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_cost = np.inf
    no_improve_epoch = 0
    if args.dataset.lower() != 'wikimath': # cumulative
        for param in model.parameters():
            if param.requires_grad:
                param.retain_grad()
    for epoch in range(args.epochs):
        model.train()
        cost_tr = 0
        for time, snapshot in enumerate(train_dataset):
            snapshot = snapshot.to(device)
            if args.special_treat.lower() == 'knn':
                snapshot.edge_index = knn_graph(snapshot.x, k=5, loop=True, cosine=True).to(device)
                snapshot.edge_attr = torch.ones(snapshot.edge_index.shape[1]).to(device)
            elif args.special_treat.lower() == 'dense':
                n = snapshot.x.shape[0]
                row = torch.arange(0, n).unsqueeze(1).repeat(1, n)
                col = torch.arange(0, n).unsqueeze(0).repeat(n, 1)
                snapshot.edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0).to(device)
                snapshot.edge_attr = torch.ones(snapshot.edge_index.shape[1]).to(device)
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            cost = torch.mean((y_hat-snapshot.y)**2)
            if args.dataset.lower() != 'wikimath': # cumulative
                cost_tr += cost
            else: # incremental
                cost_tr += cost.detach().item()
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()

        cost_tr = cost_tr / (time+1)

        if args.dataset.lower() != 'wikimath': # cumulative
            cost_tr.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        cost_val = evaluate(model, val_dataset, device, args)
        # result = [cost_tr, cost_val, cost_te]
        # logger.add_result(run, result)

        if cost_val < best_val_cost:
            best_val_cost = cost_val
            # if args.save_model:
            if not os.path.exists(f'models/{args.dataset}'):
                os.makedirs(f'models/{args.dataset}')
            torch.save(model.state_dict(), f'models/{args.dataset}/' + f'{args.method}-{suffix}.pkl')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= args.early_stopping:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Train Cost: {cost_tr:.4f}, '
                  f'Valid Cost: {cost_val:.4f}')
    # logger.print_statistics(run)
    model.load_state_dict(torch.load(f'models/{args.dataset}/' + f'{args.method}-{suffix}.pkl'))
    cost_te = evaluate(model, test_dataset, device, args)
    print(f'Test Cost: {cost_te:.4f}')
    cost_list.append(cost_te)

# results = logger.print_statistics()

results = np.array(cost_list)
print(f'Final Test: {results.mean():.4f} ± {results.std():.4f}')

# save results
if args.save_result:
    save_result(args, results)

