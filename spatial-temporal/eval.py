import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph

@torch.no_grad()
def evaluate(model, dataset, device, args):
    model.eval()
    cost_te = 0
    for time, snapshot in enumerate(dataset):
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
        cost_te += torch.mean((y_hat - snapshot.y) ** 2)
    cost_te = cost_te / (time + 1)

    return cost_te.item()