import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    out = F.log_softmax(out, dim=1)
    valid_loss = criterion(
        out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
