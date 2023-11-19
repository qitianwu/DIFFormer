import torch

from torchmetrics import AUROC

def eval_rocauc(y_pred, y_true):
    # auroc=AUROC(pos_label=1,num_classes=valid_class).to(y_pred.device)
    auroc=AUROC().to(y_pred.device)
    return auroc(y_pred,y_true)

@torch.no_grad()
def eval_batch(model, loader, eval_func, device):
    res_list=[]
    for batch in loader:
        batch=batch.to(device)
        out=model(batch)
        # for binary classification, apply sigmoid
        out=torch.sigmoid(out)
        res=eval_func(out,batch.y.int()).item()
        res_list.append(res)

    res=torch.as_tensor(res_list)
    return res.mean()
