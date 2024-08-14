import torch
import numpy as np
from torchmetrics import AUROC, F1, Recall, Accuracy
from sklearn.metrics import roc_auc_score, f1_score

def eval_rocauc(y_pred, y_true):
    # auroc=AUROC(pos_label=1,num_classes=valid_class).to(y_pred.device)
    auroc=AUROC().to(y_pred.device)
    return auroc(y_pred,y_true)

def eval_f1(y_pred, y_true):
    f1 = F1(num_classes=2, multiclass=True).to(y_pred.device)
    return f1(y_pred, y_true)

def eval_recall(y_pred, y_true):
    recall = Recall(num_classes=2, multiclass=True).to(y_pred.device)
    return recall(y_pred, y_true)

def eval_accuracy(y_pred, y_true):
    with open('file.txt', 'a') as file:
        file.write('==================\n')
        file.write('y_pred: \n')
        file.write(str(y_pred))
        file.write('\n')
        file.write('y_true: \n')
        file.write(str(y_true))
        file.write('\n')
        file.write('==================\n')
        
    accuracy = Accuracy().to(y_pred.device)
    return accuracy(y_pred, y_true)


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
