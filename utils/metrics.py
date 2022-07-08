import torch
import torch.nn.functional as F
from torchmetrics.functional import confusion_matrix


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs,threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, dim=[0,1])
    fp = torch.sum(temp_inputs                       , dim=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , dim=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score
def results(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, num_classes=2).ravel()
    # assert y_true.ndim == 1
    # assert y_pred.ndim == 1 or y_pred.ndim == 2
    #
    # if y_pred.ndim == 2:
    #     y_pred = y_pred.argmax(dim=1)
    # #torch.from_numpy(targets).type(torch.FloatTensor).long()
    # tp = (y_true * y_pred).sum().to(torch.float32)
    # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    # fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    # fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    tn = tn.to(torch.float32)
    fp = fp.to(torch.float32)
    fn = fn.to(torch.float32)
    tp = tp.to(torch.float32)

    epsilon = 1e-10
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training #
    return accuracy, precision, recall, f1