"""
Evaluation Metrics
"""
import torch
import numpy as np
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)


def ddi_rate_score(record, ddi_matrix):
    """
    Calculate DDI (Drug-Drug Interaction) rate
    
    Args:
        record: Drug interaction records
        ddi_matrix: DDI adjacency matrix
    
    Returns:
        dd_cnt: Number of DDIs
        all_cnt: Total number of drug pairs
    """
    all_cnt = 0
    dd_cnt = 0
    tril_matrix = torch.tril(torch.ones(130, 130).to(record[0].device))
    tril_matrix -= torch.eye(130, device=record[0].device)
    
    for output in record:
        dd_cnt += (output * ddi_matrix * tril_matrix).sum().sum()
        all_cnt += (output * tril_matrix).sum().sum()
    
    if all_cnt == 0:
        return 0, 1
    return dd_cnt, all_cnt


def multi_label_metric(y_gt, y_pred, y_prob, voc_size=None):
    """
    Calculate multi-label classification metrics
    
    Args:
        y_gt: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        voc_size: Vocabulary size
    
    Returns:
        jaccard: Jaccard score
        prauc: Precision-Recall AUC
        F1: F1 score
    """
    y_gt = torch.concat(y_gt).reshape(-1, voc_size[2]).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().reshape(-1, voc_size[2]).detach().numpy()
    y_prob = torch.cat(y_prob).cpu().reshape(-1, voc_size[2]).detach().numpy()
    
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)
    
    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)
    
    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)
    
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    F1 = f1(y_gt, y_pred)
    
    return ja, prauc, F1

