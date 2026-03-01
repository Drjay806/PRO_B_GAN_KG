from typing import Dict, Iterable, Tuple

import numpy as np


def ranking_metrics(ranks: Iterable[int]) -> Dict[str, float]:
    ranks = np.array(list(ranks), dtype=np.float32)
    mrr = np.mean(1.0 / ranks)
    hits1 = np.mean(ranks <= 1)
    hits3 = np.mean(ranks <= 3)
    hits10 = np.mean(ranks <= 10)
    return {
        "mrr": float(mrr),
        "hits@1": float(hits1),
        "hits@3": float(hits3),
        "hits@10": float(hits10),
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]

    tp = 0
    fp = 0
    fn = np.sum(labels == 1)
    tn = np.sum(labels == 0)

    tpr = []
    fpr = []
    precision = []
    recall = []

    for label in labels:
        if label == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / max(tp + fn, 1))

    roc_auc = np.trapz(tpr, fpr)
    aupr = np.trapz(precision, recall)

    best_f1 = 0.0
    best_threshold = 0.0
    best_mcc = -1.0
    best_mcc_threshold = 0.0

    for threshold in scores:
        preds = (scores >= threshold).astype(np.int32)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / max(mcc_denom, 1e-9)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        if mcc > best_mcc:
            best_mcc = mcc
            best_mcc_threshold = threshold

    return {
        "roc_auc": float(roc_auc),
        "aupr": float(aupr),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(best_threshold),
        "best_mcc": float(best_mcc),
        "best_mcc_threshold": float(best_mcc_threshold),
    }
