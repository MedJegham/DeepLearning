"""Métriques d'évaluation : accuracy, F1, ROC multiclasse (one-vs-rest)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (y_true, y_pred, proba_softmax) sur tout le loader."""
    model.eval()
    model.to(device)
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
) -> Dict[str, float | str]:
    """Accuracy, F1 macro, ROC AUC OvR (moyenne)."""
    acc = float((y_true == y_pred).mean())
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    try:
        auc = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    report = classification_report(
        y_true, y_pred, digits=4, zero_division=0, output_dict=False
    )
    return {"accuracy": acc, "f1_macro": f1m, "roc_auc_ovr_macro": auc, "classification_report": report}
