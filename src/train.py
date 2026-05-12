"""Boucle d'entraînement / évaluation partagée par les notebooks et l'app."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _epoch_loop(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer | None, device: str) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)

        with torch.set_grad_enabled(is_train):
            scores = model(x)
            loss = criterion(scores, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (scores.argmax(1) == y).sum().item()
        total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader | None = None,
                num_epochs: int = 12, lr: float = 1e-3,
                weight_decay: float = 1e-4, optimizer_name: str = "adam",
                device: str | None = None, verbose: bool = True,
                early_stopping_patience: int | None = None,
                label_smoothing: float = 0.0,
                use_lr_plateau: bool = False,
                plateau_factor: float = 0.5,
                plateau_patience: int = 2,
                plateau_min_lr: float = 1e-6) -> TrainHistory:
    device = device or get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                     weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler: ReduceLROnPlateau | None = None
    if use_lr_plateau and val_loader is not None:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=plateau_factor, patience=plateau_patience,
            min_lr=plateau_min_lr,
        )

    history = TrainHistory()
    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = _epoch_loop(model, train_loader, criterion, optimizer, device)
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.lr.append(optimizer.param_groups[0]["lr"])
        log = f"Epoch {epoch:02d}/{num_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f}"

        if val_loader is not None:
            v_loss, v_acc = _epoch_loop(model, val_loader, criterion, None, device)
            history.val_loss.append(v_loss)
            history.val_acc.append(v_acc)
            log += f" | val loss {v_loss:.4f} acc {v_acc:.3f}"

            if scheduler is not None:
                scheduler.step(v_loss)

            if early_stopping_patience is not None:
                if v_loss < best_val - 1e-4:
                    best_val = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping à l'epoch {epoch}")
                        break
        if verbose:
            print(log)

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: str | None = None) -> Dict[str, float]:
    device = device or get_device()
    model.to(device).eval()
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        preds = model(x).argmax(1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
    return {"accuracy": total_correct / total_samples}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader,
            device: str | None = None) -> torch.Tensor:
    device = device or get_device()
    model.to(device).eval()
    all_preds = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        all_preds.append(model(x).argmax(1).cpu())
    return torch.cat(all_preds)
