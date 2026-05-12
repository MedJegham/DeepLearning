"""Entraîne un modèle, calcule métriques complètes, sauvegarde poids + JSON pour comparaison / Streamlit.

Usage :
    python train_and_save.py --model cnn --epochs 12 --augment 5
    python train_and_save.py --model resnet18_tl --epochs 15 --augment 5 --lr 3e-4 --imagenet-norm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data import augment_dataset, imagenet_normalize, load_train_data, to_tensor_dataset
from src.metrics import collect_predictions, evaluate_multiclass
from src.model import build_model, count_parameters
from src.train import get_device, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["cnn", "resnet", "mlp", "resnet18_tl"],
        default="cnn",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None,
                   help="défaut: 3e-4 pour resnet18_tl, 1e-3 sinon")
    p.add_argument("--augment", type=int, default=5,
                   help="facteur de multiplication (1 = pas d'augmentation)")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stopping", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="0 = désactivé")
    p.add_argument("--lr-plateau", action="store_true",
                   help="ReduceLROnPlateau sur la val loss")
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--imagenet-norm", action="store_true",
                   help="normalisation ImageNet (recommandé pour resnet18_tl)")
    p.add_argument("--no-imagenet-norm", action="store_true",
                   help="forcer sans norm ImageNet")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("models"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kaggle-score", type=float, default=None,
                   help="score public Kaggle (accuracy 0–1) à enregistrer dans le JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "resnet18_tl" and not args.no_imagenet_norm:
        args.imagenet_norm = True

    lr = args.lr
    if lr is None:
        lr = 3e-4 if args.model == "resnet18_tl" else 1e-3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Chargement des données depuis {args.data_dir}/")
    images, labels = load_train_data(args.data_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    print(f"  train {X_train.shape}, val {X_val.shape}")

    if args.augment > 1:
        print(f"Augmentation x{args.augment}...")
        X_train, y_train = augment_dataset(
            X_train, y_train, n_augmentations=args.augment, seed=args.seed
        )
        print(f"  train augmenté {X_train.shape}")

    Xt, yt = to_tensor_dataset(X_train, y_train)
    Xv, yv = to_tensor_dataset(X_val, y_val)
    if args.imagenet_norm:
        Xt = imagenet_normalize(Xt)
        Xv = imagenet_normalize(Xv)
        print("  Normalisation ImageNet appliquée aux tenseurs train/val.")

    train_loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(Xv, yv), batch_size=args.batch_size, shuffle=False
    )

    device = get_device()
    print(f"Périphérique : {device}")

    if args.model == "resnet18_tl":
        model = build_model("resnet18_tl", num_classes=5, pretrained=True)
    else:
        model = build_model(
            args.model,
            in_channels=3,
            num_classes=5,
            dimension_of_image=28,
        )
    n_params = count_parameters(model)
    print(f"Paramètres entraînables : {n_params:,}")
    print(model)

    t0 = time.perf_counter()
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        device=device,
        early_stopping_patience=args.early_stopping,
        label_smoothing=args.label_smoothing,
        use_lr_plateau=args.lr_plateau,
        plateau_patience=args.plateau_patience,
    )
    train_seconds = time.perf_counter() - t0
    print(f"Temps d'entraînement : {train_seconds:.1f} s")

    yt_np, yp_np, proba = collect_predictions(model, val_loader, device)
    full_metrics = evaluate_multiclass(yt_np, yp_np, proba, num_classes=5)
    acc = full_metrics["accuracy"]
    f1m = full_metrics["f1_macro"]
    roc = full_metrics["roc_auc_ovr_macro"]
    print(f"\nValidation accuracy = {acc:.4f}")
    print(f"F1 macro = {f1m:.4f}")
    print(f"ROC AUC OvR (macro) = {roc:.4f}")
    print("\n--- classification_report ---\n" + str(full_metrics["classification_report"]))

    weight_path = args.output_dir / f"{args.model}_best.pt"
    meta = {
        "model_name": args.model,
        "state_dict": model.state_dict(),
        "val_accuracy": acc,
        "val_f1_macro": f1m,
        "val_roc_auc_ovr_macro": roc,
        "num_classes": 5,
        "input_size": 28,
        "num_parameters": n_params,
        "train_seconds": train_seconds,
        "use_imagenet_norm": bool(args.imagenet_norm),
        "kaggle_public_score": args.kaggle_score,
    }
    torch.save(meta, weight_path)

    history_path = args.output_dir / f"{args.model}_history.json"
    history_path.write_text(
        json.dumps(
            {
                "train_loss": history.train_loss,
                "train_acc": history.train_acc,
                "val_loss": history.val_loss,
                "val_acc": history.val_acc,
                "lr": history.lr,
                "val_accuracy": acc,
                "val_f1_macro": f1m,
                "val_roc_auc_ovr_macro": roc,
                "train_seconds": train_seconds,
                "num_parameters": n_params,
                "args": vars(args)
                | {
                    "data_dir": str(args.data_dir),
                    "output_dir": str(args.output_dir),
                    "lr_used": lr,
                },
            },
            indent=2,
            default=str,
        )
    )

    summary_path = args.output_dir / "last_run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "val_accuracy": acc,
                "val_f1_macro": f1m,
                "val_roc_auc_ovr_macro": roc,
                "train_seconds": train_seconds,
                "num_parameters": n_params,
                "kaggle_public_score": args.kaggle_score,
            },
            indent=2,
        )
    )

    print(f"Modèle sauvegardé : {weight_path}")
    print(f"Historique : {history_path}")
    print(f"Résumé : {summary_path}")


if __name__ == "__main__":
    main()
