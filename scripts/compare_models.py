"""Lance une série d'entraînements et produit un tableau Markdown + JSON agrégé.

Exemple :
    python scripts/compare_models.py --epochs 8 --augment 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "cnn", "resnet", "resnet18_tl"],
        choices=["mlp", "cnn", "resnet", "resnet18_tl"],
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--augment", type=int, default=3)
    p.add_argument("--lr-plateau", action="store_true")
    p.add_argument("--output-md", type=Path, default=Path("output/comparison_table.md"))
    p.add_argument("--output-json", type=Path, default=Path("output/comparison_results.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in args.models:
        cmd = [
            sys.executable,
            str(root / "train_and_save.py"),
            "--model",
            model,
            "--epochs",
            str(args.epochs),
            "--augment",
            str(args.augment),
        ]
        if args.lr_plateau:
            cmd.append("--lr-plateau")
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)
        summ = json.loads((root / "models" / "last_run_summary.json").read_text(encoding="utf-8"))
        rows.append(summ)

    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "| Modèle | # params | Temps (s) | Val acc | F1 macro | ROC AUC OvR | Kaggle |",
        "|--------|----------|-----------|---------|----------|-------------|--------|",
    ]
    for r in rows:
        kg = r.get("kaggle_public_score")
        kg_s = f"{kg:.4f}" if kg is not None else "—"
        roc = r.get("val_roc_auc_ovr_macro")
        if isinstance(roc, float) and roc == roc:
            roc_s = f"{roc:.4f}"
        else:
            roc_s = "nan"
        lines.append(
            f"| {r['model']} | {r['num_parameters']:,} | {r['train_seconds']:.1f} | "
            f"{r['val_accuracy']:.4f} | {r['val_f1_macro']:.4f} | {roc_s} | {kg_s} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nTableau écrit : {args.output_md}")
    print(f"JSON : {args.output_json}")


if __name__ == "__main__":
    main()
