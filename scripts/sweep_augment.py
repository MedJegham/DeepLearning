"""Varie le facteur d'augmentation (ex. 5, 10, 15) pour un même modèle et enregistre les métriques."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn", choices=["mlp", "cnn", "resnet", "resnet18_tl"])
    p.add_argument("--factors", nargs="+", type=int, default=[5, 10, 15])
    p.add_argument("--epochs", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    out = []
    for fac in args.factors:
        cmd = [
            sys.executable,
            str(root / "train_and_save.py"),
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--augment",
            str(fac),
        ]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)
        summ = json.loads((root / "models" / "last_run_summary.json").read_text(encoding="utf-8"))
        summ["augment_factor"] = fac
        out.append(summ)
    path = root / "output" / f"augment_sweep_{args.model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Résultats : {path}")


if __name__ == "__main__":
    main()
