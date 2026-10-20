"""Evaluate a trained checkpoint on the Fashion-MNIST test set.

Usage: python -m src.evaluate --checkpoint checkpoints/best.pt
"""
import argparse, json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import CLASSES, get_loaders
from src.model import FashionCNN


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_dl = get_loaders(args.data_dir)
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cm = torch.zeros(10, 10, dtype=torch.long)
    for x, y in test_dl:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, pr in zip(y, pred):
            cm[t, pr] += 1

    acc = cm.diag().sum().item() / cm.sum().item()
    per_class = {c: cm[i, i].item() / cm[i].sum().item() for i, c in enumerate(CLASSES)}
    print(f"Test accuracy: {acc:.4f} ({cm.diag().sum().item()}/{cm.sum().item()})")
    for c, a in per_class.items():
        print(f"  {c:<12} {a:.4f}")
    with open("assets/test_metrics.json", "w") as f:
        json.dump({"test_accuracy": acc, "per_class_accuracy": per_class}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j].item(), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix (test acc {acc:.2%})")
    fig.tight_layout(); fig.savefig("assets/confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
