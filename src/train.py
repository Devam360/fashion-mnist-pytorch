"""Train FashionCNN on Fashion-MNIST.

Usage: python -m src.train --epochs 5
"""
import argparse, json, os, time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import get_loaders
from src.model import FashionCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    train_dl, val_dl, _ = get_loaders(args.data_dir, args.batch_size, seed=args.seed)
    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"epoch {epoch}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} "
              f"val_loss={vl:.4f} val_acc={va:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    with open("assets/history.json", "w") as f:
        json.dump(history, f, indent=2)

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], "o-", label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], "o-", label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], "o-", label="val")
    ax[1].set(title="Accuracy", xlabel="epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig("assets/training_curves.png", dpi=120)
    print(f"best val_acc={best:.4f}; checkpoint saved to {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
