import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from ann_ids import load_dataset, make_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "demo_data"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "demo_model.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--save", default=str(DEFAULT_MODEL))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, paths = load_dataset(args.data, args.offset)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both benign and malicious samples.")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True
    )

    # Rprop is the resilient backpropagation family used by the paper.
    optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch:4d}/{args.epochs}  loss={running/len(Xtr):.5f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = (probs >= 0.5).astype(int)

    print("\n=== Test results ===")
    print("Accuracy :", f"{accuracy_score(yte, pred):.4f}")
    print("Precision:", f"{precision_score(yte, pred, zero_division=0):.4f}")
    print("Recall   :", f"{recall_score(yte, pred, zero_division=0):.4f}")
    if len(np.unique(yte)) == 2:
        print("AUROC    :", f"{roc_auc_score(yte, probs):.4f}")
    print("\nConfusion matrix [benign, malicious]:")
    print(confusion_matrix(yte, pred))
    print("\n" + classification_report(
        yte, pred, target_names=["benign", "malicious"], zero_division=0
    ))

    torch.save({
        "state_dict": model.state_dict(),
        "offset": args.offset,
        "input_size": 1000,
        "classes": ["benign", "malicious"],
    }, args.save)
    print(f"Saved model -> {args.save}")

if __name__ == "__main__":
    main()
