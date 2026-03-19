from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Split:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset


def ensure_node_features(dataset: TUDataset) -> None:
    # 有些 TUDataset 图没有 x 特征，这里用全 1 特征兜底
    for data in dataset:
        if getattr(data, "x", None) is None:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)


def split_dataset(dataset: TUDataset, train_ratio: float, val_ratio: float, seed: int) -> Split:
    n = len(dataset)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return Split(
        train=dataset[train_idx],
        val=dataset[val_idx],
        test=dataset[test_idx],
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GIN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer in range(num_layers):
            in_ch = in_dim if layer == 0 else hidden_dim
            mlp = MLP(in_ch, hidden_dim, hidden_dim)
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_add_pool(x, batch)
        out = self.classifier(g)
        return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)
        total_loss += float(loss) * data.num_graphs

        pred = logits.argmax(dim=-1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * data.num_graphs
        total += data.num_graphs

    return total_loss / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="GIN training (graph classification) with PyTorch Geometric")
    parser.add_argument("--dataset", type=str, default="MUTAG", help="TUDataset name, e.g. MUTAG/PROTEINS/IMDB-BINARY")
    parser.add_argument("--root", type=str, default=os.path.join("data", "TUDataset"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root=args.root, name=args.dataset)
    ensure_node_features(dataset)

    split = split_dataset(dataset, args.train_ratio, args.val_ratio, seed=args.seed)
    train_loader = DataLoader(split.train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(split.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(split.test, batch_size=args.batch_size, shuffle=False)

    sample = dataset[0]
    in_dim = sample.x.size(-1)
    num_classes = dataset.num_classes

    model = GIN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_test_acc_at_best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc_at_best_val = test_acc

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    print(f"Best val_acc={best_val_acc:.4f} | test_acc@best_val={best_test_acc_at_best_val:.4f}")


if __name__ == "__main__":
    main()