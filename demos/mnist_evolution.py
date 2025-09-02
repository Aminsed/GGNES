"""
MNIST Evolution Demo (GGNES)

Summary:
- Evolves simple MLP-like graphs on MNIST using GGNES translation to PyTorch
- Population-based search over activation, learning rate, hidden width, and epochs
- Logs per-generation results to CSV and prints best accuracy

Defaults are set for meaningful runs on CPU. Use --quick for a short sanity test.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
from ggnes.utils.rng_manager import RNGManager


def try_load_mnist(subset: int | None = None) -> Tuple[Tuple[object, object], Tuple[object, object], int]:
    """Load MNIST (prefers torchvision), fall back to sklearn digits if missing.

    Returns: ((X_train, y_train), (X_test, y_test), input_dim)
    X_* are torch.Tensors if torchvision is available; otherwise numpy arrays.
    """
    # Try torchvision MNIST
    try:
        import torch
        from torchvision import datasets, transforms

        tfm = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
        ])
        train_ds = datasets.MNIST(root='.data', train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root='.data', train=False, download=True, transform=tfm)
        def flatten(ds):
            X = ds.data.view(ds.data.size(0), -1).float() / 255.0
            y = ds.targets.long()
            return X, y
        X_train, y_train = flatten(train_ds)
        X_test, y_test = flatten(test_ds)
        if subset:
            X_train = X_train[:subset]
            y_train = y_train[:subset]
            X_test = X_test[: max(1000, subset // 5)]
            y_test = y_test[: max(1000, subset // 5)]
        return (X_train, y_train), (X_test, y_test), int(X_train.size(1))
    except Exception:
        # Fallback: sklearn digits (8x8 = 64 dims);
        from sklearn.datasets import load_digits
        import numpy as np
        digits = load_digits()
        X = digits.data.astype('float32') / 16.0
        y = digits.target.astype('int64')
        n = len(X)
        n_train = int(n * 0.8)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        if subset:
            X_train, y_train = X_train[:subset], y_train[:subset]
            X_test, y_test = X_test[: max(200, subset // 5)], y_test[: max(200, subset // 5)]
        return (X_train, y_train), (X_test, y_test), int(X.shape[1])


def build_graph(input_dim: int, hidden_dim: int, activation: str) -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': input_dim}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': activation, 'attributes': {'output_size': hidden_dim}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': 10}})
    g.add_edge(i, h, {'weight': 0.05})
    g.add_edge(h, o, {'weight': 0.05})
    return g


@dataclass
class Individual:
    activation: str
    lr: float
    hidden: int
    epochs: int
    fitness: float = 0.0


def train_eval(ind: Individual, input_dim: int, train, test) -> float:
    import torch
    torch.manual_seed(0)
    X_train, y_train = train
    X_test, y_test = test

    # Convert numpy to torch if needed
    if not hasattr(X_train, 'to'):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

    model = to_pytorch_model(build_graph(input_dim, ind.hidden, ind.activation), {'device': 'cpu', 'dtype': torch.float32})
    opt = torch.optim.Adam(model.parameters(), lr=ind.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size = 256
    steps = max(1, int(len(X_train) // batch_size))
    for _ in range(max(1, ind.epochs)):
        perm = torch.randperm(len(X_train))
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            opt.zero_grad()
            logits = model(xb, reset_states=True)  # shape [N, 10]
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits = model(X_test, reset_states=True)
        pred = logits.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    return float(acc)


def mutate(ind: Individual, rng: RNGManager) -> Individual:
    r = rng.get_context_rng('mutation')
    activations = ['relu', 'tanh']
    new_act = ind.activation if r.random() > 0.2 else r.choice(activations)
    new_lr = max(1e-5, min(5e-2, ind.lr * r.choice([0.5, 0.8, 1.0, 1.25, 1.5])))
    new_hidden = max(64, min(512, ind.hidden + r.choice([-64, -32, 0, 32, 64])))
    new_epochs = max(1, min(6, ind.epochs + r.choice([-1, 0, 1])))
    return Individual(new_act, new_lr, new_hidden, new_epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--gens', type=int, default=12)
    ap.add_argument('--subset', type=int, default=None, help='subset of training for speed; set None for full')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--quick', action='store_true', help='use a tiny config for sanity-run')
    args = ap.parse_args()

    if args.quick:
        args.pop = 8
        args.gens = 2
        args.subset = 2000
        args.epochs = 1

    rng = RNGManager(seed=1234)
    (X_train, y_train), (X_test, y_test), input_dim = try_load_mnist(args.subset)

    # Initialize population
    r = rng.get_context_rng('init')
    pop: List[Individual] = [
        Individual(
            activation=r.choice(['relu', 'tanh']),
            lr=r.choice([5e-4, 1e-3, 2e-3]),
            hidden=r.choice([128, 192, 256, 384]),
            epochs=args.epochs,
        )
        for _ in range(args.pop)
    ]

    rows: List[dict] = []
    for gen in range(args.gens):
        for ind in pop:
            ind.fitness = train_eval(ind, input_dim, (X_train, y_train), (X_test, y_test))
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        print(f"gen={gen} best_acc={best.fitness:.3f} act={best.activation} lr={best.lr} hidden={best.hidden} epochs={best.epochs}")

        # Log population
        for idx, ind in enumerate(pop):
            rows.append({
                'generation': gen,
                'idx': idx,
                'is_best': ind is best,
                'acc': round(ind.fitness, 6),
                'activation': ind.activation,
                'lr': ind.lr,
                'hidden': ind.hidden,
                'epochs': ind.epochs,
            })

        # Reproduce top-k
        parents = pop[: max(2, args.pop // 4)]
        children: List[Individual] = []
        while len(parents) + len(children) < len(pop):
            p = r.choice(parents)
            children.append(mutate(p, rng))
        pop = parents + children

    final_best = max(pop, key=lambda x: x.fitness)
    print(f"final_best_acc={final_best.fitness:.3f} act={final_best.activation} lr={final_best.lr} hidden={final_best.hidden} epochs={final_best.epochs}")

    # CSV output
    try:
        path = 'demos/mnist_log.csv'
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['generation', 'idx', 'is_best', 'acc', 'activation', 'lr', 'hidden', 'epochs'])
            w.writeheader()
            w.writerows(rows)
        print('csv_log:', path)
    except Exception:
        pass


if __name__ == '__main__':
    main()


