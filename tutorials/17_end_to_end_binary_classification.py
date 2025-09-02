"""
End-to-End Evolution: Binary Classification

Goals:
- Build a tiny population of individuals (genes = activation, lr, epochs)
- For each individual:
  - Build a graph → translate to PyTorch model
  - Train briefly on a synthetic binary classification task
  - Evaluate accuracy as fitness
- Run several generations with deterministic RNG and print best accuracy

Notes:
- We keep all layer widths equal to avoid random Linear projections so that
  initial parameters come from graph edge weights/biases deterministically.
- Torch layer randomness is controlled by torch.manual_seed for fairness.

Extras (observability and logging):
- How to produce a consolidated report and determinism signature (M32)
- How to write a simple CSV log of per-generation best metrics
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from typing import Dict, List

from ggnes.core.graph import Graph
from ggnes.core.node import NodeType
from ggnes.translation.pytorch import to_pytorch_model
from ggnes.utils.rng_manager import RNGManager


def make_dataset(n: int, rng: RNGManager) -> tuple[list[list[float]], list[int]]:
    # Linearly separable 2D points in [-1, 1]^2; label = 1 if x0 + x1 > 0
    r = rng.get_context_rng('dataset')
    xs: list[list[float]] = []
    ys: list[int] = []
    for _ in range(n):
        x0 = r.uniform(-1.0, 1.0)
        x1 = r.uniform(-1.0, 1.0)
        xs.append([x0, x1, 0.0])  # pad to width=3 for our fixed-width model
        ys.append(1 if (x0 + x1) > 0.0 else 0)
    return xs, ys


def build_graph(width: int, activation: str) -> Graph:
    g = Graph()
    i = g.add_node({'node_type': NodeType.INPUT, 'activation_function': 'linear', 'attributes': {'output_size': width}})
    h = g.add_node({'node_type': NodeType.HIDDEN, 'activation_function': activation, 'attributes': {'output_size': width}})
    o = g.add_node({'node_type': NodeType.OUTPUT, 'activation_function': 'linear', 'attributes': {'output_size': width}})
    g.add_edge(i, h, {'weight': 0.1})
    g.add_edge(h, o, {'weight': 0.1})
    return g


def train_and_eval(model, xs: list[list[float]], ys: list[int], epochs: int, lr: float) -> float:
    import torch
    torch.set_num_threads(1)
    device = next(model.parameters()).device
    X = torch.tensor(xs, dtype=torch.float32, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(max(1, int(epochs))):
        opt.zero_grad()
        logits_vec = model(X, reset_states=True)
        # Reduce vector outputs to a single logit per sample deterministically
        logits = logits_vec.mean(dim=1, keepdim=True)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits_vec = model(X, reset_states=True)
        logits = logits_vec.mean(dim=1, keepdim=True)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        acc = (pred.eq(y).float().mean()).item()
    return float(acc)


@dataclass
class Individual:
    activation: str
    learning_rate: float
    epochs: int
    fitness: float = 0.0


def mutate(ind: Individual, rng: RNGManager) -> Individual:
    r = rng.get_context_rng('mutation')
    activations = ['relu', 'tanh', 'sigmoid']
    new_act = ind.activation if r.random() > 0.3 else activations[r.randrange(len(activations))]
    scale = 1.0 if r.random() > 0.5 else r.choice([0.5, 0.8, 1.2, 1.5, 2.0])
    new_lr = max(1e-4, min(1.0, ind.learning_rate * scale))
    new_epochs = max(1, min(200, ind.epochs + r.choice([-5, -2, 0, 2, 5])))
    return Individual(new_act, new_lr, new_epochs)


def evaluate(ind: Individual, xs: list[list[float]], ys: list[int]) -> float:
    import torch
    torch.manual_seed(0)  # deterministic model init where used
    g = build_graph(width=3, activation=ind.activation)
    model = to_pytorch_model(g, {'device': 'cpu', 'dtype': torch.float32})
    acc = train_and_eval(model, xs, ys, epochs=ind.epochs, lr=ind.learning_rate)
    return acc


def main():
    rng = RNGManager(seed=123)
    xs, ys = make_dataset(200, rng)

    # --- Optional: Produce a consolidated report & determinism signature ---
    # This demonstrates how to build a report from a trivial derivation.
    # It is not used by the evolution loop below; it's purely for observability.
    # You can safely remove this block if not needed.
    try:
        from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
        from ggnes.hierarchical.derivation import DerivationEngine
        from ggnes.utils.observability import consolidated_report, determinism_signature

        demo_graph = Graph()
        demo_spec = ModuleSpec(
            name="Unit", version=1,
            parameters=[ParameterSpec("dim", default=1)],
            ports=[PortSpec("in", 1), PortSpec("out", 1)],
            invariants=["out.size == dim"],
        )
        engine = DerivationEngine(demo_graph)
        root = engine.expand(demo_spec, {"dim": 1})
        report = consolidated_report(engine, root, demo_graph, rng_manager=rng, device='cpu', dtype='float32')
        sig = determinism_signature(report)
        print("demo_report_schema:", report.get("schema_version"), "demo_sig:", sig)
    except Exception:
        # Observability demo is optional; continue even if unavailable in this environment
        pass

    # Initialize population
    r = rng.get_context_rng('init')
    pop: List[Individual] = [
        Individual(activation=r.choice(['relu', 'tanh']), learning_rate=r.choice([0.05, 0.1, 0.2]), epochs=r.choice([20, 40, 60]))
        for _ in range(6)
    ]

    generations = 5
    # Prepare CSV logging (append rows, then write at end)
    csv_rows: List[Dict[str, object]] = []
    for gen in range(generations):
        # Evaluate
        for ind in pop:
            ind.fitness = evaluate(ind, xs, ys)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        print(f"gen={gen} best_acc={best.fitness:.3f} act={best.activation} lr={best.learning_rate} epochs={best.epochs}")

        # Per-generation consolidated report + determinism signature (for logging)
        gen_report_sig = None
        gen_report_checksum = None
        try:
            from ggnes.hierarchical.module_spec import ModuleSpec, ParameterSpec, PortSpec
            from ggnes.hierarchical.derivation import DerivationEngine
            from ggnes.utils.observability import consolidated_report, determinism_signature

            # Build a minimal report using the best individual's architecture as the fingerprint source
            report_graph = build_graph(width=3, activation=best.activation)
            rep_spec = ModuleSpec(
                name="Report", version=1,
                parameters=[ParameterSpec("dim", default=1)],
                ports=[PortSpec("in", 1), PortSpec("out", 1)],
                invariants=["out.size == dim"],
            )
            rep_engine = DerivationEngine(report_graph)
            rep_root = rep_engine.expand(rep_spec, {"dim": 1})
            rep = consolidated_report(rep_engine, rep_root, report_graph, rng_manager=rng, device='cpu', dtype='float32')
            gen_report_sig = determinism_signature(rep)
            gen_report_checksum = rep.get('determinism_checksum')
        except Exception:
            pass

        # Log full population rows for this generation
        for idx, ind in enumerate(pop):
            csv_rows.append({
                "generation": gen,
                "individual_index": idx,
                "is_best": bool(ind is best),
                "accuracy": round(ind.fitness, 6),
                "activation": ind.activation,
                "learning_rate": ind.learning_rate,
                "epochs": ind.epochs,
                "gen_report_sig": gen_report_sig,
                "gen_report_checksum": gen_report_checksum,
            })

        # Reproduce: top 3 spawn mutated children to refill population
        parents = pop[:3]
        children: List[Individual] = []
        while len(children) + len(parents) < len(pop):
            p = r.choice(parents)
            children.append(mutate(p, rng))
        pop = parents + children

    # Final best evaluation
    final_best = max(pop, key=lambda x: x.fitness)
    print(f"final_best_acc={final_best.fitness:.3f} act={final_best.activation} lr={final_best.learning_rate} epochs={final_best.epochs}")

    # --- Write CSV log ---
    # The file will be created under tutorials/ by default.
    try:
        csv_path = 'tutorials/e2e_binary_classification_log.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "generation",
                    "individual_index",
                    "is_best",
                    "accuracy",
                    "activation",
                    "learning_rate",
                    "epochs",
                    "gen_report_sig",
                    "gen_report_checksum",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print("csv_log:", csv_path)
    except Exception:
        # If filesystem is read-only, skip without failing the tutorial
        pass


if __name__ == '__main__':
    main()


