# Evolution and Selection

This document formalizes genotype representations, evolutionary operators, and selection procedures used in GGNES. All randomness is mediated through RNGManager contexts to ensure reproducibility; where probabilities appear, they are computed deterministically from canonical state.

## Genotype representations

### Classic rule-list genotype

A classic genotype is an ordered list of rules:
\[
G = (r_1, r_2, \dots, r_n),\quad r_i = (\text{rule\_id}, \text{lhs}, \text{rhs}, \text{embedding}, \text{metadata})
\]

- metadata optionally contains integer priority \(\pi \in \mathbb{Z}\) and probability \(p \in (0,1]\).
- The genotype may also store \(\text{fitness} \in \mathbb{R}\) and auxiliary history.

### CompositeGenotype (G1/G2/G3)

The composite genotype is a typed product:
\[
\mathcal{C} = (\underbrace{G_1}_{\text{Grammar}},\ \underbrace{G_2}_{\text{Policy}},\ \underbrace{G_3}_{\text{Hierarchy}})
\]
- \(G_1\): list of rule dicts with keys `rule_id` (UUID string), optional `priority` (int), `probability` (float in \((0,1]\)).
- \(G_2\): numeric and boolean policy hyperparameters (e.g., learning rate, batch size, epochs, WL iterations).
- \(G_3\): module parameter maps and attribute summaries (JSON-serializable scalars).

Deterministic identity: a UUID \(U = \mathrm{UUID}(\mathrm{canon}(\mathcal{C}))\) is derived by SHA-256 of a canonical JSON serialization (sorted keys, fixed float precision), then truncated and RFC-4122 normalized.

## Mutation operators

Let \(\mathcal{R}\) denote the current rule set. Mutation type \(T\) is drawn by roulette selection from a normalized probability vector \(\mathbf{q}\):
\[
q_k = \frac{w_k}{\sum_j w_j},\quad T \sim \text{Categorical}(\mathbf{q})\,.
\]
The RNG used is the mutation context RNG derived from \(\text{genotype\_id}\), ensuring order independence across runs.

Implemented mutations (classic genotype):
- modify_metadata: if metadata exists, update
  - \(\pi' \leftarrow \pi + \delta_\pi\), \(\delta_\pi \in \{-1,+1\}\)
  - \(p' \leftarrow \mathrm{clamp}(p\cdot \alpha,\ 10^{-12},\ 1]\), \(\alpha\) sampled from a bounded factor set
- modify_rhs: pick one RHS `add_edges` entry and scale a weight \(w' \leftarrow w\cdot \alpha\)
- modify_lhs: add or toggle a benign match criterion for a randomly chosen LHS node
- add_rule: duplicate a template rule, assign a fresh \(\text{rule\_id}\), adjust metadata (priority/probability)
- delete_rule: remove a random rule if \(|\mathcal{R}| > n_{\min}\)

Composite genotype mutation `mutate_composite(\mathcal{C}, \Delta)` applies clamped deltas:
\[
\begin{aligned}
\text{lr}' &= \mathrm{clamp}(\text{lr} + \Delta_{\text{lr}},\ 10^{-8},\ 10)\\
\text{batch}' &= \max(1,\ \text{batch} + \Delta_{\text{batch}})\\
\text{epochs}' &= \max(1,\ \text{epochs} + \Delta_{\text{epochs}})\\
\text{priority}'_i &= \max\{0,\ \text{priority}_i + \Delta_{\pi,i}\}\\
\text{probability}'_i &= \min\{1, \max\{10^{-12},\ \text{probability}_i \cdot \alpha_i\}\}\,.
\end{aligned}
\]
All deltas \(\Delta\) are supplied deterministically by the caller (often derived from an RNG context).

## Crossover operators

### Uniform crossover (classic genotype)
Let \(\mathcal{I} = \{\text{rule\_id}\}\) be the union of rule identifiers from both parents. For each \(i \in \mathcal{I}\):
- If both parents contain distinct versions of \(i\), choose each offspring’s version independently with Bernoulli(\(1/2\)):
\[
\tilde{r}^{(o)}_i \sim \begin{cases}
\text{parent1 version of } i & \text{w.p. } 1/2\\
\text{parent2 version of } i & \text{w.p. } 1/2
\end{cases}
\]
- Otherwise copy the present version.
- Include \(\tilde{r}^{(o)}_i\) in offspring \(o\) with probability \(p_{\text{incl}}\) (config `crossover_probability_per_rule`).
- Enforce size constraints by padding/truncation: if \(|\mathcal{R}^{(o)}|<n_{\min}\), fill from a deterministically ordered pool; if \(|\mathcal{R}^{(o)}|>n_{\max}|\), truncate.

Determinism: the rule id pool is sorted lexicographically by string; the RNG comes from the crossover context seeded on a commutative pair key (parent ids sorted), making \((A,B)=(B,A)\).

### Composite crossover
A deterministic union/median policy:
- \(G_1\): union by `rule_id` with parent-precedence tie-break via lexicographic parent signatures
- \(G_2\): combine by component wise min/mean/min (per field policy)
- \(G_3\): union with A-first precedence (or configurable)

## Selection procedures

### Priority→Probability→Order (rule match selection)
Let \(\mathcal{M} = [(r_m, b_m)]\) be matches with metadata priority \(\pi_m\) and probability weights \(w_m\in (0,1]\). The procedure:
1) Filter to maximal priority \(\Pi = \max_m \pi_m\), keep \(\mathcal{M}'=\{m: \pi_m = \Pi\}\).
2) Normalize weights \(p_m = w_m/\sum_{j\in \mathcal{M}'} w_j\). Apply grouping precision \(\epsilon\) by rounding \(\hat{p}_m = \epsilon \cdot \mathrm{round}(p_m/\epsilon)\).
3) Partition \(\mathcal{M}'\) into groups by \(\hat{p}\) descending. Choose a group by sampling proportionally to \(\hat{p} \cdot |\text{group}|\). Within the selected group, break ties by original genotype order (stable index).

This yields a categorical choice with deterministic grouping and tie-breaking. All random draws use RNGManager context `selection`.

#### Implementation clarification (group-weight sampling)
Let groups be indexed by \(g\) with rounded mass \(\hat{p}_g\) and size \(|G_g|\). Define
\[
W = \sum_g \hat{p}_g\,|G_g|\,.
\]
The implementation accumulates \(\hat{p}_g\,|G_g|\) against a uniform \(r\sim \mathrm{U}[0,1)\) and selects the first group where the running sum exceeds \(r\), with a deterministic fallback to the last group. This is equivalent to sampling a group with probability \(\frac{\hat{p}_g\,|G_g|}{W}\) if one interprets the comparison as using a scaled threshold \(r' = r\,W\). Rounding can make \(W\) slightly deviate from 1; the fallback ensures total mass coverage. Within-group choice uses stable index order, preserving determinism.

### NSGA‑II (multi-objective selection)
We assume \(M\) minimization objectives and a population of size \(N\) with objective vectors \(\mathbf{f}_i \in \mathbb{R}^M\).

Dominance relation: \(i\) dominates \(j\) (\(i \prec j\)) iff
\[
\forall k: f_{i,k} \le f_{j,k} \quad\text{and}\quad \exists k: f_{i,k} < f_{j,k}\,.
\]

Fast non-dominated sort (Deb et al.): computes fronts \(F_1, F_2, \dots\) where \(F_1\) are non-dominated, \(F_2\) are dominated only by \(F_1\), etc. Determinism is enforced by processing indices in ascending order and sorting each front.

Crowding distance within a front \(F\): for objective \(k\), sort indices \(i \in F\) by \(f_{i,k}\). Set boundary distances to \(+\infty\), and accumulate normalized gaps for interior points:
\[
\mathrm{dist}(i) \mathrel{+}= \frac{f_{i+1,k} - f_{i-1,k}}{\max_j f_{j,k} - \min_j f_{j,k}}\,.
\]
If \(\max=\min\) for an objective, skip that term (all equal). The final distance is the sum over objectives. Deterministic tie-breaking uses the stable index order.

Selection of \(K\) individuals:
1) Concatenate fronts in order until the next front would exceed \(K\).
2) For the last partial front, sort by decreasing crowding distance, then by ascending index; take the top remaining to reach \(K\).

Complexity: \(\mathcal{O}(MN^2)\) for the standard fast sort, with small constants for typical \(M\) (e.g., 2–3).

## Fitness and evaluation

GGNES does not prescribe a fitness function; typical choices:
- Accuracy/error rates on validation splits (maximize accuracy / minimize error)
- Multi-objective: (error, parameters, latency), all to be minimized for NSGA‑II
- Repair penalty incorporation: overall \(\tilde{f} = f + \lambda \cdot \mathrm{penalty}(\text{repair})\) for minimization

A sample repair penalty used in tests:
\[
\mathrm{penalty}(\text{impact}) = \begin{cases}
0 & \text{if } \text{impact}=0\\
0.05\cdot \text{impact} & 0<\text{impact}\le 0.1\\
0.005 + 0.1\cdot(\text{impact}-0.1) & 0.1<\text{impact}\le 0.5\\
0.045 + 0.2\cdot(\text{impact}-0.5) & \text{otherwise}
\end{cases}
\]

## Determinism and order-independence

- All categorical draws (mutation type, crossover inclusion, selection) are performed with RNGManager context RNGs whose seeds are derived from stable, order-independent keys.
- Parent order in uniform crossover is neutralized by sorting parent identifiers before deriving the RNG seed.
- Rule pools are ordered deterministically by stringified `rule_id` when padding to \(n_{\min}\).

## Pseudocode references

Selection (priority→probability→order):
```python
candidates = [m for m in matches if m.priority == max_priority]
weights = normalize([m.prob for m in candidates])
rounded = [round(w / eps) * eps for w in weights]
# group by rounded, pick group proportional to rounded * group_size
# break ties by stable index
```

NSGA‑II (selection to K):
```python
fronts = fast_non_dominated_sort(objs)
selected = []
for front in fronts:
  if len(selected) + len(front) <= K:
    selected.extend(front)
  else:
    dist = crowding_distance(front, objs)
    ordered = sorted(front, key=lambda i: (-dist[i], i))
    selected.extend(ordered[: K - len(selected)])
    break
return [population[i] for i in selected]
```

## Islands and migration (optional)

GGNES supports island-style evolution via an island scheduler (see `ggnes/evolution/islands.py`). A migration step moves individuals between islands according to a configurable topology (e.g., ring) and migration size \(m\). The observability layer emits an `island_migration_report`:

- topology: string key (e.g., "ring"); migration_size \(m\in\mathbb{N}\); island count \(n\).
- per_island: for each island \(i\), sizes before/after and estimated moved_in/out counts.
- migration_checksum: \(\mathrm{SHA256}([\text{before}, \text{after}])_{\text{short}}\).
- rng_namespace_sig_island0: short signature of island 0 RNG state for reproducibility.

Deterministic island RNG contexts are obtained via `scheduler.get_island_rng(i)`. Migration traces can be correlated with consolidated reports to ensure end-to-end reproducibility.

## Notes on constraints and clamping

- Probabilities are kept in \((0,1]\) to avoid degenerate zero-probability rules; a floor of \(10^{-12}\) is used.
- Integer counts \(\ge 1\) (e.g., batch size, epochs) are clamped at the lower bound.
- Learning rates are bounded to a conservative domain \([10^{-8}, 10]\) to avoid overflow/underflow in downstream backends.

## References
- Deb, K. et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE Trans. Evol. Computation, 6(2), 2002.
- Standard roulette selection and uniform crossover operators from classical GA literature, adapted with deterministic seeding and stable ordering.
