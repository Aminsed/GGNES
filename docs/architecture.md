# Architecture

## System overview

GGNES composes neural architectures as typed directed graphs and evolves them through deterministic graph rewrites. The runtime is split into orthogonal subsystems:
- Core graph (nodes/edges, validation, ID/UUID)
- Generation (matching, rule engine, transactions, oscillation, network generation)
- Hierarchical derivation (ModuleSpec and DerivationEngine)
- Evolution (genotypes, operators, selection)
- Translation (PyTorch implementation and state manager)
- Observability (reports, explains, signatures) and utilities

## Core
- Graph: simple or multigraph; INPUT/HIDDEN/OUTPUT nodes with attributes such as `output_size`, `aggregation`.
- Validation: activation/aggregation admissibility, finite parameters, dimensionality, reachability; advanced aggregation parameter domains.
- Fingerprint: Weisfeiler–Lehman hashing with multiplicity awareness.
- UUID: deterministic provider derived from canonicalized inputs; cache with bounded eviction.

## Generation
- LHSPattern/RHSAction/EmbeddingLogic: describe subgraph rewrites.
- RuleEngine: applies rules inside a transaction, reconnects boundaries, validates, and commits.
- Oscillation detection and cooldown: prevents repeated cycles.
- generate_network: outer loop coordinating matching, selection, application, repair.

## Hierarchical derivation
- ModuleSpec: parameters (with expression defaults), ports, attributes, invariants.
- Parameter binding: guarded AST evaluation; deterministic binding signatures; invariant checks.
- DerivationEngine: nested transactions, UUIDs incorporate (module, version, path, signature); budget limits (depth/expansion/time) with structured errors.

## Evolution
- Genotype/CompositeGenotype: classic rule lists and composite (G1/G2/G3) schemas.
- Operators: mutate/crossover over genotypes and composite genotypes with clamped, deterministic deltas.
- Selection: PRIORITY_THEN_PROBABILITY_THEN_ORDER and NSGA‑II (stable non‑dominated sort and crowding distance ordering).

## Translation
- Per‑edge projections when source/target dimensions differ; advanced aggregations introduce learned parameters (attention, top‑k, MoE, attn_pool).
- Recurrent edges integrate with StateManager for sequence handling.
- Hierarchical submodule cache with metrics; parity verified by tests.

## Observability and determinism
- Consolidated reports (schema v2): derivation checksum, WL fingerprint, batches, env (seed/device/dtype), RNG signatures; determinism checksum.
- Validators (tolerant and strict) and migration helpers.
- Determinism signatures for CI gates over key report fields.

---

## 1. Formal Graph Semantics
Let the architecture be a directed graph \(G=(V,E)\). Each node \(v\in V\) has dimension \(d_v\in\mathbb{N}_{\ge1}\), activation \(a_v: \mathbb{R}^{d_v}\to\mathbb{R}^{d_v}\), bias \(b_v\in\mathbb{R}^{d_v}\) (implemented as broadcasting a scalar where applicable), and aggregation operator \(\alpha_v\).

For non‑INPUT node \(v\), define the enabled incoming set \(\mathcal{I}(v) = \{(u,e): e=(u\to v)\in E,\ \mathbf{1}_e=1\}\). If an edge is recurrent, the source value is drawn from the previous timestep \(x_u^{(t-1)}\) instead of \(x_u^{(t)}\).

Per‑edge projection:
\[ \tilde{x}_{u\to v} = (\pi_{u\to v} x_u)\odot w_{(u\to v)}\,, \quad \pi_{u\to v}\in\{I, P\}, \]
where identity is used when \(d_u=d^{in}_v\) and \(P\in\mathbb{R}^{d^{in}_v\times d_u}\) otherwise.

Aggregation \(\alpha_v\) produces \(z_v\) as:
- sum: \( z_v = \sum_{(u,e)\in\mathcal{I}(v)} \tilde{x}_{u\to v} \)
- mean: \( z_v = \frac{1}{|\mathcal{I}(v)|}\sum_{(u,e)\in\mathcal{I}(v)} \tilde{x}_{u\to v} \)
- max: \( z_v = \max_{(u,e)\in\mathcal{I}(v)} \tilde{x}_{u\to v} \) (elementwise)
- concat: \( z_v = \operatorname{concat}(\tilde{x}_{u\to v}) \)
- matrix\_product: stack to \(X\in\mathbb{R}^{k\times d}\), \( z_v = \operatorname{vec}(X) \)

Advanced:
- attention: \( s = \frac{X q}{\sqrt{\max(1,d_q)}\,T} \), \( p=\operatorname{softmax}(s) \), \( z_v = \sum_i p_i X_{i,:} \)
- multi‑head attention: \( z_v = \operatorname{concat}_h z^{(h)} \)
- top‑k weighted: select \(\mathcal{K}\subseteq\{1..k\}\), \( z_v = \sum_{i\in\mathcal{K}} \operatorname{softmax}(s_{\mathcal{K}})_i X_{i,:} \)
- gated sum: \( z_v = \sum (\sigma(g_{u\to v})\, \tilde{x}_{u\to v}) \)
- mixture of experts (softmax): \( z_v = \sum_i \operatorname{softmax}(r)_i X_{i,:} \)
- attention pool: \( z_v = \operatorname{concat}_h \sum_i \operatorname{softmax}(X q^{(h)})_i X_{i,:} \)

Node output: \( x_v = a_v(z_v + b_v) \).

## 2. Validation Conditions
- Activations/aggregations must be registered.
- \(d_v\ge 1\), \(b_v, w_e\in\mathbb{R}\) finite.
- Advanced parameter domains: \(\text{num\_heads}\ge1\), \(T>0\), \(\text{dropout\_p}\in[0,1)\), \(\text{top\_k}\in\mathbb{N}_{\ge1}\) and \(\text{top\_k}\le \text{fan\_in}\).
- Reachability: every OUTPUT has a path from some INPUT via enabled edges.

## 3. Weisfeiler–Lehman Fingerprint
Initialization: \(\ell_v^{(0)}=H(\tau(v), a_v, \deg^-_e(v), \deg^+_e(v))\) using enabled edges.

Iterative refinement for \(t=1..k\):
\[ \ell_v^{(t)} = H\Big( \ell_v^{(t-1)},\ \operatorname{multiset}\{(\text{in},\ell_u^{(t-1)})\}_{(u\to v)\in E_e},\ \operatorname{multiset}\{(\text{out},\ell_w^{(t-1)})\}_{(v\to w)\in E_e},\ (d_v,\alpha_v) \Big). \]

Final hash: \( H( \operatorname{sort}(\{\ell_v^{(k)}\}), |\{v: \tau(v)=\text{INPUT}\}|, |\{v: \tau(v)=\text{OUTPUT}\}| ) \).

## 4. Deterministic UUID Scheme
Canonicalize inputs by sorting keys and fixing float precision; payload \(P\) includes namespace, scheme version, entity type, optional salt, and canonical inputs. UUID bytes are \(\text{sha256}(\operatorname{json}(P))_{[:16]}\) with RFC‑4122 version/variant bits set.

## 5. RNG Context Seeding
Global seed \(S\) defines per‑context seeds \( s_c = \operatorname{int}(\text{sha256}(S\Vert c)) \bmod 2^{32} \). Crossover derives order‑independent seeds by sorting parent identifiers before hashing.

## 6. Selection and NSGA‑II
- PRIORITY_THEN_PROBABILITY_THEN_ORDER: filter by max priority; normalize probabilities; bucket by precision \(\epsilon\); sample bucket proportionally (group weight = rounded probability × group size); tie‑break by stable index.
- NSGA‑II: dominance \(x\prec y\iff f(x)\le f(y)\wedge \exists k: f(x)_k<f(y)_k\); crowding distance \(D(i)=\sum_k \frac{f(i+1)_k - f(i-1)_k}{\max f_k - \min f_k}\); select by rank then \(-D\) then index.

## 7. Consolidated Report and Signature
Report \(R\) contains: schema_version=2, derivation checksum, WL fingerprint, per‑batch metrics, env (seed/device/dtype, RNG signatures), determinism checksum. Signature \(\sigma(R)=h(\text{core fields})\) detects drift; include env for strict gating.
