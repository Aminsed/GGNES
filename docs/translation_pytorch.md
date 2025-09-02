# Translation to PyTorch

This document describes the exact behavior of `ggnes/translation/pytorch_impl.py` and the shapes/parameters it constructs. All statements below are consistent with the current implementation.

## Build and configuration
- Entry point: `to_pytorch_model(graph, config)` returns an `nn.Module` subclass (`TranslatedModel`).
- Device/dtype: from `config['device']` (default `'cpu'`) and `config['dtype']` (default `torch.float32`). The module is moved to that device/dtype.
- Cycle handling and order: the translator calls `graph.detect_cycles()` and uses `graph.topological_sort(ignore_recurrent=True)` to compute execution order, ignoring edges marked recurrent.

## Input validation
- Required input width is the sum of `output_size` over all INPUT nodes in `graph.input_node_ids`.
- At `forward(x, reset_states=False)`, if `x.shape[1]` does not equal this sum, a `ValueError` is raised.
- If there are multiple INPUT nodes, the input tensor is split along the feature dimension in the same order as `graph.input_node_ids`.

## Parameters and layers constructed
- Per-node activations: for each non-INPUT node, a layer is created based on `activation_function`. Supported: `relu`, `sigmoid`, `tanh`, `softmax(dim=1)`, `leaky_relu`, `elu`, `gelu`, `lstm` (LSTMCell), `gru` (GRUCell), and `linear` (Identity).
- Per-node bias: each non-INPUT node registers a parameter `bias_{node_id}`. If the node has `attributes['derivation_uuid']` and the translation cache is enabled, the bias parameter is cached/reused under that UUID. Cache metrics `hits/misses/entries` are updated accordingly.
- Per-edge weights: for every enabled edge into a non-INPUT target, a parameter is registered. Naming:
  - Simple graph: `weight_{source_id}_{target_id}`
  - Multigraph: `weight_{edge.edge_id}`
- Projections: if `source_size != target_input_size` for an edge, a linear projection is created once per (source,target) in simple graphs, or once per `edge_id` in multigraphs:
  - Simple graph: `proj_{source_id}_{target_id}`
  - Multigraph: `proj_{edge.edge_id}`
- Post-aggregation projection: if the aggregated input size for a node differs from the node’s `output_size`, a linear layer `post_{node_id}` is created. When the node has `derivation_uuid` and cache is enabled, this layer may be reused from cache if shapes match; metrics are updated.

## Aggregation semantics
Let `inputs` be the list of incoming (weighted and optionally projected) tensors for a node in the current timestep. The aggregation is determined by `node.attributes['aggregation']` (default `'sum'`). Implemented:
- `sum`: \(\sum_j x_j\)
- `mean`: \(\frac{1}{J}\sum_j x_j\)
- `max`: elementwise max over stacked inputs
- `concat`: concatenation along feature dimension
- `matrix_product`: if multiple inputs, stacks to shape `[B, J, D]`, flattens to `[B, J\cdot D]`; otherwise passes through the lone input
- `attention`: single-head additive: uses a learnable query `q_{node_id}` of shape `[Dq]`; computes \(\mathrm{softmax}((X q)/ (\sqrt{D_q}\cdot \text{temperature}))\) then weighted sum
- `multi_head_attention`: `q_{node_id}` has shape `[H, Dq]`. For each head \(h\), compute softmax weights and sum, then concatenate heads → output size `H * Dq`
- `topk_weighted_sum`: scores are mean over features; if `top_k` is set, selects top-k indices per sample and does a softmax-weighted sum over the selected; otherwise softmax over all
- `moe` (mixture-of-experts): collects router scalars per incoming edge; softmax or top-k gating over routers to weight stacked inputs and sum
- `attn_pool`: like attention pooling with `q_{node_id}` of shape `[H, D]`; computes per-head scores and concatenates heads
- `gated_sum`: applies `sigmoid(gate)` per incoming edge and multiplies each weighted input by that gate before summing

Notes on shapes and helper parameters:
- For attention-like ops, `head_dim` defaults to the node’s input size if not provided. `num_heads`/`pool_heads` default to 1. Temperature and epsilon parameters are taken from attributes if present (with validation enforced at graph level).
- The aggregated size that feeds the activation/bias is computed per aggregation; for `concat` and `matrix_product` it can differ from a node’s `output_size`, in which case `post_{node_id}` rescales.

## Recurrent edges and state
- Edges with `attributes['is_recurrent'] == True` consume the previous timestep’s output of the source node, managed via `StateManager`.
- `reset_states=True` in forward resets all internal states before processing the batch.
- After computing outputs for all nodes in topological order for a timestep, the translator stores outputs in `StateManager` and uses them as previous outputs for recurrent edges in subsequent steps.

## Multigraph behavior
- Adjacency: when `graph.config['multigraph']` is `True`, incoming and outgoing edge maps contain lists; the translator expands them to `(target_id, edge)` pairs in stable order (sorted by `edge_id` when needed).
- Parameter keys: per-edge weights and gates/routers are keyed by `edge_id` to disambiguate parallel edges. Projections likewise use `proj_{edge_id}`.

## Cache parity and metrics
- Global cache is keyed by `derivation_uuid` and stores per-node bias parameters and `post_{node_id}` linear layers when enabled.
- Toggling the cache must not change outputs; metrics (`hits/misses/entries`) are for observability only.
- Control flags:
  - `set_translation_cache_enabled(True|False)` (global)
  - Per-model override via `config['enable_translation_cache']`
  - `clear_translation_cache()` and `get_translation_cache_metrics()` for tests and demos

## Determinism considerations
- Parameter initialization for translator-created tensors (e.g., attention queries, linear layers) uses PyTorch defaults and therefore depends on the global PyTorch RNG state. Deterministic end-to-end experiments should set `torch.manual_seed(...)` before model construction and training.
- Graph-side randomness is never used during translation; graph structure and attributes fully determine translator construction.

## Error handling summary
- Unknown activation raises `ValueError` at construction.
- Forward raises `ValueError` if input width mismatches expected sum from INPUT nodes.
- All other dimensional adjustments are handled via projections and post-aggregation layers.
