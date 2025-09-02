## Determinism

This document specifies the determinism guarantees and constructions used throughout GGNES. All definitions are made precise with unambiguous hashing and integer conversions.

### Notation
- Let SHA256(x) denote the 32‑byte output of the SHA‑256 hash on bytes x.
- Let hex16(x) denote the first 16 hexadecimal characters of the SHA‑256 hexdigest (8 bytes, 64 bits in hex) of string x, i.e., hex16(x) = hexdigest(SHA256(utf8(x)))[:16].
- Let int64_be(b) be the unsigned 64‑bit integer formed from the first 8 bytes of b in big‑endian order.
- All JSON serializations are performed with sorted keys and compact separators (",", ":").

### RNG contexts
GGNES uses a single global seed S ∈ [0, 2^32−1] and maintains a map of named RNG contexts. Two mechanisms are used, both deterministic under S:
- Pre‑allocated contexts at initialization: {selection, mutation, crossover, repair, application}. These are seeded by a base RNG Random(S) via independent draws base_rng.randint(0, 2^32−1). This yields reproducible but not cryptographically hashed seeds for these well‑known names.
- On‑demand contexts: For any context name c not present, derive a 32‑bit seed s_c by hashing the tuple (S, c):
\[
  s_c = \big(\text{int64\_be}(\text{SHA256}(\text{utf8}(\text{f"{S}:{c}"}))[:8])\big) \bmod 2^{32}.
\]
The RNG for context c is Random(s_c). This ensures stable, order‑independent derivation for ad‑hoc contexts (e.g., hierarchical names).

Order‑independent crossover RNG. For parents with identifiers a and b (cast to strings), sort lexicographically so that \( (p_1, p_2) = (\min(a,b), \max(a,b)) \). Define
\[
  s_{\text{xo}} = \big(\text{int64\_be}(\text{SHA256}(\text{utf8}(\text{"crossover:"}+p_1+":"+p_2))[:8])\big) \bmod 2^{32}.
\]
Implementation detail: we temporarily reseed the crossover context RNG to s_xo, draw one 32‑bit sample r ∈ [0,2^32−1], restore the prior context state, and return Random(r) as the per‑crossover RNG. This yields identical RNG streams for (a,b) and (b,a) while keeping the crossover context’s long‑term state unchanged.

Hierarchical derivation contexts. During DerivationEngine.expand for a module with name m, version v and path indices π (e.g., root π=∅, child paths like (0,1,...)), a hierarchical context name is constructed:
\[
  c_\text{expand} = \text{"expand:module:"}+m+"@"+\text{str}(v)+":path:"+\begin{cases}
  \text{"root"} & \text{if } \pi=\emptyset,\\
  \text{".".join(map(str,}\pi\text{))} & \text{otherwise.}
  \end{cases}
\]
A single RNG step from this context advances the state deterministically.

### Deterministic UUIDs
A DeterministicUUIDProvider creates UUIDs by hashing a canonical JSON payload and then conforming to RFC‑4122 variant and version fields.

Canonicalization K with float precision p (default p=12) is defined recursively over Python values:
- K(None, p) = null; K(str, p) = str; K(bool, p) = bool; K(int, p) = int
- K(float x, p) = string with fixed decimal precision p (no scientific notation)
- K(list L, p) = [K(L[i], p)] in order
- K(tuple T, p) = [K(T[i], p)] (normalized to list)
- K(dict D, p) = {key: K(D[key], p)} with keys in sorted order
- K(UUID u, p) = str(u)
- Any other simple types are stringified deterministically.

Given entity_type e and input mapping I, the canonical payload is the JSON string
\[
  P = \text{json\_dump}({\text{"namespace"}: N, \text{"scheme\_version"}: k, \text{"entity\_type"}: e, \text{"salt"}: \sigma, \text{"inputs"}: K(I,p)}),
\]
with keys sorted and compact separators.

UUID byte construction:
- Compute D = SHA256(utf8(P)).
- Take the first 16 bytes U = D[:16].
- Conform to version and variant bits to produce a UUID v4 with deterministic bytes:
  - U[6] = (U[6] & 0x0F) | (4 << 4)   (version 4)
  - U[8] = (U[8] & 0x3F) | 0x80       (variant 10xx)
- The resulting 16‑byte U is emitted as a UUID.

Cache and metrics. The provider memoizes {P → UUID} with bounded capacity; hits/misses are tracked deterministically and eviction is FIFO over insertion order.

### Weisfeiler–Lehman (WL) graph fingerprint
Let G=(V,E) be the directed graph with enabled‑edge view. Initial labels L₀(v) are tuples
\[
  L_0(v) = \big(\text{node\_type}(v), \text{activation}(v), \deg^-_{\text{en}}(v), \deg^+_{\text{en}}(v)\big),
\]
stringified deterministically. For k ≥ 0, define neighbor summaries using enabled edges, respecting multiplicity in multigraph mode.

One iteration updates each node label via
\[
  L_{k+1}(v) = H\Big( L_k(v),\ \text{sorted\_multiset}\big(\{(\text{dir}, L_k(u))\}\big),\ \text{node\_dim}(v) \Big),
\]
where H is SHA‑256 of the deterministic stringification of the triple, and node_dim(v) captures (output_size, aggregation) attributes. After K iterations (default K=3 or configured as wl_iterations), the fingerprint is
\[
  F(G) = \text{SHA256}\Big( \text{utf8}\big( \text{str}\big(\text{sorted}([L_K(v): v\in V]),\ |\text{INPUT}|,\ |\text{OUTPUT}|\big) \big) \Big).
\]
This fingerprint is stable under relabelings that preserve the WL refinement structure and under the enabled‑edge view.

### Consolidated reports and signatures
A consolidated report R is a mapping with at least the following fields (schema v2):
- schema_version = 2
- derivation_checksum ∈ Hex (hex16‑style digest)
- wl_fingerprint ∈ Hex (full SHA‑256 hex)
- batches: list of per‑batch derivation metrics
- env: {seed, device, dtype, rng_signatures?}
- determinism_checksum ∈ Hex (hex16‑style digest over the report mapping)

Report checksum. Implementation uses
\[
  C(R) = \text{hex16}(\text{str}(R)),
\]
where str(R) is the deterministic Python string of the (JSON‑serializable) mapping; this is pragmatic and stable across GGNES versions.

Determinism signature. For comparison across runs we define a stable signature over a reduced base B(R):
\[
  B(R) = \{\text{derivation\_checksum}:R,\ \text{wl\_fingerprint}:R,\ \text{batches}:R\} \cup \begin{cases}
  \{\text{env}:\{\text{seed},\text{device},\text{dtype}\}\} & \text{if include\_env},\\
  \emptyset & \text{otherwise.}
  \end{cases}
\]
The signature is then
\[
  S(R) = \text{hex16}(\text{str}(B(R))).
\]
GGNES exposes assert_determinism_equivalence which computes S(R_i) across inputs and raises if any S(R_i) differ.

RNG state signatures. For a context RNG, signature σ_c is
\[
  \sigma_c = \text{hex16}(\text{str}(\text{rng.getstate()})).
\]
These are included in R.env.rng_signatures for diagnostic parity.

### Determinism properties
- Context independence: Pre‑allocated contexts are independent via distinct draws from Random(S); on‑demand contexts are independent via SHA‑256 derivation; no state is shared unless explicitly composed.
- Order‑independent crossover: Swapping parent order leaves s_xo unchanged by construction.
- UUID stability: If canonicalized inputs K(I,p) are unchanged (including float precision and sorting), the UUID is identical; strict/freeze policies reject non‑finite floats or post‑UUID mutation.
- WL stability and sensitivity: F(G) is invariant to graph isomorphisms that preserve WL refinement; it is sensitive to enabled‑edge modifications and attribute changes that affect the labeling.
- Report comparability: S(R) remains stable under addition of extra fields outside B(R); strict validators ensure presence and types of required fields.

### Recurrent state determinism
StateManager initializes hidden states deterministically on reset and updates them using forward order. For any fixed input sequence X and fixed graph G, repeated runs with identical seeds and device/dtype yield identical sequences of outputs when the cache is toggled on/off.

### Cache parity
Let f_cache and f_nocache be the translated models with cache enabled/disabled. GGNES enforces
\[
  \forall x\in\mathbb{R}^{N\times d}:\quad f_{\text{cache}}(x) = f_{\text{nocache}}(x)
\]
exactly for the same graph parameters (up to standard floating‑point arithmetic). In tests and demos we check
\[
  \|f_{\text{cache}}(x) - f_{\text{nocache}}(x)\|_\infty \le \varepsilon,\quad \varepsilon = 10^{-6}.
\]

### Practical reproducibility recipe
1) Fix seeds: RNGManager(seed = S).
2) Capture a consolidated report (v2) and record S(R), C(R), WL fingerprint, and env dtype/device.
3) Record UUIDs for composite genotypes and module binding signatures for derivations.
4) In CI, compute S(R) on a golden run and fail if S(R′) ≠ S(R) under fixed S and config.

### Appendix: Worked examples
- Context seed derivation: with S=123 and c="selection", pre‑allocated context is seeded from a draw of Random(123); for a custom context c*, s_{c*} = int64_be(SHA256(b"123:c*")[:8]) mod 2^32.
- Deterministic UUID: given inputs I, precision p=12, namespace N and scheme_version k, compute P then U = SHA256(utf8(P))[:16]; set version/variant bits; output UUID.
- Determinism signature: form B(R) with or without env, then S(R) = hex16(str(B(R))).
