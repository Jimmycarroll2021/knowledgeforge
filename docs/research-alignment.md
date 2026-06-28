# Research Alignment

Every layer of KnowledgeForge maps to a primary source, and — critically — *implements what that
source specifies* rather than wrapping it. Where an earlier version of this document overclaimed
(a parameter-free "GraphSAGE", an unmeasured resolver), those claims are corrected below and the
code is now the source of truth.

---

## Layer → Paper mapping

| Layer | Paper | Key concept actually implemented |
|-------|-------|----------------------------------|
| Triple Primitive | W3C RDF 1.1 Concepts (2014) | Subject → Predicate → Object graph model |
| Provenance | W3C PROV-O (2013) | `source` (producing agent), `lineage` (`wasDerivedFrom`), `valid_from`/`valid_to`, `schema_version` |
| Validation | W3C SHACL (2017) | `PropertyShape`: cardinality, target class, datatype, `sh:severity` ladder |
| Entity Resolution | Winkler 1990 (Jaro-Winkler); Union-Find WCC | metaphone blocking, jaro+cosine, 0.90 semantic floor, transitive Union-Find — **measured F1 = 1.00** |
| Embedding Pipeline | Hamilton et al. 2017 — GraphSAGE | **Learned** unsupervised aggregator with graph loss + negative sampling (Algorithm 1) |
| Relation Embeddings | Bordes et al. 2013 — TransE | `h + r ≈ t` — cited as theoretical grounding for the triple unit (scorer not yet trained — see roadmap) |
| Similarity Search | Google TurboQuant (2024) | 4-bit SIMD quantisation for approximate nearest-neighbour |
| Communities + GraphRAG | Edge et al. 2024 — arXiv:2404.16130 | hierarchical Leiden communities + LLM summaries; local k-hop + global rank-top-K-then-synthesise (single LLM call; not yet map-reduce) |
| GNN Foundation | Battaglia et al. 2018 — Relational Inductive Biases | Graph networks as the right inductive bias for relational data |
| Graph Convolution | Kipf & Welling 2016 — GCN | Spectral graph convolutions (foundation for GraphSAGE) |

---

## Triple Primitive — W3C RDF 1.1

Every fact is stored as `(subject, predicate, object)` — the canonical RDF graph model. The `Triple`
dataclass (`contracts.py`) carries the full provenance and validity surface:

```
subject → predicate → object
+ source_doc, extraction_method, confidence, timestamp, layer, adapter   (core)
+ source, lineage, valid_from, valid_to, schema_version                  (PROV-O)
+ object_is_literal, object_datatype                                     (RDF literal vs entity)
```

The `triple_id` hashes `(subject, predicate, object, source_doc, layer)`, so the **layer is part of
identity** — the same `(s,p,o)` fact can coexist as a source fact and as an inferred relation without
collision.

**Source:** W3C RDF 1.1 Concepts and Abstract Syntax, February 2014.

---

## Provenance — W3C PROV-O

Each triple distinguishes:

- `source` — the producing **agent/process** (PROV-O `wasAttributedTo` / `wasGeneratedBy`), distinct
  from `source_doc` (the file path the fact came from). Resolver-derived `SAME_AS` edges carry
  `source=EntityResolver`.
- `lineage` — parent triple ids (PROV-O `wasDerivedFrom`); empty for raw source facts.
- `valid_from` / `valid_to` — bitemporal validity window (ISO-8601; `valid_to=None` means still valid).
- `schema_version` — the contract version the triple was minted under.

Provenance is **non-destructive**: entity resolution writes alias rows and `SAME_AS` edges; it never
deletes an original entity.

**Source:** W3C PROV-O: The PROV Ontology, April 2013.

---

## Validation — SHACL-style PropertyShape

`AdapterSchema` may carry `PropertyShape` constraints (`contracts.py`):

- `allowed_target_kinds` — target-class constraint (`sh:class`)
- `min_count` / `max_count` — per-subject cardinality (`sh:minCount` / `sh:maxCount`)
- `datatype` — required xsd datatype for literal objects (`sh:datatype`)
- `severity` — `Violation | Warning | Info` (`sh:severity`)

`ForgePipeline._validate` enforces a severity ladder: in **soft** mode (default) every triple is
admitted and findings are recorded as messages; in **strict** mode (`run(..., strict=True)`)
triples carrying any `Violation` are excluded and counted as rejected, while `Warning`/`Info` are
always admit-and-flag. This is real shape validation, not predicate-set membership.

**Source:** W3C Shapes Constraint Language (SHACL), July 2017.

---

## Entity Resolution — measured, multi-signal pipeline

> Correction: earlier docs marked resolution "complete" with **no measurement**. It is now measured.

The resolver (`resolution/resolver.py`) is evaluated against a hand-labelled benchmark
(`tests/fixtures/er_labelled_pairs.json`: 25 true surface variants + 24 confusable negatives such as
GraphSAGE/GraphSAINT, TransE/TransR, GCN/R-GCN):

- **F1 = 1.00 on the benchmark** (`tests/test_resolution_eval.py`, gate ≥ 0.85 — red by design on regression)
- **Generalisation:** not yet measured in CI — only the 49-pair benchmark is gated (F1 = 1.00, gate ≥ 0.85). A held-out generalisation set is roadmap.

Pipeline:

1. **Phase 1 — exact normalised match** within a kind (case / punctuation / spacing only).
2. **Phase 2 — blocked similarity.** Candidate pairs come from metaphone + prefix blocking (not a
   pure type-quadratic scan). An initialism rule (`GNN` ⇄ `Graph Neural Network`) auto-merges
   deterministically. Otherwise a non-identical pair must clear a combined
   `jaro·0.6 + cosine·0.4` score **and** a **0.90 cosine semantic floor** — the semantic floor is
   what separates near-string confusables (high string overlap, divergent meaning) from true
   variants. Scores in the flag band `[0.70, 0.85)` are recorded, never auto-merged.
3. **Phase 3 — structural WCC** on `SIMILAR_TO` edges.

All confirmed matches feed one path-compressed **Union-Find**, so transitive duplicates
(a → b → c) collapse to a single canonical root. Each merge writes an alias row and a `SAME_AS`
edge to the `inferred_relations` layer.

**Source:** Winkler (1990), Jaro-Winkler string similarity; classical Union-Find weakly-connected components.

---

## Embedding Pipeline — learned GraphSAGE aggregator

> Correction: an earlier version of this document described a fixed, parameter-free average
> (`h_v = normalise((h_self + mean(h_neighbours)) / 2)`). That did **not** implement Hamilton 2017,
> which learns an aggregation function. The aggregator is now a genuine learned layer.

Hamilton et al. 2017 introduced inductive node embedding by **learning an aggregation function**
over neighbourhoods. KnowledgeForge (`embeddings/pipeline.py`) implements this:

```
z_v = L2normalize( ReLU( W · CONCAT(h_v_self, MEAN({h_u : u ∈ N(v)})) ) )
```

with a single weight matrix `W` of shape `(out_dim, 2·in_dim)`. `W` is trained **unsupervised**
(there are no labels in a knowledge graph) on GraphSAGE's graph-based loss:

```
J_G(z_u) = -log σ(z_u · z_v) - Q · E_{v_n ~ P_n}[ log σ(-z_u · z_{v_n}) ]
```

- **Positives** `(u, v)` are drawn from short random walks over the *semantic* neighbour graph
  (structural-noise predicates like `HAS_FILE_TYPE`/`CONTAINS_KEY`/`HAS_TAG` are excluded).
- `Q` **negatives** are sampled uniformly per positive.
- The forward pass and back-propagation through `Linear → ReLU → L2` are **hand-derived in numpy**
  and optimised with plain SGD; all randomness is seeded for determinism.
- The aggregator is applied for `K=2` layers (sharing `W`) so structure propagates up to 2 hops.
- The trained `W` is persisted (`graphsage_w.npy`) and reused for reproducibility.

Base vectors come from `sentence-transformers/all-MiniLM-L6-v2` (384-dim), so the final embedding
blends semantic meaning with learned structural position.

On the 73-doc research vault, all 511 entities embed in ~25s on a single machine.

**Source:** Hamilton, Ying & Leskovec (2017). Inductive Representation Learning on Large Graphs. NeurIPS 2017.

---

## Relation Embeddings — TransE

Bordes et al. 2013 established the translation model: for a valid triple `(h, r, t)`, the relation
acts as a translation, `h + r ≈ t`. KnowledgeForge cites this as the **theoretical grounding** for
why the triple is the right unit of storage — it is the minimal structure that supports both symbolic
reasoning and geometric embedding.

**Honest status:** a *trained* TransE scorer (relation as a first-class learned vector) is **not yet
implemented** — it is on the roadmap. This section is grounding, not a claim of a shipped scorer.

**Source:** Bordes, Usunier, Garcia-Duran, Weston & Yakhnenko (2013). Translating Embeddings for Modeling Multi-relational Data. NeurIPS 2013.

---

## Communities + GraphRAG — local and global (Edge et al. 2024)

Edge et al. 2024 showed that grounding answers in graph structure (and, for corpus-wide questions,
in *community summaries*) outperforms RAG over raw vector chunks. KnowledgeForge implements **both**
modes.

**Community detection** (`community/detector.py`): build an undirected graph from semantic triples
(weighting `SIMILAR_TO`/`SAME_AS`/`RELATED_TO` edges higher), run **Louvain**
(`networkx.algorithms.community.louvain_communities`, seeded), filter to a minimum community size,
then generate and cache an LLM summary per community. On the vault this yields 16 summarised
communities. *(Leiden + hierarchical communities are deferred — see roadmap; Louvain approximates
Leiden quality for this scale.)*

**Local mode** (`GraphRAG.ask(mode="local")`):
1. **Anchor extraction** — semantic anchors (embedding search over the question, cosine floor 0.3)
   UNIONed with substring-`LIKE` fallback, then mapped through the resolver's `canonical()`.
2. **k-hop BFS expansion** — subgraph of up to 500 triples, confined to **trusted layers**
   (`source_facts`, `normalised_triples`, `inferred_relations`) — `llm_hypotheses` is excluded so a
   speculative LLM-authored triple can never back a "grounded" answer.
3. **Serialisation** — facts ordered by confidence so the strongest survive truncation.
4. **Grounded LLM call** — system prompt instructs the model to answer ONLY from provided facts and
   to say "The graph does not contain evidence for this." otherwise.
5. **Return** — answer + cited triples with provenance.

**Global mode** (`GraphRAG.ask(mode="global")`):
1. Load cached community summaries.
2. **Map step** — rank summaries by cosine similarity of the question vs each summary (or keyword
   overlap fallback), take the top-K.
3. **Reduce step** — synthesise an answer across the top community summaries, citing themes.

Both modes are reachable over HTTP via the `mode` field on `POST /query`.

**Source:** Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt & Larson (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.

---

## GNN Foundation — Battaglia 2018

Battaglia et al. 2018 established the case for graph networks as the right inductive bias for
relational data: systems that compose entities and relations generalise better. KnowledgeForge
reflects this — the triple is the unit of storage, the adapter protocol is the unit of domain
generalisation, and the learned GraphSAGE aggregator is the unit of structural learning.

**Source:** Battaglia et al. (2018). Relational Inductive Biases, Deep Learning, and Graph Networks. arXiv:1806.01261.

---

## Scope and deferred work

KnowledgeForge is **single-machine, hundreds-to-low-thousands of entities**. The following are
honestly **not yet built**:

- Neo4j / GDS backend tier for millions of entities (the per-build N×N GraphSAGE training matrix and
  O(n²) candidate blocking are sized for this scale, not for millions).
- Leiden + hierarchical communities (current: flat Louvain).
- A trained TransE relation-as-first-class-vector scorer.
- An inductive new-node embedding path (embed an unseen node without a full rebuild).
- A pluggable adapter registry.

---

## What this is NOT

- Not a wrapper around LangChain or LlamaIndex.
- Not pure vector search (RAG over chunks).
- Not a notebook demo.

The stack — triple extraction, SHACL validation, measured entity resolution, **learned** GraphSAGE
aggregation, local + global GraphRAG — implements what the primary literature specifies, at
single-machine scale, with no claim it does not back in code.
