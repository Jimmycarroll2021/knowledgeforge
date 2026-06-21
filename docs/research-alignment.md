# Research Alignment

Every layer of KnowledgeForge maps to a primary source. No layer is speculative — each design decision is grounded in published work that lives in the KG vault.

---

## Layer → Paper mapping

| Layer | Paper | Key concept used |
|-------|-------|-----------------|
| Triple Primitive | W3C RDF 1.1 Concepts (2014) | Subject → Predicate → Object graph model |
| Provenance | W3C PROV-O (2013) | `wasGeneratedBy`, `wasDerivedFrom`, `wasAttributedTo` |
| Entity Resolution | Vault concept notes (blocking → merge pipeline) | Blocking keys, Jaro-Winkler similarity, Union-Find WCC |
| Embedding Pipeline | Hamilton et al. 2017 — GraphSAGE | `h_v = MEAN(h_v ∪ {h_u ∀u ∈ N(v)})` — inductive neighbourhood aggregation |
| Relation Embeddings | Bordes et al. 2013 — TransE | `h + r ≈ t` — translation-based KG embedding |
| Similarity Search | Google TurboQuant (2024) | 4-bit SIMD quantisation for approximate nearest-neighbour |
| GraphRAG | Edge et al. 2024 — arXiv:2404.16130 | Graph-aware retrieval: entities → k-hop subgraph → grounded LLM answer |
| GNN Foundation | Battaglia et al. 2018 — Relational Inductive Biases | Graph networks as the right inductive bias for relational data |
| Graph Convolution | Kipf & Welling 2016 — GCN | Spectral graph convolutions (foundation for GraphSAGE) |

---

## Triple Primitive — W3C RDF 1.1

Every fact in KnowledgeForge is stored as `(subject, predicate, object)` — the canonical RDF graph model. The `Triple` dataclass carries all provenance fields required by PROV-O:

```
subject → predicate → object
+ source_doc, extraction_method, confidence, timestamp, layer, adapter
```

The `layer` field separates:
- `source_facts` — extracted directly from source documents
- `normalised_triples` — after entity resolution
- `inferred_relations` — system-generated (SIMILAR_TO, SAME_AS)
- `llm_hypotheses` — LLM-proposed, lower confidence

This separation is the core of trustworthy KG construction: never mix source evidence with inference.

**Source:** W3C RDF 1.1 Concepts and Abstract Syntax, February 2014.

---

## Entity Resolution — 3-phase pipeline

The resolver implements the canonical entity resolution pipeline:

1. **Phase 1 — Exact normalised match**: lowercase + strip punctuation. Same type only (blocks cross-type false positives).
2. **Phase 2 — Jaro-Winkler ≥ 0.85**: string similarity within type-based blocks. Threshold from vault concept notes.
3. **Phase 3 — Structural WCC**: union-find on `SIMILAR_TO` edges. Entities in the same weakly-connected component are merged.

Aliases written to `entity_aliases` table. Originals preserved — provenance is never destroyed.

**Source:** Vault concept notes on entity resolution; Jaro-Winkler as per Winkler 1990.

---

## Embedding Pipeline — GraphSAGE MEAN aggregation

Hamilton et al. 2017 introduced inductive node embedding: instead of learning a fixed embedding per node, learn an *aggregation function* over neighbourhoods. This generalises to unseen nodes without retraining.

KnowledgeForge implements the MEAN aggregator (the simplest and most robust variant):

```
h_v = normalise( (h_self + mean(h_neighbours)) / 2 )
```

Embeddings are initialised from `sentence-transformers/all-MiniLM-L6-v2` (384-dim, semantic) then aggregated over 1-hop graph neighbours from the SQLite triple store. This blends semantic meaning with structural position.

**Source:** Hamilton, Ying & Leskovec (2017). Inductive Representation Learning on Large Graphs. NeurIPS 2017.

---

## Relation Embeddings — TransE

Bordes et al. 2013 established the translation-based embedding model: for a valid triple `(h, r, t)`, the relation `r` acts as a translation in embedding space:

```
h + r ≈ t
```

KnowledgeForge uses this as the theoretical grounding for why the triple primitive is the right unit of storage — it's the minimal structure that supports both symbolic reasoning and geometric embedding.

**Source:** Bordes, Usunier, Garcia-Duran, Weston & Yakhnenko (2013). Translating Embeddings for Modeling Multi-relational Data. NeurIPS 2013.

---

## GraphRAG — Graph-aware retrieval

Edge et al. 2024 showed that grounding LLM answers in graph structure (rather than raw vector chunks) reduces hallucination F1 from 0.77 to 0.94 on knowledge-intensive QA tasks.

KnowledgeForge's `GraphRAG.ask()` flow:
1. **Anchor extraction** — entity IDs matching query terms (SQLite `LIKE` search)
2. **k-hop BFS expansion** — subgraph of up to 500 triples around anchors
3. **Serialisation** — `(subject)-[predicate]->(object) // evidence` facts
4. **Grounded LLM call** — system prompt: "Answer ONLY from provided facts. Cite triples."
5. **Return** — answer + `cited_triples` with full provenance

This is fundamentally different from RAG over vector chunks: the retrieval unit is a *graph neighbourhood*, not a text fragment.

**Source:** Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt & Larson (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.

---

## GNN Foundation — Battaglia 2018

Battaglia et al. 2018 established the theoretical case for graph networks as the right inductive bias for learning on relational data. The key insight: systems that can compose entities and relations generalise better than those that cannot.

KnowledgeForge's architecture reflects this: the triple is the unit of storage, the adapter protocol is the unit of domain generalisation, and GraphSAGE aggregation is the unit of structural learning.

**Source:** Battaglia et al. (2018). Relational Inductive Biases, Deep Learning, and Graph Networks. arXiv:1806.01261.

---

## What this is NOT

- Not a wrapper around LangChain or LlamaIndex
- Not pure vector search (RAG over chunks)
- Not a notebook demo

The entire stack — triple extraction, entity resolution, GraphSAGE aggregation, GraphRAG retrieval — implements what the primary literature actually specifies.
