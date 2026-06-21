# Competency Questions — KnowledgeForge ML/Algorithms Domain

**Objective:** Define queries the KG must answer. Each CQ maps to one or more **Ontology Design Patterns (ODPs)** — reusable solutions for common information needs.

**Domain:** Machine learning, graph algorithms, deep learning, papers, datasets, metrics.

---

## Competency Questions (CQs)

| # | Question | ODP | Predicates Used | Entity Types | Gap? |
|---|----------|-----|-----------------|---------------|------|
| CQ-1 | "What is TransE?" | Definition pattern | DEFINED_AS | Concept | ✅ |
| CQ-2 | "Who proposed TransE?" | Agent pattern | PROPOSED_BY, AUTHORED_BY | Person, Paper | ✅ |
| CQ-3 | "What algorithms improve on TransE?" | Predecessor chain | IMPROVES_ON | Algorithm | ✅ |
| CQ-4 | "What is TransE evaluated on?" | Evaluation pattern | EVALUATES_ON | Dataset, Benchmark | ✅ |
| CQ-5 | "What related concepts does TransE build on?" | Composition pattern | EXTENDS, REQUIRES | Algorithm, Concept | ✅ |
| CQ-6 | "What papers cite / reference TransE?" | Citation pattern | — **MISSING** | Paper | ❌ MISSING |
| CQ-7 | "How does the performance of TransE compare to Graph2Vec?" | Comparison pattern | OUTPERFORMS, ACHIEVES | Algorithm, Metric | ✅ |
| CQ-8 | "What systems implement TransE?" | Implementation pattern | IMPLEMENTED_IN | System, Codebase | ✅ |
| CQ-9 | "What are TransE's key hyperparameters?" | Property pattern | HAS_PARAMETER, HAS_PROPERTY | Value, Parameter | ❌ MISSING |
| CQ-10 | "What application domains use TransE?" | Application pattern | USED_IN | Domain, Application | ✅ |
| CQ-11 | "What is the lineage from Word2Vec → TransE → Graph embeddings?" | Genealogy pattern | EXTENDS, IMPROVES_ON, DERIVED_FROM | Algorithm | ⚠️ PARTIAL |
| CQ-12 | "What datasets are benchmarks for knowledge graphs?" | Classification pattern | TYPE_OF, EVALUATES_ON | Dataset | ✅ |
| CQ-13 | "What prerequisites does understanding transformers require?" | Dependency pattern | REQUIRES, PREREQUISITE_FOR | Concept | ✅ |

---

## ODP → Predicate Mapping

### ✅ Covered ODPs (Current Ontology)

| ODP | Pattern | Predicates | Example |
|-----|---------|-----------|---------|
| **Definition** | "X is defined as Y" | DEFINED_AS | TransE is defined as "embeddings in vector space" |
| **Agent** | "Person P proposed Algorithm A" | PROPOSED_BY, AUTHORED_BY | Bordes et al. proposed TransE |
| **Predecessor** | "A improves on B" | IMPROVES_ON, EXTENDS | Graph2Vec improves on TransE |
| **Evaluation** | "A evaluated on dataset D" | EVALUATES_ON | TransE evaluated on FB15k, WN18 |
| **Composition** | "A requires B" | REQUIRES, EXTENDS | Graph embeddings require TransE |
| **Performance** | "A achieves X% on metric M" | ACHIEVES, OUTPERFORMS | TransE achieves 95% on WN18 |
| **Implementation** | "A implemented in system S" | IMPLEMENTED_IN | TransE implemented in DGL, PyKEEN |
| **Application** | "A used in domain D" | USED_IN | TransE used in recommendation systems |
| **Type/Classification** | "X is a Y" | TYPE_OF | FB15k is a dataset |

### ❌ Missing ODPs (Ontology Gaps)

| ODP | Pattern | Missing Predicate | Example | CQ |
|-----|---------|-------------------|---------|-----|
| **Citation** | "Paper P1 cites P2" | CITES / CITED_BY / REFERENCES | "Kipf & Welling cite Bordes et al." | CQ-6 |
| **Property** | "Algorithm A has property P" | HAS_PARAMETER / HAS_PROPERTY | "TransE has 64 dimensions" | CQ-9 |
| **Genealogy** | "A derived from B over time" | DERIVED_FROM / EVOLVES_FROM | "Graph2Vec evolved from Word2Vec" | CQ-11 |
| **Prerequisite** | "Understanding A requires B" | PREREQUISITE_FOR | "Understanding RNNs requires linear algebra" | CQ-13 |

---

## Recommended Ontology Additions

### New Predicates (4)

1. **CITES / CITED_BY** — Paper-to-Paper citation edges
   - Domain: Paper → Paper
   - Inverse: CITED_BY
   - Example: `(Kipf2016_GCN, CITES, Bordes2013_TransE)`
   - Why: Critical for literature retrieval ("what papers reference this?")

2. **HAS_PARAMETER / HAS_PROPERTY** — Algorithm → Value mapping
   - Domain: Algorithm → Value
   - Example: `(TransE, HAS_PARAMETER, "embedding_dimension=64")`
   - Why: Enables comparison and replication queries

3. **PREREQUISITE_FOR** — Concept → Concept dependency (learning graph)
   - Domain: Concept → Concept (inverse of REQUIRES)
   - Example: `(LinearAlgebra, PREREQUISITE_FOR, NeuralNetworks)`
   - Why: Supports "learning paths" and dependency resolution

4. **DERIVED_FROM / EVOLVES_FROM** — Genealogy of algorithms
   - Domain: Algorithm → Algorithm
   - Example: `(Graph2Vec, DERIVED_FROM, Word2Vec)`
   - Why: Traces innovation lineage; complements IMPROVES_ON

### Enhanced Entity Types (0 additions, but refinement)

Current types are sufficient, but add optional **subtype tags** for refinement:
- `Dataset` → subtypes: `{Benchmark, Real-world, Synthetic, Standard}`
- `Algorithm` → subtypes: `{Embedding, Neural, Graph, Ranking, Optimization}`
- `Concept` → subtypes: `{Mathematical, Algorithmic, Architectural}`

---

## Implementation Checklist

- [ ] **Phase 3.5 (Ontology Refinement):** Update `llm.py` predicate list with 4 new predicates
- [ ] **Phase 3.5:** Add CITES/CITED_BY extraction to paper-notes extraction jobs
- [ ] **Phase 3.5:** Add HAS_PARAMETER extraction for algorithm sections
- [ ] **Phase 3.5:** Update SHACL shape validation to permit new predicates + domains
- [ ] **Phase 3.5:** Re-extract vault sections with expanded ontology
- [ ] **Validation:** Re-run CQ queries to verify answerability

---

## Validation: Can We Answer All CQs?

After adding 4 predicates, test each CQ end-to-end:

```bash
knowledgeforge query "What papers cite TransE?"  # ← CITES/CITED_BY
knowledgeforge query "What are TransE's dimensions?" # ← HAS_PARAMETER
knowledgeforge query "How does Graph2Vec descend from Word2Vec?" # ← DERIVED_FROM
knowledgeforge query "What do I need to understand transformers?" # ← PREREQUISITE_FOR
```

All CQs should resolve without gaps.

---

**Decision:** Should we add these 4 predicates to Phase 3.5, or ship Phase 4 with the current 16?

- **Pros (add now):** Closes obvious gaps; re-extraction is fast (~2 min vault-wide)
- **Cons (defer):** Phase 4 embedding pipeline will work regardless; can evaluate ODP completeness empirically
