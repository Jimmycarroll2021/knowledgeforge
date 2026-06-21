"""GraphRAG — graph-aware retrieval + LLM grounded answering.

Implements BOTH modes from Edge et al. 2024 (arXiv:2404.16130):

  LOCAL mode (default):
    k-hop BFS from anchor entities → serialised facts → grounded LLM answer.
    Best for specific entity questions. KGSWC 2024: hallucination F1 0.77→0.94.

  GLOBAL mode:
    Community summaries → relevance scoring → top-K context → LLM synthesis.
    Best for broad thematic questions ("what are the main topics?").
    Requires `knowledgeforge community` to have been run first.

Edge et al. 2024: "local retrieval is insufficient for questions that require
understanding the dataset as a whole."
"""
from __future__ import annotations

import os
import subprocess
from collections import deque
from typing import TYPE_CHECKING, Any

from ..store.sqlite import SQLiteGraphStore

if TYPE_CHECKING:
    import anthropic

    from ..embeddings.pipeline import EmbeddingPipeline
    from ..resolution.resolver import EntityResolver

# Trusted layers — answers ground only on these. The llm_hypotheses layer is
# excluded so a speculative LLM-authored triple can never back a "grounded"
# answer (correctness + safety).
_TRUSTED_LAYERS = ("source_facts", "normalised_triples", "inferred_relations")

# Anchors seeded from semantic search keep only hits above this cosine floor —
# below it the vector match is too weak to trust as an anchor.
_ANCHOR_SCORE_FLOOR = 0.3


def _claude_cli(prompt: str, system: str, model: str) -> str:
    """Call Claude via `claude -p` (OAuth — no API key needed)."""
    full = f"<system>\n{system}\n</system>\n\n{prompt}"
    r = subprocess.run(
        ["claude", "-p", "--model", model, "-"],
        input=full, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {r.stderr[:200]}")
    return r.stdout.strip()

_SYSTEM_PROMPT = """You are a knowledge graph assistant. Answer questions using ONLY the provided graph facts.

Rules:
- Ground every claim in a specific cited triple from the facts provided
- If the facts don't support an answer, say exactly: "The graph does not contain evidence for this."
- Do not add background knowledge or infer beyond what the facts state
- Cite triples as: [Subject → Predicate → Object]
- Be precise and concise
"""


class GraphRAG:
    """Graph-aware retrieval + LLM grounded answering.

    Implements the flow from KF - LLM Integration.md:
      1. Find anchor entities matching query terms
      2. k-hop BFS subgraph expansion
      3. Serialise triples as human-readable facts
      4. Claude API call with grounded prompt
      5. Return answer + cited evidence
    """

    def __init__(
        self,
        store: SQLiteGraphStore,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        hops: int = 2,
        embed: "EmbeddingPipeline | None" = None,
        resolver: "EntityResolver | None" = None,
        trusted_layers: set[str] | None = None,
    ) -> None:
        self._store = store
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._hops = hops
        self._client: anthropic.Anthropic | None = None
        self._use_cli = not bool(self._api_key)
        self._embed = embed
        self._resolver = resolver
        # Layers retrieval may ground on (default: all but llm_hypotheses).
        self._trusted_layers = (
            set(trusted_layers) if trusted_layers is not None else set(_TRUSTED_LAYERS)
        )

    def ask(self, question: str, mode: str = "local") -> dict[str, Any]:
        """Answer a question using graph-grounded retrieval.

        mode="local"  — k-hop BFS from anchor entities (default)
        mode="global" — community summary search (Edge et al. 2024 global mode)

        Returns: {answer, evidence, subgraph_size, anchor_entities, mode}
        """
        if mode == "global":
            return self._ask_global(question)
        anchors = self._find_anchors(question)
        subgraph = self._expand_subgraph(anchors, self._hops)

        if not subgraph:
            return {
                "answer": "No relevant entities found in the graph for this question.",
                "evidence": [],
                "subgraph_size": 0,
                "anchor_entities": anchors,
            }

        facts = self._serialise(subgraph)
        answer = self._call_llm(question, facts)

        return {
            "answer": answer,
            "mode": "local",
            "evidence": subgraph[:50],  # top 50 triples as evidence
            "subgraph_size": len(subgraph),
            "anchor_entities": anchors,
        }

    # ── global mode (Edge et al. 2024) ────────────────────────────────────────

    _GLOBAL_SYSTEM = """You are a knowledge graph analyst. Given summaries of thematic
communities detected in a knowledge graph, synthesise a comprehensive answer to the question.

Rules:
- Draw on the community summaries provided — these represent the graph's major themes
- Cite which community/theme each claim comes from
- If the communities don't address the question, say so explicitly
- Be precise and structured — use the terminology from the summaries
"""

    def _ask_global(self, question: str) -> dict[str, Any]:
        """Global query mode — Edge et al. 2024.

        Searches across LLM-generated community summaries rather than
        k-hop local neighbourhoods. Answers thematic/holistic questions.
        """
        from ..community.detector import CommunityDetector
        detector = CommunityDetector(self._store, model=self._model)
        summaries = detector.load_summaries()

        if not summaries:
            return {
                "answer": "No community summaries found. Run 'knowledgeforge community' first to build global query support.",
                "mode": "global",
                "evidence": [],
                "subgraph_size": 0,
                "anchor_entities": [],
                "communities_used": 0,
            }

        # Map step (Edge et al. 2024): when embeddings are available, rank
        # community summaries by cosine similarity of the question against each
        # summary, then take the top-K. Falls back to keyword overlap otherwise.
        top_k = self._rank_communities(question, summaries)

        context = "\n\n---\n\n".join(
            f"[Community {s['community_id']} — {s['member_count']} entities]\n{s['summary']}"
            for s in top_k
        )
        answer = self._call_llm(question, context, system=self._GLOBAL_SYSTEM)

        return {
            "answer": answer,
            "mode": "global",
            "evidence": [],
            "subgraph_size": sum(s["member_count"] for s in top_k),
            "anchor_entities": [],
            "communities_used": len(top_k),
        }

    def _rank_communities(
        self, question: str, summaries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rank community summaries for global mode, return the top 20.

        Map step: with embeddings, score each summary by cosine similarity of
        ``embed_query(question)`` vs ``embed_query(summary)`` (both L2-normalised
        → dot product). Without embeddings (or on any failure), fall back to the
        keyword-overlap ranking.
        """
        if self._embed is not None:
            try:
                import numpy as np

                q_vec = self._embed.embed_query(question)
                scored = [
                    (float(np.dot(q_vec, self._embed.embed_query(s["summary"]))), s)
                    for s in summaries
                ]
                scored.sort(key=lambda pair: pair[0], reverse=True)
                return [s for _, s in scored[:20]]
            except Exception:
                pass  # fall through to keyword-overlap ranking

        q_terms = set(self._clean_terms(question))

        def score(s: dict[str, Any]) -> float:
            text = s["summary"].lower()
            return sum(1.0 for t in q_terms if t in text) / max(len(q_terms), 1)

        ranked = sorted(summaries, key=score, reverse=True)
        return ranked[:20]

    def _clean_terms(self, text: str) -> list[str]:
        import re as _re
        return [
            _re.sub(r"[^\w]", "", w)
            for w in text.lower().split()
            if len(w) > 3 and w not in self._STOPWORDS
        ]

    # ── retrieval ─────────────────────────────────────────────────────────────

    _STOPWORDS = {
        "what", "that", "this", "with", "from", "have", "been", "does",
        "into", "when", "where", "which", "they", "them", "their", "will",
        "would", "could", "should", "between", "about", "using", "used",
        "how", "the", "and", "for", "are", "was", "how", "building",
        "relationship", "connect", "connection",
    }

    def _find_anchors(self, question: str) -> list[str]:
        """Find anchor entity IDs for a question.

        When an embedding pipeline is set, semantic anchors (vector search over
        the question) are seeded first, then UNIONed with the substring-LIKE
        fallback so dense and lexical matches both contribute. When a resolver
        is set, every anchor is mapped through ``canonical()`` so retrieval runs
        on resolved/canonical entities (deduped). Triple-derived anchors are
        confined to the trusted layers (llm_hypotheses excluded).
        """
        import re as _re
        anchors: list[str] = []
        seen: set[str] = set()

        def _add(eid: str) -> None:
            if eid not in seen:
                seen.add(eid)
                anchors.append(eid)

        # ── semantic anchors (vector search) ──────────────────────────────────
        if self._embed is not None:
            try:
                for hit in self._embed.search_by_text(question, k=10):
                    if hit.get("score", 0.0) >= _ANCHOR_SCORE_FLOOR:
                        _add(hit["entity_id"])
            except Exception:
                pass  # degrade to LIKE fallback if embedding search unavailable

        # ── substring-LIKE fallback (UNION with semantic anchors) ─────────────
        terms = [
            _re.sub(r"[^\w]", "", w)
            for w in question.lower().split()
        ]
        terms = [w for w in terms if len(w) > 3 and w not in self._STOPWORDS]

        layer_clause = self._layer_in_clause("layer")
        for term in terms:
            rows = self._store._conn.execute(
                "SELECT id FROM entities WHERE LOWER(id) LIKE ? LIMIT 10",
                (f"%{term}%",),
            ).fetchall()
            for row in rows:
                _add(row[0])

            # also search triple objects — trusted layers only
            rows = self._store._conn.execute(
                f"""SELECT DISTINCT subject FROM triples
                    WHERE LOWER(object) LIKE ? AND {layer_clause}
                    LIMIT 10""",
                (f"%{term}%", *self._trusted_layers),
            ).fetchall()
            for row in rows:
                _add(row[0])

        # ── canonicalise through the resolver (dedupe after) ──────────────────
        if self._resolver is not None:
            canon: list[str] = []
            canon_seen: set[str] = set()
            for eid in anchors:
                try:
                    cid = self._resolver.canonical(eid)
                except Exception:
                    cid = eid
                if cid not in canon_seen:
                    canon_seen.add(cid)
                    canon.append(cid)
            anchors = canon

        return anchors[:20]

    def _layer_in_clause(self, column: str) -> str:
        """SQL fragment restricting ``column`` to the trusted layers.

        Returns e.g. ``layer IN (?,?,?)`` with one placeholder per trusted
        layer; bind ``*self._trusted_layers`` (in iteration order) after the
        other params.
        """
        placeholders = ",".join("?" * len(self._trusted_layers))
        return f"{column} IN ({placeholders})"

    def _expand_subgraph(self, anchors: list[str], hops: int) -> list[dict[str, Any]]:
        """BFS k-hop expansion from anchor entities."""
        visited: set[str] = set()
        frontier: deque[tuple[str, int]] = deque()
        for a in anchors:
            frontier.append((a, 0))
            visited.add(a)

        subgraph: list[dict[str, Any]] = []
        seen_triples: set[str] = set()

        while frontier and len(subgraph) < 500:
            node, depth = frontier.popleft()

            rows = self._store._conn.execute(
                f"""SELECT triple_id, subject, predicate, object,
                          source_kind, target_kind, confidence, evidence, source_doc
                   FROM triples
                   WHERE (subject = ? OR object = ?)
                     AND {self._layer_in_clause("layer")}
                   LIMIT 50""",
                (node, node, *self._trusted_layers),
            ).fetchall()

            for row in rows:
                tid = row[0]
                if tid in seen_triples:
                    continue
                seen_triples.add(tid)
                t = dict(zip(
                    ["triple_id", "subject", "predicate", "object",
                     "source_kind", "target_kind", "confidence", "evidence", "source_doc"],
                    row,
                ))
                subgraph.append(t)

                if depth < hops:
                    for neighbour in [row[1], row[3]]:  # subject, object
                        if neighbour not in visited:
                            visited.add(neighbour)
                            frontier.append((neighbour, depth + 1))

        return subgraph

    def _serialise(self, subgraph: list[dict[str, Any]]) -> str:
        """Serialise triples as human-readable facts for the LLM context."""
        # Skip noise predicates, then topic-guided pruning: order by confidence
        # (desc) so the strongest evidence survives the 200-line truncation.
        _SKIP = {"CONTAINS_KEY", "HAS_FILE_TYPE"}
        useful = [t for t in subgraph if t["predicate"] not in _SKIP]
        useful.sort(key=lambda t: (-(t.get("confidence") or 0.0), t["predicate"], t["subject"]))

        lines: list[str] = []
        for t in useful[:200]:
            conf = f" [conf={t['confidence']:.2f}]" if t["confidence"] < 0.9 else ""
            lines.append(
                f"({t['subject']})-[{t['predicate']}]->({t['object']}){conf}"
                f"  // {t['evidence'][:80]}" if t.get("evidence") else
                f"({t['subject']})-[{t['predicate']}]->({t['object']}){conf}"
            )
        return "\n".join(lines)

    def _call_llm(self, question: str, facts: str, system: str | None = None) -> str:
        sys = system or _SYSTEM_PROMPT
        context = f"Graph facts:\n{facts}\n\nQuestion: {question}"
        try:
            if self._use_cli:
                return _claude_cli(context, sys, self._model)
            client = self._get_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=sys,
                messages=[{"role": "user", "content": context}],
            )
            block = response.content[0]
            return block.text.strip() if block.type == "text" else ""
        except Exception as exc:
            return f"LLM call failed: {exc}"

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ── offline path query (no LLM needed) ───────────────────────────────────

    def path(
        self, from_entity: str, to_entity: str, max_depth: int = 4
    ) -> list[list[dict[str, Any]]]:
        """Find shortest path between two entities in the graph (BFS, no LLM)."""
        if from_entity == to_entity:
            return [[]]

        parent: dict[str, tuple[str, dict[str, Any]] | None] = {from_entity: None}
        frontier: deque[str] = deque([from_entity])
        depth_map: dict[str, int] = {from_entity: 0}

        while frontier:
            node = frontier.popleft()
            if depth_map[node] >= max_depth:
                continue

            rows = self._store._conn.execute(
                "SELECT subject, predicate, object FROM triples WHERE subject=? OR object=?",
                (node, node),
            ).fetchall()

            for subj, pred, obj in rows:
                neighbour = obj if subj == node else subj
                if neighbour not in parent:
                    parent[neighbour] = (node, {"subject": subj, "predicate": pred, "object": obj})
                    depth_map[neighbour] = depth_map[node] + 1
                    if neighbour == to_entity:
                        return [self._reconstruct_path(parent, to_entity)]
                    frontier.append(neighbour)
        return []

    def _reconstruct_path(
        self, parent: dict[str, tuple[str, dict[str, Any]] | None], node: str
    ) -> list[dict[str, Any]]:
        path: list[dict[str, Any]] = []
        while parent.get(node) is not None:
            entry = parent[node]
            assert entry is not None  # loop guard: parent.get(node) is not None
            prev, edge = entry
            path.append(edge)
            node = prev
        path.reverse()
        return path
