"""GraphRAG — graph-aware retrieval + LLM grounded answering.

From concepts/GraphRAG.md and KF - LLM Integration.md:
  - k-hop subgraph extraction (BFS on SQLite)
  - Serialise as structured facts: "(GraphSAGE)-[PROPOSED_BY]->(Hamilton 2017)"
  - System: "Answer only from provided facts."
  - Returns: {answer, evidence, subgraph_size}

KGSWC 2024: chain-of-thought + KG injection reduces hallucination F1 0.77 → 0.94.
Default hops=2 safe to ~10K nodes per KF spec. hops=3+ can exceed context window.
"""
from __future__ import annotations

import os
import subprocess
from collections import deque

from ..store.sqlite import SQLiteGraphStore


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
    ) -> None:
        self._store = store
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._hops = hops
        self._client: object | None = None
        self._use_cli = not bool(self._api_key)

    def ask(self, question: str) -> dict:
        """Answer a question using graph-grounded retrieval.

        Returns: {answer, evidence, subgraph_size, anchor_entities}
        """
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
            "evidence": subgraph[:50],  # top 50 triples as evidence
            "subgraph_size": len(subgraph),
            "anchor_entities": anchors,
        }

    # ── retrieval ─────────────────────────────────────────────────────────────

    _STOPWORDS = {
        "what", "that", "this", "with", "from", "have", "been", "does",
        "into", "when", "where", "which", "they", "them", "their", "will",
        "would", "could", "should", "between", "about", "using", "used",
        "how", "the", "and", "for", "are", "was", "how", "building",
        "relationship", "connect", "connection",
    }

    def _find_anchors(self, question: str) -> list[str]:
        """Find entity IDs that match terms in the question."""
        terms = [
            w for w in question.lower().split()
            if len(w) > 3 and w not in self._STOPWORDS
        ]
        anchors: list[str] = []
        seen: set[str] = set()

        for term in terms:
            rows = self._store._conn.execute(
                "SELECT id FROM entities WHERE LOWER(id) LIKE ? LIMIT 10",
                (f"%{term}%",),
            ).fetchall()
            for row in rows:
                eid = row[0]
                if eid not in seen:
                    seen.add(eid)
                    anchors.append(eid)

            # also search triple objects
            rows = self._store._conn.execute(
                "SELECT DISTINCT subject FROM triples WHERE LOWER(object) LIKE ? LIMIT 10",
                (f"%{term}%",),
            ).fetchall()
            for row in rows:
                eid = row[0]
                if eid not in seen:
                    seen.add(eid)
                    anchors.append(eid)

        return anchors[:20]

    def _expand_subgraph(self, anchors: list[str], hops: int) -> list[dict]:
        """BFS k-hop expansion from anchor entities."""
        visited: set[str] = set()
        frontier: deque[tuple[str, int]] = deque()
        for a in anchors:
            frontier.append((a, 0))
            visited.add(a)

        subgraph: list[dict] = []
        seen_triples: set[str] = set()

        while frontier and len(subgraph) < 500:
            node, depth = frontier.popleft()

            rows = self._store._conn.execute(
                """SELECT triple_id, subject, predicate, object,
                          source_kind, target_kind, confidence, evidence, source_doc
                   FROM triples
                   WHERE subject = ? OR object = ?
                   LIMIT 50""",
                (node, node),
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

    def _serialise(self, subgraph: list[dict]) -> str:
        """Serialise triples as human-readable facts for the LLM context."""
        # Sort by predicate for readability, skip noise predicates
        _SKIP = {"CONTAINS_KEY", "HAS_FILE_TYPE"}
        useful = [t for t in subgraph if t["predicate"] not in _SKIP]
        useful.sort(key=lambda t: (t["predicate"], t["subject"]))

        lines: list[str] = []
        for t in useful[:200]:
            conf = f" [conf={t['confidence']:.2f}]" if t["confidence"] < 0.9 else ""
            lines.append(
                f"({t['subject']})-[{t['predicate']}]->({t['object']}){conf}"
                f"  // {t['evidence'][:80]}" if t.get("evidence") else
                f"({t['subject']})-[{t['predicate']}]->({t['object']}){conf}"
            )
        return "\n".join(lines)

    def _call_llm(self, question: str, facts: str) -> str:
        context = f"Graph facts:\n{facts}\n\nQuestion: {question}"
        try:
            if self._use_cli:
                return _claude_cli(context, _SYSTEM_PROMPT, self._model)
            client = self._get_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            return f"LLM call failed: {exc}"

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ── offline path query (no LLM needed) ───────────────────────────────────

    def path(self, from_entity: str, to_entity: str, max_depth: int = 4) -> list[list[dict]]:
        """Find shortest path between two entities in the graph (BFS, no LLM)."""
        if from_entity == to_entity:
            return [[]]

        parent: dict[str, tuple[str, dict] | None] = {from_entity: None}
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

    def _reconstruct_path(self, parent: dict, node: str) -> list[dict]:
        path: list[dict] = []
        while parent.get(node) is not None:
            prev, edge = parent[node]
            path.append(edge)
            node = prev
        path.reverse()
        return path
