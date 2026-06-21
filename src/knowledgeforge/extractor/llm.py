"""LLM semantic triple extractor.

Reads text sections and extracts factual (subject, predicate, object) triples
using Claude. Grounded in TransE: every valid triple satisfies h + r ≈ t.

From KF - LLM Integration.md:
  - Use claude-haiku by default (fast, cheap)
  - Extract per section, not per file (better precision, smaller context)
  - Extraction method = "llm" (distinguishes from rule-based)

From KF - Triple Primitive.md:
  - subject_type + object_type → typed heterogeneous nodes
  - confidence: 0.9 direct statement, 0.7 implied
  - inferred: False for extraction (True only for inference engine)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..contracts import SourceDocument, Triple, now_iso

_HEADING_SPLIT_RE = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n?", re.MULTILINE)  # strip fences only, keep content
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_MIN_SECTION_CHARS = 60

# Predicates aligned with the KnowledgeForge ontology (from vault notes)
_PREDICATE_GUIDANCE = """
Use ONLY these predicates (SCREAMING_SNAKE_CASE):
  PROPOSED_BY       — Algorithm/Concept → Paper/Person
  TYPE_OF           — Entity → Category
  IMPROVES_ON       — Algorithm → Algorithm (predecessor)
  EXTENDS           — Algorithm → Algorithm (builds on)
  EVALUATES_ON      — Algorithm → Dataset/Benchmark
  ACHIEVES          — Algorithm → Metric/Result (e.g. "51% F1 improvement")
  SCALES_TO         — Algorithm → Scale/System ("3B nodes")
  USED_IN           — Algorithm → Application/Domain
  AUTHORED_BY       — Paper → Person
  PUBLISHED_IN      — Paper → Venue/Year
  RELATED_TO        — Concept → Concept (bidirectional)
  DEFINED_AS        — Concept → Definition (use literal short definition as object)
  PART_OF           — Component → System
  REQUIRES          — Algorithm → Prerequisite
  OUTPERFORMS       — Algorithm → Algorithm (on a benchmark)
  IMPLEMENTED_IN    — Algorithm → System/Codebase
  SIMILAR_TO        — Entity → Entity (structural or semantic similarity)
"""

_SYSTEM_PROMPT = f"""You are a precision knowledge graph triple extractor specialising in machine learning and graph algorithms.

Extract factual (subject, predicate, object) triples ONLY from facts EXPLICITLY stated in the text.
Do NOT infer, generalise, or add background knowledge.

Mathematical grounding (TransE): each triple (h, r, t) should satisfy h + r ≈ t in embedding space.
This means: subjects and objects must be specific named entities, predicates must be directional relations.

{_PREDICATE_GUIDANCE}

Entity types (use these exactly):
  Algorithm, Concept, Paper, Person, Dataset, Organization, Metric, System, Value

Output JSON only. No markdown. No explanation.
Schema:
{{
  "triples": [
    {{
      "subject": "exact name",
      "subject_type": "EntityType",
      "predicate": "PREDICATE",
      "object": "exact name or short literal",
      "object_type": "EntityType",
      "evidence": "verbatim sentence from text supporting this triple",
      "confidence": 0.9
    }}
  ]
}}

Rules:
- confidence 0.9 = directly stated; 0.75 = strongly implied; 0.6 = background knowledge being applied
- If no clear triples exist, return {{"triples": []}}
- Max 15 triples per section
- Objects must be specific, not vague ("Hamilton 2017" not "a paper")
"""


def _call_claude_cli(prompt: str, system: str, model: str) -> str:
    """Call Claude via `claude -p` CLI (uses OAuth — no API key needed)."""
    import subprocess
    full_prompt = f"<system>\n{system}\n</system>\n\n{prompt}"
    result = subprocess.run(
        ["claude", "-p", "--model", model, "-"],
        input=full_prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:200]}")
    return result.stdout.strip()


class LLMExtractor:
    """Extracts semantic triples from text using Claude.

    Splits text at section boundaries (headings), extracts per section.

    Auth priority:
      1. ANTHROPIC_API_KEY env var → uses Anthropic Python SDK
      2. Fallback → calls `claude -p` CLI (uses Claude Code OAuth auth)
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_sections: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_sections = max_sections
        self._client: object | None = None
        self._use_cli = not bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def extract(self, doc: SourceDocument) -> list[Triple]:
        """Extract semantic triples from a SourceDocument."""
        try:
            text = doc.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self.extract_text(text, doc)

    def extract_text(self, text: str, doc: SourceDocument) -> list[Triple]:
        """Extract triples from raw text attributed to doc."""
        sections = self._split_sections(text)
        if self._max_sections:
            sections = sections[: self._max_sections]

        all_triples: list[Triple] = []
        for section in sections:
            triples = self._extract_section(section, doc)
            all_triples.extend(triples)
        return all_triples

    # ── private ───────────────────────────────────────────────────────────────

    def _split_sections(self, text: str) -> list[str]:
        # Strip YAML frontmatter, then strip just code fences (keep equation content)
        text = _FRONTMATTER_RE.sub("", text)
        text = _CODE_FENCE_RE.sub("", text)
        boundaries = [m.start() for m in _HEADING_SPLIT_RE.finditer(text)]
        if not boundaries:
            return [text.strip()] if len(text.strip()) >= _MIN_SECTION_CHARS else []

        sections: list[str] = []
        # include content before first heading (intro/abstract section)
        intro = text[: boundaries[0]].strip()
        if len(intro) >= _MIN_SECTION_CHARS:
            sections.append(intro)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            section = text[start:end].strip()
            if len(section) >= _MIN_SECTION_CHARS:
                sections.append(section)
        return sections

    def _extract_section(self, section: str, doc: SourceDocument) -> list[Triple]:
        ts = now_iso()

        try:
            if self._use_cli:
                raw = _call_claude_cli(section[:4000], _SYSTEM_PROMPT, self._model)
            else:
                client = self._get_client()
                response = client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": section[:4000]}],
                )
                raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
        except Exception:
            return []

        triples: list[Triple] = []
        for item in data.get("triples", []):
            try:
                t = Triple(
                    subject=str(item["subject"])[:240],
                    predicate=str(item["predicate"])[:80],
                    object=str(item["object"])[:240],
                    source_kind=str(item.get("subject_type", "Entity")),
                    target_kind=str(item.get("object_type", "Entity")),
                    evidence=str(item.get("evidence", ""))[:700],
                    confidence=float(item.get("confidence", 0.8)),
                    source_doc=doc.relative_path,
                    extraction_method="llm",
                    timestamp=ts,
                    adapter="llm_extractor",
                )
                triples.append(t)
            except (KeyError, ValueError):
                continue
        return triples
