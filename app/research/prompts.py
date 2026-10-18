"""
app/research/prompts.py
-----------------------
Prompt engineering for the AutoResearch AI research agent.

All prompts are assembled here so they can be read, audited, and
updated without touching agent logic.
"""

from typing import Optional
from app.rag.vector_store import SearchResult


def build_research_prompt(
    topic: str,
    results: list[SearchResult],
    source_registry: Optional[dict[str, dict]] = None,
) -> str:
    """
    Build the full Gemini prompt for research report generation.

    The context window includes each retrieved chunk labelled with its unique
    source identifier ([S1], [S2], …). Multiple chunks from the same source
    URL share the same identifier. Gemini is instructed to cite these
    identifiers when making claims.

    Args:
        topic:           The original research topic.
        results:         Retrieved SearchResult objects from the RAG layer.
        source_registry: Optional pre-computed mapping of identifier -> {title, url}.

    Returns:
        A complete prompt string ready to pass to GeminiClient.generate().
    """
    if source_registry is None:
        source_registry = assign_source_identifiers(results)

    url_to_identifier = {meta["url"]: ident for ident, meta in source_registry.items()}
    context_block = _build_context_block(results, url_to_identifier)

    prompt = f"""You are a professional research analyst. Your task is to produce a
structured, grounded research report on the following topic:

TOPIC: {topic}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH CONTEXT (retrieved from trusted sources)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Answer ONLY using the research context provided above.
2. Do NOT invent facts, statistics, URLs, or claims not present in the context.
3. You may ONLY use source identifiers that appear in the provided research context. Never invent or create a source identifier.
4. If the context does not contain enough information on a subtopic, explicitly
   state: "The provided context does not contain sufficient information on this
   subtopic."
5. Clearly distinguish established facts from your interpretation.
6. Cite every factual claim using the source identifiers [S1], [S2], etc.
   from the context above.
7. Do NOT include URLs in the report body — URLs are managed separately.
8. Produce ONLY valid JSON. Do not include any text before or after the JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (strict JSON, no markdown fences)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "executive_summary": "2–4 sentence overview grounded in the context.",
  "key_findings": [
    "Finding 1 with citation e.g. [S1]",
    "Finding 2 with citation e.g. [S2]",
    "..."
  ],
  "detailed_analysis": "Multi-paragraph detailed analysis citing [S1], [S2], etc.",
  "emerging_trends": [
    "Trend 1 observed in the context [S1]",
    "..."
  ],
  "conclusion": "1–2 paragraph conclusion grounded only in the context."
}}
"""
    return prompt.strip()


def _build_context_block(
    results: list[SearchResult],
    url_to_identifier: Optional[dict[str, str]] = None,
) -> str:
    """
    Format retrieved SearchResults into a numbered context block.

    Each result is labelled with its unique source identifier [S1], [S2], …
    corresponding to its URL. Multiple chunks from the same URL map to the
    same identifier.
    """
    if not results:
        return "(No context retrieved — context is empty.)"

    if url_to_identifier is None:
        registry = assign_source_identifiers(results)
        url_to_identifier = {meta["url"]: ident for ident, meta in registry.items()}

    parts: list[str] = []
    for result in results:
        url = result.metadata.get("source_url", "")
        identifier = url_to_identifier.get(url, "[S1]")
        title = result.metadata.get("source_title", "Unknown Source")
        text = (result.text or "").strip()

        parts.append(
            f"{identifier}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(parts)


def assign_source_identifiers(results: list[SearchResult]) -> dict[str, dict]:
    """
    Build a mapping of identifier → {title, url} from retrieved results,
    deduplicated by canonical URL.

    Each unique URL receives exactly one source identifier ([S1], [S2], ...).
    Multiple chunks from the same URL map to the same source identifier.
    URLs come exclusively from the SearchResult metadata.

    Returns:
        {"[S1]": {"title": "...", "url": "..."}, "[S2]": ...}
    """
    registry: dict[str, dict] = {}
    seen_urls: dict[str, str] = {}

    for result in results:
        url = result.metadata.get("source_url", "")
        title = result.metadata.get("source_title", "Unknown Source")

        if url not in seen_urls:
            identifier = f"[S{len(seen_urls) + 1}]"
            seen_urls[url] = identifier
            registry[identifier] = {"title": title, "url": url}

    return registry
