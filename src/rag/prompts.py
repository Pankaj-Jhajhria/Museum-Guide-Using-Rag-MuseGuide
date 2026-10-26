"""Prompt engineering and context formatting templates for MuseGuide RAG."""

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are MuseGuide, an expert, articulate museum docent and art historian representing the Cleveland Museum of Art.
Your purpose is to answer visitors' questions about historical artefacts with scholarly accuracy, cultural sensitivity, and engaging storytelling.

CRITICAL INSTRUCTIONS:
1. STRICT EVIDENCE GROUNDING: Use ONLY the information provided in the "MUSEUM EVIDENCE" section below. Do NOT fabricate, assume, or extrapolate facts, creators, dates, cultures, or meanings that are not directly supported by the text.
2. INLINE CITATIONS: For every factual claim, creator attribution, date, material, or interpretive detail, append an inline citation using bracketed numbers matching the document number, e.g., [1] or [1][2].
3. HONESTY & ANTI-HALLUCINATION: If the provided museum evidence does NOT contain sufficient information to answer the question, clearly state: "Based on the museum records available, I cannot answer this fully." Do NOT invent history or fill in gaps with external knowledge.
4. ENGAGING TONE: Present your answer in a natural, museum-guide voice with clear paragraphs. Avoid bulleted walls of text unless explicitly requested.
5. ACCURACY: Mention key identifying features (artefact title, creator, date/period, medium, culture) as documented in the records."""

REFUSAL_TEMPLATE = """Based on the current Cleveland Museum of Art records in the MuseGuide collection, I could not find sufficient historical evidence to answer your question regarding: "{query}".

The retrieved artefacts had a low relevance score (top match score: {top_score:.2f}). To explore the collection, please try searching with specific artist names, historical periods, cultures (e.g., "Ancient Egyptian", "Greek amphora", "Edo period Japan"), or artistic mediums."""


def format_context_document(doc_index: int, item: Dict[str, Any]) -> str:
    """Format a single retrieved museum record into a clean, dense evidence block."""
    title = item.get("title") or "Untitled"
    creator = item.get("creator_name") or "Unknown Artist"
    date_str = item.get("creation_date") or "Date Unknown"
    culture = item.get("culture") or "Culture Unspecified"
    dept = item.get("department") or "General Collection"
    technique = item.get("technique") or "Medium Unspecified"
    url = item.get("museum_url") or "N/A"
    desc = item.get("description") or "No curatorial description available."

    lines = [
        f"--- [Document {doc_index}] ---",
        f"Title: {title}",
        f"Creator: {creator}",
        f"Date: {date_str}",
        f"Culture: {culture}",
        f"Medium/Technique: {technique}",
        f"Department: {dept}",
        f"Museum Record Link: {url}",
        f"Curatorial Description: {desc}",
    ]
    return "\n".join(lines)


def build_rag_prompt(query: str, retrieved_artworks: List[Dict[str, Any]], max_items: int = 4) -> str:
    """Assemble the complete prompt containing system instructions, evidence documents, and user query.

    Args:
        query: User's natural language question.
        retrieved_artworks: List of retrieved artwork dicts from SemanticSearcher.
        max_items: Maximum evidence documents to include.

    Returns:
        Formatted prompt string ready for LLM input.
    """
    selected = retrieved_artworks[:max_items]
    evidence_blocks = [
        format_context_document(i + 1, item)
        for i, item in enumerate(selected)
    ]
    context_str = "\n\n".join(evidence_blocks)

    prompt = f"""{SYSTEM_PROMPT}

==============================
MUSEUM EVIDENCE:
==============================
{context_str}

==============================
VISITOR QUESTION:
{query}

==============================
YOUR GUIDED ANSWER (with [1], [2] citations):"""
    return prompt
