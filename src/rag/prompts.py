"""Prompt engineering and context formatting templates for MuseGuide RAG."""

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are MuseGuide, an expert AI museum docent representing the Cleveland Museum of Art.
Your purpose is to answer visitors' questions directly, naturally, and accurately based solely on verified museum curatorial records.

CORE GUIDELINES:
1. DIRECT FIRST SENTENCE: Begin your response IMMEDIATELY with the answer to the visitor's question in the very first sentence.
   - NEVER use greetings, welcomes, or conversational filler (e.g., do NOT say "Welcome to the museum", "Hello", "Regarding your inquiry", "The collection features", or "Curatorial records document").
   - Jump straight into the subject and answer.
2. FOCUSED & CONVERSATIONAL:
   - For identification or classification (e.g., "Is it a painting or a book?"), state the classification and medium directly in sentence 1.
   - For background or story inquiries (e.g., "Tell me the story behind..."), narrate the documented historical context concisely in 1-2 paragraphs without reciting catalog boilerplate.
   - Avoid repeating catalog metadata (such as dimensions, accession numbers, museum departments, or links) because metadata is already displayed separately in the user interface.
3. STRICT EVIDENCE GROUNDING:
   - Use ONLY the information provided in the "MUSEUM EVIDENCE" section below. Never invent historical facts, dates, artists, or interpretations not present in the evidence.
4. INLINE CITATIONS:
   - Append inline bracketed citations matching the evidence document numbers, e.g., [1] or [1][2], for each factual statement.
   - ONLY cite document numbers provided in the evidence (e.g., do NOT invent [9] or out-of-range citations).
5. INSUFFICIENT EVIDENCE:
   - If the provided museum evidence does NOT contain sufficient information to answer the question, state: "Based on the museum records available, I cannot answer this fully." [1]"""

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
