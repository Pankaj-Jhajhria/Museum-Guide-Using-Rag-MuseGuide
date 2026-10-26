"""Automated unit and evaluation tests for MuseGuide Grounded Answer Generation (RAG)."""

import json
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.answer import AnswerGenerator, Citation, RAGResponse
from src.rag.prompts import build_rag_prompt, format_context_document
from src.retrieval.search import SemanticSearcher


@pytest.fixture(scope="module")
def searcher():
    """Shared SemanticSearcher instance."""
    index_path = Path("data/index/cma_index.faiss")
    if not index_path.exists():
        pytest.skip("Full FAISS index not built at data/index/cma_index.faiss")
    return SemanticSearcher(
        index_path=str(index_path),
        metadata_path="data/index/cma_metadata.json",
    )


@pytest.fixture(scope="module")
def generator(searcher):
    """AnswerGenerator instance using extractive docent synthesizer."""
    return AnswerGenerator(searcher=searcher, min_confidence=0.35, top_k=4)


def test_format_context_document():
    """Verify single document formatting includes essential fields."""
    sample_doc = {
        "title": "Nathaniel Hurd",
        "creator_name": "John Singleton Copley",
        "creation_date": "c. 1765",
        "culture": "United States",
        "department": "American Painting and Sculpture",
        "technique": "oil on canvas",
        "museum_url": "https://clevelandart.org/art/1915.534",
        "description": "Hurd was a silversmith.",
    }
    block = format_context_document(1, sample_doc)
    assert "[Document 1]" in block
    assert "Nathaniel Hurd" in block
    assert "John Singleton Copley" in block
    assert "oil on canvas" in block
    assert "https://clevelandart.org/art/1915.534" in block


def test_build_rag_prompt():
    """Verify complete prompt assembly contains system instructions, evidence, and query."""
    docs = [
        {
            "title": "Uraeus Cobra",
            "creator_name": "Unknown",
            "creation_date": "1307-1196 BCE",
            "culture": "Egypt",
            "department": "Egyptian Art",
            "technique": "bronze",
            "museum_url": "https://clevelandart.org/art/2023.102",
            "description": "A rearing cobra symbolizing divine power.",
        }
    ]
    prompt = build_rag_prompt("Tell me about Egyptian cobras", docs)
    assert "You are MuseGuide" in prompt
    assert "MUSEUM EVIDENCE:" in prompt
    assert "[Document 1]" in prompt
    assert "Tell me about Egyptian cobras" in prompt
    assert "YOUR GUIDED ANSWER" in prompt


def test_generator_grounded_answer(generator):
    """Verify answering a valid museum query produces a grounded response with citations."""
    response = generator.generate("Who was Nathaniel Hurd and what portrait is in the museum?")
    assert isinstance(response, RAGResponse)
    assert response.grounded is True
    assert len(response.answer) > 50
    assert "[1]" in response.answer

    # Verify structured citations
    assert len(response.citations) >= 1
    cit1 = response.citations[0]
    assert isinstance(cit1, Citation)
    assert cit1.citation_id == 1
    assert "Nathaniel Hurd" in cit1.title
    assert cit1.museum_url.startswith("https://")
    assert cit1.similarity_score > 0.40


def test_generator_anti_hallucination_refusal(generator):
    """Verify out-of-domain query is rejected with refusal template to prevent hallucination."""
    irrelevant_query = "What is the capital city of planet Mars and how was its government founded?"
    response = generator.generate(irrelevant_query)
    assert isinstance(response, RAGResponse)
    assert response.grounded is False
    assert "could not find sufficient historical evidence" in response.answer.lower()
    assert len(response.citations) == 0


def test_empty_query(generator):
    """Verify empty query is handled gracefully."""
    response = generator.generate("")
    assert response.grounded is False
    assert "Please enter a question" in response.answer


def test_citation_markers_match_structured_citations(generator):
    """Verify that [N] markers in the answer text strictly match the structured citation objects."""
    import re
    questions = [
        "Who was Nathaniel Hurd and what portrait of him is in the collection?",
        "What Ancient Egyptian artefacts with cobra or royal symbolism are in the museum?",
        "Are there French landscape paintings depicting ponds or water in the collection?",
    ]
    for q in questions:
        resp = generator.generate(q)
        assert resp.grounded is True
        text_markers = set(re.findall(r"\[(\d+)\]", resp.answer))
        citation_ids = {str(c.citation_id) for c in resp.citations}
        assert text_markers == citation_ids, (
            f"For query '{q}', markers in text {text_markers} do not match citation IDs {citation_ids}"
        )


def test_specific_artwork_does_not_add_unrelated_pieces(generator):
    """Verify that an artwork-specific query focuses on the target artwork and does not append unrelated pieces."""
    q = "Who was Nathaniel Hurd and what portrait of him is in the collection?"
    resp = generator.generate(q)
    assert resp.grounded is True
    # Should only cite the target portrait
    assert len(resp.citations) == 1
    assert "Nathaniel Hurd" in resp.citations[0].title
    assert "Gainsborough" not in resp.answer
    assert "Mildmay" not in resp.answer


def test_eval_questions_suite(generator):
    """Run all curated evaluation questions and verify expected grounding behavior."""
    import re
    eval_file = Path("tests/eval_questions.json")
    if not eval_file.exists():
        pytest.skip("eval_questions.json not found")

    with open(eval_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) >= 5, "Evaluation suite must contain at least 5 realistic questions"

    for case in cases:
        q = case["question"]
        expected_grounded = case["should_be_grounded"]
        resp = generator.generate(q)

        assert resp.grounded == expected_grounded, (
            f"Question '{q}' expected grounded={expected_grounded}, got {resp.grounded}"
        )

        if expected_grounded:
            assert len(resp.citations) >= 1
            # Check at least one expected keyword is present in the answer
            matched_keywords = [
                kw for kw in case["expected_keywords"]
                if kw.lower() in resp.answer.lower()
            ]
            assert len(matched_keywords) > 0, (
                f"For question '{q}', none of expected keywords {case['expected_keywords']} found in answer."
            )
            # Verify marker match
            text_markers = set(re.findall(r"\[(\d+)\]", resp.answer))
            citation_ids = {str(c.citation_id) for c in resp.citations}
            assert text_markers == citation_ids
        else:
            assert len(resp.citations) == 0


def test_portrait_of_dora_wheeler_regression(generator):
    """Regression test: Ensure 'Portrait of Dora Wheeler' is retrieved correctly,
    answers whether it is a painting or book using metadata, and is properly grounded.
    """
    q = "what is Portrait of Dora Wheeler, is it a book, a painting or something else?"
    resp = generator.generate(q)

    assert resp.grounded is True, "Answer must be grounded in verified museum record"
    assert len(resp.citations) == 1, "Should cite exactly one matching artwork"

    cit = resp.citations[0]
    assert cit.artwork_id == 102578, f"Expected ID 102578, got {cit.artwork_id}"
    assert "Portrait of Dora Wheeler" in cit.title
    assert "William Merritt Chase" in cit.creator
    assert "1921.1239" in cit.museum_url

    # Must directly answer whether it is a painting, book, or another object
    ans_lower = resp.answer.lower()
    assert "painting" in ans_lower, "Answer must state the artwork is a painting"
    assert "oil on canvas" in ans_lower, "Answer must cite the medium oil on canvas"
    assert "book" in ans_lower, "Answer should contrast with the user's question about a book"


def test_unrelated_named_entity_not_falsely_grounded(generator):
    """Ensure an answer is NEVER marked as grounded when the queried entity is absent from the museum."""
    q = "What is the Portrait of Ada Lovelace, is it a book or painting?"
    resp = generator.generate(q)

    # Must NOT claim grounding for an entity not in the collection
    assert resp.grounded is False, "Must not claim grounding for entity not in the collection"
    assert len(resp.citations) == 0, "Must not emit citations for ungrounded question"
    assert "could not find sufficient historical evidence" in resp.answer.lower()


def test_exact_artwork_titles_retrieval(searcher):
    """Verify that exact named artwork queries rank the exact matching artwork at Rank 1."""
    test_cases = [
        ("what is Portrait of Dora Wheeler, is it a book, a painting or something else?", 102578),
        ("Portrait of Isabella Brant", 125247),
        ("Pond at Ville-d'Avray", 124078),
    ]
    for q, expected_id in test_cases:
        results = searcher.search(q, top_k=3)
        assert len(results) > 0
        assert results[0]["id"] == expected_id, (
            f"Query '{q}' expected top result ID {expected_id}, but got {results[0]['id']} ('{results[0]['title']}')"
        )


def test_answer_no_boilerplate(generator):
    """Verify that answers do NOT contain repetitive boilerplate phrases."""
    queries = [
        "Who was Nathaniel Hurd and what portrait of him is in the collection?",
        "what is Portrait of Dora Wheeler, is it a book, a painting or something else?",
        "Are there French landscape paintings depicting ponds or water in the collection?",
    ]
    boilerplate_phrases = [
        "regarding your inquiry",
        "curatorial records document the work as follows",
        "in curatorial notes",
        "the collection features nathaniel hurd",
    ]
    for q in queries:
        resp = generator.generate(q)
        assert resp.grounded is True
        ans_lower = resp.answer.lower()
        for bp in boilerplate_phrases:
            assert bp not in ans_lower, f"Query '{q}' contains boilerplate '{bp}' in answer: {resp.answer}"


def test_answer_separated_from_metadata(generator):
    """Verify that metadata is cleanly separated into artwork_details rather than cluttering the answer."""
    resp = generator.generate("Who was Nathaniel Hurd and what portrait of him is in the collection?")
    assert resp.grounded is True
    assert hasattr(resp, "artwork_details")
    assert len(resp.artwork_details) == 1

    details = resp.artwork_details[0]
    assert details["artwork"] == "Nathaniel Hurd"
    assert "John Singleton Copley" in details["artist"]
    assert "Oil on canvas" in details["medium"] or "oil on canvas" in details["medium"].lower()
    assert "American Painting and Sculpture" in details["department"]

    # Direct answer starts immediately with who he was
    assert resp.answer.startswith("Nathaniel Hurd was a prominent silversmith")
    assert "[1]" in resp.answer


