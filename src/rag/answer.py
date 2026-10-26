"""Grounded Answer Generation pipeline with citations and anti-hallucination guardrails."""

import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.rag.prompts import REFUSAL_TEMPLATE, build_rag_prompt
from src.retrieval.search import SemanticSearcher

logger = logging.getLogger("rag_answer")


@dataclass
class Citation:
    """Structured citation metadata for an evidence document."""
    citation_id: int
    artwork_id: Any
    title: str
    creator: Optional[str]
    creation_date: Optional[str]
    culture: Optional[str]
    department: Optional[str]
    museum_url: str
    image_url: Optional[str]
    similarity_score: float
    evidence_snippet: str


@dataclass
class RAGResponse:
    """Full response package returned by the RAG pipeline."""
    query: str
    answer: str
    grounded: bool
    citations: List[Citation] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    model_used: str = "extractive_synthesizer"

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to standard dictionary."""
        data = asdict(self)
        return data


class AnswerGenerator:
    """Orchestrates semantic retrieval, relevance gating, and grounded answer synthesis with citations."""

    def __init__(
        self,
        searcher: Optional[SemanticSearcher] = None,
        min_confidence: float = 0.35,
        top_k: int = 4,
        llm_backend: str = "auto",
        model_name: Optional[str] = None,
    ):
        """Initialize RAG generator.

        Args:
            searcher: Instance of SemanticSearcher. If None, initialized with defaults.
            min_confidence: Minimum cosine similarity score required for top match to avoid hallucination.
            top_k: Number of evidence documents to retrieve.
            llm_backend: 'auto', 'api', 'local', or 'extractive'.
            model_name: Specific LLM model identifier if using api or local.
        """
        self.searcher = searcher if searcher is not None else SemanticSearcher()
        self.min_confidence = min_confidence
        self.top_k = top_k
        self.llm_backend = llm_backend
        self.model_name = model_name
        self._local_pipeline = None

        # Determine effective backend
        self._effective_backend = self._resolve_backend()
        logger.info("AnswerGenerator initialized with backend: %s", self._effective_backend)

    def _resolve_backend(self) -> str:
        """Resolve which generation backend to use based on available keys or config."""
        if self.llm_backend != "auto":
            return self.llm_backend

        # Check for API keys
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"

        # Default to robust extractive synthesizer (works 100% offline with zero dependencies)
        return "extractive"

    def generate(
        self,
        query: str,
        department: Optional[str] = None,
        culture: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> RAGResponse:
        """Generate a grounded answer for a natural language user query.

        Args:
            query: User's question.
            department: Optional department filter.
            culture: Optional culture filter.
            min_year: Optional earliest creation year filter.
            max_year: Optional latest creation year filter.

        Returns:
            RAGResponse object with answer, citations, and grounding status.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return RAGResponse(
                query=query,
                answer="Please enter a question about historical artefacts or museum collections.",
                grounded=False,
                model_used=self._effective_backend,
            )

        # 1. Semantic Retrieval
        retrieved = self.searcher.search(
            query=clean_query,
            top_k=self.top_k,
            department=department,
            culture=culture,
            min_year=min_year,
            max_year=max_year,
        )

        # 2. Anti-Hallucination Relevance Gating
        is_relevant = self._is_topically_relevant(clean_query, retrieved[0]) if retrieved else False
        if not retrieved or retrieved[0]["score"] < self.min_confidence or not is_relevant:
            top_score = retrieved[0]["score"] if retrieved else 0.0
            refusal_text = REFUSAL_TEMPLATE.format(query=clean_query, top_score=top_score)
            return RAGResponse(
                query=clean_query,
                answer=refusal_text,
                grounded=False,
                retrieved_documents=retrieved,
                model_used=self._effective_backend,
            )

        # 3. Answer Generation via resolved backend
        answer_text, citations, model_label = self._generate_answer(clean_query, retrieved)

        # Grounded is strictly True ONLY if citations exist AND evidence is topically relevant
        is_grounded = bool(citations) and is_relevant

        return RAGResponse(
            query=clean_query,
            answer=answer_text,
            grounded=is_grounded,
            citations=citations,
            retrieved_documents=retrieved,
            model_used=model_label,
        )

    def _is_topically_relevant(self, query: str, retrieved_doc: Dict[str, Any]) -> bool:
        """Verify that the retrieved museum record is genuinely relevant to the user's question.

        Never marks an answer as grounded merely because an unrelated record passed the score threshold.
        """
        stopwords = {
            "what", "who", "where", "when", "how", "why", "which", "is", "was",
            "are", "were", "the", "a", "an", "and", "or", "in", "of", "to", "for",
            "with", "by", "on", "at", "from", "about", "tell", "me", "collection",
            "museum", "art", "artefact", "artefacts", "artwork", "artworks", "piece",
            "pieces", "there", "have", "you", "does", "known", "into", "their",
            "something", "else", "it", "can", "show", "give", "any", "some"
        }
        generic_terms = {
            "portrait", "painting", "sculpture", "statue", "drawing", "print",
            "book", "image", "item", "artist", "period", "style", "view", "scene"
        }
        query_tokens = set(re.findall(r"\w+", query.lower()))
        distinctive_query_words = {
            w for w in query_tokens
            if len(w) >= 3 and w not in stopwords and w not in generic_terms
        }

        # If the query only consists of generic browsing terms (e.g. "show portraits"),
        # semantic similarity score threshold is sufficient.
        if not distinctive_query_words:
            return True

        # Check evidence text (title, creator, culture, description, department, technique)
        doc_text = " ".join([
            str(retrieved_doc.get("title") or ""),
            str(retrieved_doc.get("creator_name") or ""),
            str(retrieved_doc.get("culture") or ""),
            str(retrieved_doc.get("description") or ""),
            str(retrieved_doc.get("department") or ""),
            str(retrieved_doc.get("technique") or ""),
            str(retrieved_doc.get("type") or ""),
        ]).lower()
        doc_tokens = set(re.findall(r"\w+", doc_text))

        # Distinctive query concept requirement:
        # If the user asks a multi-concept question (>= 4 distinctive words, e.g. "capital city planet Mars government"),
        # matching only 1 isolated word while the main question concepts are completely absent is not genuine grounding.
        shared_distinctive = distinctive_query_words.intersection(doc_tokens)
        if len(distinctive_query_words) >= 4:
            return len(shared_distinctive) >= 2 or (len(shared_distinctive) / len(distinctive_query_words) >= 0.34)
        return len(shared_distinctive) >= 1

    def _generate_answer(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
    ) -> Tuple[str, List[Citation], str]:
        """Dispatch generation to the active backend."""
        backend = self._effective_backend

        if backend == "groq":
            try:
                answer = self._call_groq_api(query, retrieved)
                citations = self._extract_citations_from_text(answer, retrieved)
                return answer, citations, "groq"
            except Exception as e:
                logger.warning("Groq API call failed (%s); falling back to extractive synthesizer.", e)

        elif backend == "openai":
            try:
                answer = self._call_openai_api(query, retrieved)
                citations = self._extract_citations_from_text(answer, retrieved)
                return answer, citations, "openai"
            except Exception as e:
                logger.warning("OpenAI API call failed (%s); falling back to extractive synthesizer.", e)

        # Built-in high-quality deterministic synthesizer
        answer, citations = self._synthesize_extractive_answer(query, retrieved)
        return answer, citations, "extractive_synthesizer"

    def _synthesize_extractive_answer(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
    ) -> Tuple[str, List[Citation]]:
        """Synthesize an articulate museum docent response strictly from verified evidence.

        Only includes secondary artworks if they are meaningfully relevant to the specific inquiry.
        Maintains 1:1 correspondence between inline [N] citations and structured metadata.
        """
        stopwords = {
            "what", "who", "where", "when", "how", "why", "which", "is", "was",
            "are", "were", "the", "a", "an", "and", "or", "in", "of", "to", "for",
            "with", "by", "on", "at", "from", "about", "tell", "me", "collection",
            "museum", "art", "artefact", "artefacts", "artwork", "artworks", "piece",
            "pieces", "there", "have", "you", "does", "known", "into", "their"
        }
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        meaningful_query_words = {w for w in query_words if len(w) > 2 and w not in stopwords}

        # Check if the query is asking about a single specific artwork or person
        is_specific_inquiry = False
        specific_triggers = ["who was", "tell me about the portrait", "tell me about the painting",
                             "tell me about the statue", "what is the portrait", "portrait of",
                             "what does the", "who painted", "who created", "who made"]
        if any(trig in query_lower for trig in specific_triggers):
            is_specific_inquiry = True

        top_score = retrieved[0].get("score", 0.0)
        if len(retrieved) > 1:
            second_score = retrieved[1].get("score", 0.0)
            # A large score gap (> 0.12) strongly indicates a single dominant match
            if (top_score - second_score) > 0.12:
                is_specific_inquiry = True

        # 1. Select only meaningfully relevant artworks
        selected_artworks = [retrieved[0]]

        if not is_specific_inquiry and len(retrieved) > 1:
            for item in retrieved[1:]:
                item_score = item.get("score", 0.0)
                # Must satisfy strict minimum score and be close to top score
                if item_score < 0.52 or item_score < (top_score * 0.82):
                    continue

                item_desc = (item.get("description") or "").lower()
                item_title = (item.get("title") or "").lower()
                item_culture = (item.get("culture") or "").lower()
                item_text = f"{item_title} {item_desc} {item_culture}"

                # Must share non-trivial query keywords
                matched_kw = [kw for kw in meaningful_query_words if kw in item_text]
                if matched_kw:
                    selected_artworks.append(item)
                    if len(selected_artworks) >= 3:
                        break

        # 2. Build narrative paragraphs and corresponding structured citations
        paragraphs = []
        citations: List[Citation] = []

        for cit_idx, item in enumerate(selected_artworks, start=1):
            title = item.get("title") or f"Artefact {cit_idx}"
            creator = item.get("creator_name") or "an unknown artist"
            date_str = item.get("creation_date") or "an unrecorded date"
            culture = item.get("culture") or "its respective culture"
            dept = item.get("department") or "the museum collection"
            technique = item.get("technique")
            desc = item.get("description") or ""

            # Extract 1-2 most informative sentences
            raw_sentences = re.split(r"(?<=[.!?])\s+", desc)
            valid_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

            def sentence_relevance(s: str) -> int:
                swords = set(re.findall(r"\w+", s.lower()))
                return len(swords.intersection(meaningful_query_words))

            valid_sentences.sort(key=sentence_relevance, reverse=True)
            chosen_sentences = valid_sentences[:2] if valid_sentences else []
            body_text = " ".join(chosen_sentences)

            meta_details = []
            if technique:
                meta_details.append(f"executed in {technique.lower()}")
            if date_str and date_str != "an unrecorded date":
                meta_details.append(f"dating to {date_str}")
            meta_str = f" ({', '.join(meta_details)})" if meta_details else ""

            if cit_idx == 1:
                # Detect object classification and address classification questions directly
                raw_type = (item.get("type") or "").strip()
                category_keywords = {
                    "book", "painting", "sculpture", "statue", "drawing", "print",
                    "photograph", "manuscript", "textile", "tapestry", "vessel", "ceramic"
                }
                queried_cats = [c for c in category_keywords if c in query_lower]
                actual_cat = raw_type.lower() if raw_type else ""
                contrasting_cats = [c for c in queried_cats if c != actual_cat]

                technique_str = f"executed in {technique.lower()}" if technique else ""

                dept_lower = (dept or "").lower()
                dept_phrase = dept if any(w in dept_lower for w in ["department", "collection", "gallery", "art"]) else f"{dept} department"

                if actual_cat and contrasting_cats:
                    contrast_phrase = f", rather than a {', or '.join(contrasting_cats)}"
                    type_phrase = f"is a **{actual_cat}**"
                    if technique_str:
                        type_phrase += f" ({technique_str})"
                    type_phrase += contrast_phrase
                    intro = (
                        f"Regarding your inquiry, *{title}* [{cit_idx}] {type_phrase}. "
                        f"Created by {creator} ({date_str}) within the context of {culture}, "
                        f"this work is preserved in the {dept_phrase} [{cit_idx}]."
                    )
                elif actual_cat and any(kw in query_lower for kw in ["what is", "is it a", "type of", "kind of"]):
                    type_phrase = f"is a **{actual_cat}**"
                    if technique_str:
                        type_phrase += f" ({technique_str})"
                    intro = (
                        f"Regarding your inquiry, *{title}* [{cit_idx}] {type_phrase}. "
                        f"Created by {creator} ({date_str}) within the context of {culture}, "
                        f"this work is preserved in the {dept_phrase} [{cit_idx}]."
                    )
                else:
                    intro = (
                        f"Regarding your inquiry, the Cleveland Museum of Art collection features *{title}* [{cit_idx}]. "
                        f"Created by {creator} ({date_str}) within the context of {culture}, "
                        f"this work is preserved in the {dept_phrase} [{cit_idx}]."
                    )
                paragraphs.append(intro)
                if body_text:
                    paragraphs.append(f"Curatorial records document the work{meta_str} as follows: {body_text} [{cit_idx}]")
            else:
                creator_str = f" by {creator}" if creator and creator != "an unknown artist" else ""
                p = (
                    f"Another notable work in the collection is *{title}*{creator_str} [{cit_idx}], "
                    f"originating from {culture} ({date_str}): {body_text} [{cit_idx}]"
                )
                paragraphs.append(p)

            # Build matching structured citation
            snippet = (body_text[:150] + "...") if len(body_text) > 150 else (body_text or desc[:150])
            citations.append(
                Citation(
                    citation_id=cit_idx,
                    artwork_id=item.get("id"),
                    title=title,
                    creator=item.get("creator_name"),
                    creation_date=item.get("creation_date"),
                    culture=item.get("culture"),
                    department=item.get("department"),
                    museum_url=item.get("museum_url") or "",
                    image_url=item.get("image_url"),
                    similarity_score=item.get("score", 0.0),
                    evidence_snippet=snippet,
                )
            )

        full_answer = "\n\n".join(paragraphs)
        return full_answer, citations

    def _call_groq_api(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Call Groq Cloud API for generation."""
        import requests
        api_key = os.getenv("GROQ_API_KEY")
        model = self.model_name or "llama-3.1-8b-instant"
        prompt = build_rag_prompt(query, retrieved)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_openai_api(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        """Call OpenAI API for generation."""
        import requests
        api_key = os.getenv("OPENAI_API_KEY")
        model = self.model_name or "gpt-4o-mini"
        prompt = build_rag_prompt(query, retrieved)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _extract_citations_from_text(
        self,
        answer_text: str,
        retrieved: List[Dict[str, Any]],
    ) -> List[Citation]:
        """Extract [N] citation markers from LLM-generated text and map to retrieved documents."""
        matches = re.findall(r"\[(\d+)\]", answer_text)
        cited_indices = sorted(list(set(int(m) for m in matches)))

        citations: List[Citation] = []
        for doc_num in cited_indices:
            idx = doc_num - 1
            if 0 <= idx < len(retrieved):
                item = retrieved[idx]
                desc = item.get("description") or ""
                snippet = desc[:150] + "..." if len(desc) > 150 else desc

                cit = Citation(
                    citation_id=doc_num,
                    artwork_id=item.get("id"),
                    title=item.get("title") or "Untitled",
                    creator=item.get("creator_name"),
                    creation_date=item.get("creation_date"),
                    culture=item.get("culture"),
                    department=item.get("department"),
                    museum_url=item.get("museum_url") or "",
                    image_url=item.get("image_url"),
                    similarity_score=item.get("score", 0.0),
                    evidence_snippet=snippet,
                )
                citations.append(cit)

        return citations


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Query the MuseGuide Grounded Answer Generator.")
    parser.add_argument("query", type=str, nargs="?", default="Who was Nathaniel Hurd and what portrait is in the museum?", help="User question")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "extractive", "groq", "openai"], help="Generation backend")
    parser.add_argument("--top-k", type=int, default=4, help="Number of evidence documents to retrieve")
    parser.add_argument("--dept", type=str, default=None, help="Department filter")
    parser.add_argument("--culture", type=str, default=None, help="Culture filter")
    return parser.parse_args()


def main():
    args = parse_args()
    generator = AnswerGenerator(llm_backend=args.backend, top_k=args.top_k)
    print(f"\n=======================================================")
    print(f"QUESTION: {args.query}")
    print(f"BACKEND: {generator._effective_backend}")
    print(f"=======================================================\n")

    response = generator.generate(
        query=args.query,
        department=args.dept,
        culture=args.culture,
    )

    print("GUIDED ANSWER:\n")
    print(response.answer)
    print("\n-------------------------------------------------------")
    print(f"GROUNDED: {response.grounded} | MODEL: {response.model_used}")
    print(f"CITATIONS ({len(response.citations)}):")
    for cit in response.citations:
        print(f"  [{cit.citation_id}] {cit.title} ({cit.creation_date or 'Unknown Date'})")
        print(f"      Creator: {cit.creator or 'Unknown'} | Culture: {cit.culture or 'N/A'}")
        print(f"      URL: {cit.museum_url}")
        print(f"      Similarity Score: {cit.similarity_score:.4f}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
