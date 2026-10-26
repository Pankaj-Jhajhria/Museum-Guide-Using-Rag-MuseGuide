import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress HF symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Crucial for Windows PyTorch + CUDA initialization
import torch
HAS_CUDA = torch.cuda.is_available()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("semantic_search")


class SemanticSearcher:
    """Semantic search engine that queries a FAISS vector index with natural-language text."""

    def __init__(
        self,
        index_path: str = "data/index/cma_index.faiss",
        metadata_path: str = "data/index/cma_metadata.json",
        model_name: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize the search engine with FAISS index, metadata, and embedding model.

        Args:
            index_path: Path to .faiss file.
            metadata_path: Path to metadata JSON file.
            model_name: SentenceTransformer model name. If None, loaded from index_config.json if available.
            device: 'auto', 'cuda', or 'cpu'.
        """
        if device == "auto":
            device = "cuda" if HAS_CUDA else "cpu"

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.device = device

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        # Load FAISS index
        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata records
        logger.info("Loading metadata records from %s", self.metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        if self.index.ntotal != len(self.metadata):
            logger.warning(
                "Index count (%d) does not match metadata count (%d)",
                self.index.ntotal,
                len(self.metadata),
            )

        # Build fast ID lookup
        self.id_to_index: Dict[Any, int] = {}
        for idx, item in enumerate(self.metadata):
            item_id = item.get("id")
            if item_id is not None:
                self.id_to_index[item_id] = idx
                self.id_to_index[str(item_id)] = idx

        # Determine model name
        if model_name is None:
            config_file = self.index_path.parent / "index_config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    model_name = config.get("model_name", "all-MiniLM-L6-v2")
            else:
                model_name = "all-MiniLM-L6-v2"

        self.model_name = model_name
        logger.info("Loading embedding model %s on %s", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query string into a normalized 2D float32 numpy vector."""
        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Lowercase and strip punctuation to alphanumeric tokens."""
        return re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()

    def _compute_lexical_boost(self, query: str, record: Dict[str, Any]) -> float:
        """Compute an exact title / entity match score boost without breaking semantic search."""
        q_clean = self._clean_text(query)
        q_words = set(q_clean.split())
        if not q_words:
            return 0.0

        title = record.get("title") or ""
        t_clean = self._clean_text(title)
        if not t_clean:
            return 0.0

        boost = 0.0
        t_words = t_clean.split()

        # 1. Exact full title match (contiguous substring or exact match)
        # e.g., "portrait of dora wheeler" in "what is portrait of dora wheeler..."
        # If title is a single short word (e.g. "Mars"), only boost if the query is focused (<= 4 words)
        if len(t_clean) >= 4 and (f" {t_clean} " in f" {q_clean} " or t_clean == q_clean):
            if len(t_words) >= 2 or len(q_words) <= 4:
                boost += 0.20

        # 2. Distinctive title words match (requires >= 2 distinctive words or focused query)
        generic_words = {
            "portrait", "of", "a", "an", "the", "in", "with", "and", "or", "by", "for",
            "painting", "sculpture", "figure", "study", "sketch", "view", "scene", "head",
            "composition", "drawing", "plate", "print", "statue", "vase", "from", "near",
            "after", "unknown", "untitled", "depicting", "representation", "storage", "vessel"
        }
        distinctive_t_words = [w for w in t_words if len(w) >= 3 and w not in generic_words]
        if len(distinctive_t_words) >= 2:
            matched_words = [w for w in distinctive_t_words if w in q_words]
            match_ratio = len(matched_words) / len(distinctive_t_words)
            if match_ratio == 1.0:
                boost += 0.10
            elif match_ratio >= 0.5:
                boost += 0.05
        elif len(distinctive_t_words) == 1 and len(q_words) <= 4:
            if distinctive_t_words[0] in q_words:
                boost += 0.05

        # 3. Creator match
        creator = record.get("creator_name") or ""
        c_clean = self._clean_text(creator)
        c_words = [w for w in c_clean.split() if len(w) >= 4 and w not in generic_words]
        if c_words:
            matched_c = [w for w in c_words if w in q_words]
            if len(matched_c) >= 2:
                boost += 0.08
            elif len(matched_c) == 1:
                boost += 0.03

        return round(boost, 4)

    def search(
        self,
        query: str,
        top_k: int = 5,
        department: Optional[str] = None,
        culture: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most semantically relevant artworks for a text query.

        Args:
            query: Natural language question or search query.
            top_k: Number of results to return.
            department: Filter by department name (case-insensitive substring).
            culture: Filter by culture name (case-insensitive substring).
            min_year: Filter by minimum creation year (earliest).
            max_year: Filter by maximum creation year (latest).
            min_score: Minimum cosine similarity score cutoff.

        Returns:
            List of result dictionaries with 'rank', 'score', and artwork metadata.
        """
        if not query or not query.strip():
            return []

        query_vec = self.encode_query(query.strip())

        # Overfetch candidates to enable exact title/entity re-ranking and compound filtering
        has_filters = any([department, culture, min_year is not None, max_year is not None])
        fetch_k = min(self.index.ntotal, top_k * 10 if has_filters else max(top_k * 6, 40))

        scores, indices = self.index.search(query_vec, fetch_k)

        candidates: List[Dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            score_val = float(score)
            record = self.metadata[idx]

            # Apply filters
            if department:
                dept_val = record.get("department") or ""
                if department.lower() not in dept_val.lower():
                    continue

            if culture:
                cult_val = record.get("culture") or ""
                if culture.lower() not in cult_val.lower():
                    continue

            if min_year is not None:
                earliest = record.get("creation_date_earliest")
                if earliest is not None and earliest < min_year:
                    continue

            if max_year is not None:
                latest = record.get("creation_date_latest")
                if latest is not None and latest > max_year:
                    continue

            # Compute entity / exact title boost
            lex_boost = self._compute_lexical_boost(query, record)
            final_score = min(1.0, score_val + lex_boost)

            if min_score is not None and final_score < min_score:
                continue

            candidates.append({
                "record": record,
                "score": final_score,
                "base_score": score_val,
                "boost": lex_boost,
            })

        # Re-rank candidates by final score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, cand in enumerate(candidates[:top_k], start=1):
            record = cand["record"]
            result = {
                "rank": rank,
                "score": round(cand["score"], 4),
                "base_score": round(cand["base_score"], 4),
                "lexical_boost": round(cand["boost"], 4),
                "id": record.get("id"),
                "title": record.get("title"),
                "creator_name": record.get("creator_name"),
                "creation_date": record.get("creation_date"),
                "creation_date_earliest": record.get("creation_date_earliest"),
                "creation_date_latest": record.get("creation_date_latest"),
                "culture": record.get("culture"),
                "technique": record.get("technique"),
                "department": record.get("department"),
                "type": record.get("type"),
                "tombstone": record.get("tombstone"),
                "current_location": record.get("current_location"),
                "museum_url": record.get("museum_url"),
                "image_url": record.get("image_url"),
                "description": record.get("description"),
                "retrieval_text": record.get("retrieval_text"),
            }
            results.append(result)

        return results

    def find_similar(
        self,
        artwork_id: Any,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find visually or thematically related artworks using the target artwork's vector.

        Args:
            artwork_id: Museum artwork ID.
            top_k: Number of similar artworks to return.
            exclude_self: Whether to omit the query artwork itself from results.

        Returns:
            List of similar artwork dictionaries.
        """
        idx = self.id_to_index.get(artwork_id)
        if idx is None:
            # Try string conversion
            idx = self.id_to_index.get(str(artwork_id))

        if idx is None:
            raise KeyError(f"Artwork ID {artwork_id} not found in index metadata.")

        # Reconstruct vector directly from FAISS IndexFlatIP
        try:
            vec = self.index.reconstruct(idx).reshape(1, -1).astype(np.float32)
        except Exception:
            # Fallback if reconstruct is not supported by some custom index types
            text = self.metadata[idx].get("retrieval_text", "")
            vec = self.encode_query(text)

        fetch_k = top_k + 1 if exclude_self else top_k
        scores, indices = self.index.search(vec, fetch_k)

        results: List[Dict[str, Any]] = []
        rank = 1

        for score, candidate_idx in zip(scores[0], indices[0]):
            if candidate_idx < 0 or candidate_idx >= len(self.metadata):
                continue
            if exclude_self and candidate_idx == idx:
                continue

            record = self.metadata[candidate_idx]
            result = {
                "rank": rank,
                "score": round(float(score), 4),
                "id": record.get("id"),
                "title": record.get("title"),
                "creator_name": record.get("creator_name"),
                "creation_date": record.get("creation_date"),
                "culture": record.get("culture"),
                "technique": record.get("technique"),
                "department": record.get("department"),
                "museum_url": record.get("museum_url"),
                "image_url": record.get("image_url"),
                "description": record.get("description"),
                "retrieval_text": record.get("retrieval_text"),
            }
            results.append(result)
            rank += 1

            if len(results) >= top_k:
                break

        return results

    def get_artwork_by_id(self, artwork_id: Any) -> Optional[Dict[str, Any]]:
        """Get full record by ID."""
        idx = self.id_to_index.get(artwork_id)
        if idx is None:
            idx = self.id_to_index.get(str(artwork_id))
        if idx is not None:
            return self.metadata[idx]
        return None

    def get_departments(self) -> List[str]:
        """Get unique departments sorted alphabetically."""
        depts = {r.get("department") for r in self.metadata if r.get("department")}
        return sorted(list(depts))

    def get_cultures(self) -> List[str]:
        """Get unique cultures sorted alphabetically."""
        cults = {r.get("culture") for r in self.metadata if r.get("culture")}
        return sorted(list(cults))


def parse_args():
    parser = argparse.ArgumentParser(description="Query the MuseGuide semantic search index.")
    parser.add_argument("query", type=str, nargs="?", default="", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--dept", type=str, default=None, help="Filter by department")
    parser.add_argument("--culture", type=str, default=None, help="Filter by culture")
    parser.add_argument("--similar-to", type=str, default=None, help="Find artworks similar to artwork ID")
    parser.add_argument("--index-path", type=str, default="data/index/cma_index.faiss")
    parser.add_argument("--metadata-path", type=str, default="data/index/cma_metadata.json")
    return parser.parse_args()


def main():
    args = parse_args()
    searcher = SemanticSearcher(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
    )

    if args.similar_to:
        print(f"\n--- Finding Artworks Similar to ID {args.similar_to} ---")
        target = searcher.get_artwork_by_id(args.similar_to)
        if target:
            print(f"Target: \"{target.get('title')}\" by {target.get('creator_name') or 'Unknown'}\n")
        results = searcher.find_similar(args.similar_to, top_k=args.top_k)
    else:
        query = args.query or "Ancient Egyptian bronze sculpture or gold jewelry"
        print(f"\n--- Semantic Search: \"{query}\" ---")
        results = searcher.search(
            query=query,
            top_k=args.top_k,
            department=args.dept,
            culture=args.culture,
        )

    for r in results:
        print(f"#{r['rank']} [Score: {r['score']:.4f}] {r['title']}")
        print(f"   Creator: {r['creator_name'] or 'Unknown'} | Date: {r['creation_date'] or 'Unknown'}")
        print(f"   Culture: {r['culture'] or 'N/A'} | Dept: {r['department'] or 'N/A'}")
        desc = (r['description'] or '')[:120].replace('\n', ' ')
        print(f"   Snippet: {desc}...")
        print(f"   URL: {r['museum_url']}")
        print()


if __name__ == "__main__":
    main()
