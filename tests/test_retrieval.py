import json
import shutil
import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.build_index import build_faiss_index, load_records
from src.retrieval.search import SemanticSearcher


@pytest.fixture(scope="module")
def sample_records():
    """Load sample 50 records for quick testing."""
    sample_path = Path("data/processed/cma_sample_50.json")
    if not sample_path.exists():
        pytest.skip("Sample dataset not found at data/processed/cma_sample_50.json")
    return load_records(str(sample_path))


@pytest.fixture(scope="module")
def sample_index_dir(sample_records):
    """Build a temporary index from 50 sample records."""
    temp_dir = tempfile.mkdtemp()
    build_faiss_index(
        records=sample_records,
        model_name="all-MiniLM-L6-v2",
        output_dir=temp_dir,
        batch_size=32,
        device="cpu",
    )
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_files_created(sample_index_dir):
    """Verify all index artifacts are created correctly."""
    index_path = Path(sample_index_dir) / "cma_index.faiss"
    meta_path = Path(sample_index_dir) / "cma_metadata.json"
    config_path = Path(sample_index_dir) / "index_config.json"

    assert index_path.exists(), "FAISS index file missing"
    assert meta_path.exists(), "Metadata file missing"
    assert config_path.exists(), "Index config file missing"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["total_records"] == 50
    assert cfg["dimension"] == 384
    assert cfg["metric"] == "inner_product_cosine"


def test_semantic_search_basic(sample_index_dir):
    """Verify basic semantic search returns ranked records."""
    searcher = SemanticSearcher(
        index_path=str(Path(sample_index_dir) / "cma_index.faiss"),
        metadata_path=str(Path(sample_index_dir) / "cma_metadata.json"),
    )

    results = searcher.search("oil painting portrait", top_k=3)
    assert len(results) == 3
    for r in results:
        assert "rank" in r
        assert "score" in r
        assert "title" in r
        assert "retrieval_text" in r
        assert isinstance(r["score"], float)

    # Check scores are sorted descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Scores should be sorted descending"


def test_find_similar_artworks(sample_index_dir, sample_records):
    """Verify artefact-to-artefact similarity search."""
    searcher = SemanticSearcher(
        index_path=str(Path(sample_index_dir) / "cma_index.faiss"),
        metadata_path=str(Path(sample_index_dir) / "cma_metadata.json"),
    )

    target_id = sample_records[0]["id"]
    similar = searcher.find_similar(target_id, top_k=3, exclude_self=True)

    assert len(similar) <= 3
    # Ensure query item is not in its own similar results
    assert all(r["id"] != target_id for r in similar), "Target artwork should not be in its own similar results"


def test_empty_query_returns_empty(sample_index_dir):
    """Verify empty or whitespace query returns empty list gracefully."""
    searcher = SemanticSearcher(
        index_path=str(Path(sample_index_dir) / "cma_index.faiss"),
        metadata_path=str(Path(sample_index_dir) / "cma_metadata.json"),
    )
    assert searcher.search("") == []
    assert searcher.search("   ") == []


def test_search_with_department_filter(sample_index_dir):
    """Verify search filtering by department."""
    searcher = SemanticSearcher(
        index_path=str(Path(sample_index_dir) / "cma_index.faiss"),
        metadata_path=str(Path(sample_index_dir) / "cma_metadata.json"),
    )

    depts = searcher.get_departments()
    if depts:
        target_dept = depts[0]
        results = searcher.search("art", top_k=5, department=target_dept)
        for r in results:
            assert target_dept.lower() in (r.get("department") or "").lower()
