"""Build FAISS vector index from processed museum records using Sentence Transformers."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress HF symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Crucial for Windows PyTorch + CUDA initialization
import torch
HAS_CUDA = torch.cuda.is_available()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging to write to stdout for clean PowerShell execution
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_index")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INPUT = "data/processed/cma_artworks.json"
DEFAULT_OUTPUT_DIR = "data/index"


def load_records(input_path: str) -> List[Dict[str, Any]]:
    """Load processed artwork records from JSON."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading records from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Loaded %d records", len(records))
    return records


def build_faiss_index(
    records: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate dense embeddings and build a FAISS cosine similarity index.

    Args:
        records: List of artwork dictionaries with 'retrieval_text'.
        model_name: HuggingFace sentence-transformer model name.
        output_dir: Directory to save index files.
        batch_size: Batch size for embedding inference.
        device: 'auto', 'cuda', 'cpu', or None.

    Returns:
        Dictionary summarizing index statistics.
    """
    if device is None or device == "auto":
        device = "cuda" if HAS_CUDA else "cpu"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract texts
    texts = [r.get("retrieval_text", "") for r in records]
    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count > 0:
        logger.warning("%d records have empty retrieval text!", empty_count)

    # 2. Load model
    logger.info("Loading SentenceTransformer model: %s (device=%s)", model_name, device)
    start_load = time.time()
    model = SentenceTransformer(model_name, device=device)
    logger.info("Model loaded in %.2f seconds", time.time() - start_load)

    # 3. Generate embeddings
    logger.info("Generating embeddings for %d records (batch_size=%d)...", len(texts), batch_size)
    start_embed = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization for cosine similarity with IndexFlatIP
    )
    embed_duration = time.time() - start_embed
    logger.info(
        "Embeddings generated in %.2f seconds (%.1f items/sec). Shape: %s, dtype: %s",
        embed_duration,
        len(texts) / max(embed_duration, 0.001),
        embeddings.shape,
        embeddings.dtype,
    )

    # Ensure float32 for FAISS
    embeddings = embeddings.astype(np.float32)
    dimension = embeddings.shape[1]

    # 4. Build FAISS index (IndexFlatIP for exact inner product / cosine similarity on unit vectors)
    logger.info("Building FAISS IndexFlatIP (dimension=%d)...", dimension)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("FAISS index populated. Total vectors: %d", index.ntotal)

    # 5. Save index, metadata, and config
    index_file = out_dir / "cma_index.faiss"
    metadata_file = out_dir / "cma_metadata.json"
    config_file = out_dir / "index_config.json"

    logger.info("Saving FAISS index to %s", index_file)
    faiss.write_index(index, str(index_file))

    # Save metadata matching row-to-id mapping
    logger.info("Saving metadata to %s", metadata_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Save configuration and stats
    config_data = {
        "model_name": model_name,
        "dimension": dimension,
        "total_records": index.ntotal,
        "metric": "inner_product_cosine",
        "index_type": "IndexFlatIP",
        "embedding_time_seconds": round(embed_duration, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_file": index_file.name,
        "metadata_file": metadata_file.name,
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    logger.info("Index build complete! Config saved to %s", config_file)
    return config_data


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS vector index for MuseGuide.")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to processed artworks JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save FAISS index and metadata (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', or 'cpu', default: 'auto')",
    )
    return parser.parse_args()


def main():
    import traceback
    try:
        args = parse_args()
        records = load_records(args.input)
        stats = build_faiss_index(
            records=records,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        print("\n--- FAISS Index Summary ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("\n[FATAL ERROR in build_index]:", e, file=sys.stdout)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
