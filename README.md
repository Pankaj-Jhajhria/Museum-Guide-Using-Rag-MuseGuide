# 🏛️ MuseGuide — AI Museum Guide Using RAG

An intelligent, portfolio-grade museum exploration application powered by **Retrieval-Augmented Generation (RAG)**, dense vector search, and verified evidence citations.

MuseGuide indexes 2,000 public-domain artworks from the **Cleveland Museum of Art (CMA)** Open Access collection, allowing users to browse high-resolution artefacts, perform natural-language semantic searches, and receive authoritative curatorial answers strictly grounded in museum records.

---

## 🌟 Key Features

* **🏛️ AI Museum Docent (RAG)**: Natural-language Q&A backed by curatorial evidence with inline `[1]`, `[2]` citations.
* **🛡️ Anti-Hallucination Guardrails**: Out-of-domain and unsupported inquiries (e.g., questions about sci-fi, modern sports, or unrecorded topics) trigger an honest refusal rather than fabricating facts.
* **🔗 Structured Verified Citations**: Every factual claim links to official museum accession records (`https://clevelandart.org/art/...`) with high-resolution image thumbnails and relevance scores.
* **🔍 Semantic Vector Search**: Dense semantic retrieval powered by Sentence Transformers and FAISS cosine similarity (`IndexFlatIP`).
* **✨ Related Artefact Discovery**: Explores conceptual and stylistic sister works using vector-space nearest neighbors (`find_similar`).
* **🎨 Interactive Gallery & Compound Filters**: Explores the 2,000-artefact collection by department, culture, medium, and date range.
* **⚡ 100% Offline-Capable**: Built-in deterministic curatorial docent synthesizer generates grounded answers locally without requiring external API keys or heavy GPU downloads.

---

## 🏗️ System Architecture

```
  ┌─────────────────────────────────────────────────────────────┐
  │         1. DATA ACQUISITION & QUALITY AUDIT (CMA API)        │
  │   - 2,000 CC0 Museum Records (100% Imaged, 100% Described) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 2. TEXT CHUNKING & EMBEDDINGS               │
  │   - Structured Curatorial Evidence Document                 │
  │   - Sentence Transformers: all-MiniLM-L6-v2 (384-dim)       │
  │   - Normalization: L2 unit vectors for exact cosine sim     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 3. FAISS VECTOR INDEXING                    │
  │   - IndexFlatIP (2,000 vectors, exact inner product)        │
  │   - Sub-millisecond retrieval latency (~15 ms steady-state) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                     User Natural-Language Query
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          4. RELEVANCE GATING & INTENT CLASSIFICATION        │
  │   - Cosine threshold: min_confidence = 0.35                 │
  │   - Artwork-Specific vs. Thematic Inquiry Distinction       │
  └──────────────┬───────────────────────────────┬──────────────┘
                 │ (Score < 0.35)                │ (Score >= 0.35)
                 ▼                               ▼
     ┌───────────────────────┐       ┌───────────────────────┐
     │ 5a. HONEST REFUSAL    │       │ 5b. ANSWER GENERATOR  │
     │ "Insufficient evidence│       │ - Extractive Docent   │
     │  in museum collection"│       │   (Default Offline)   │
     └───────────────────────┘       │ - Cloud API Adapters  │
                                     │   (Groq / OpenAI)     │
                                     └───────────┬───────────┘
                                                 │
                                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         6. CITATION VERIFICATION & 1:1 MARKER MAPPING       │
  │   - Strict 1:1 alignment between text [N] and Citation objs │
  │   - Official CMA portal links & thumbnail verification      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          7. INTERACTIVE STREAMLIT WEB APP (Port 8501)       │
  │   - AI Docent Tab • Semantic Search • Gallery • Analytics   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 💻 Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.13 | High performance, modern typing, robust ecosystem |
| **Data Engine** | Pandas & NumPy | Fast data transformation and array vector math |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Compact (384-dim), fast CPU/GPU inference, top semantic benchmark score |
| **Vector Index** | `faiss-cpu` (`IndexFlatIP`) | Exact inner product (cosine similarity), zero recall loss, fast lookup |
| **UI Framework** | Streamlit | Rapid interactive web interface with native layout & data visualization |
| **Data Source** | Cleveland Museum of Art API | 64,000+ open access artworks, CC0 public domain licensing |
| **Testing** | `pytest` | 16 automated unit, integration, and RAG regression tests |

---

## 🚀 Quick Start Guide (Windows / Linux / macOS)

### 1. Clone & Setup Virtual Environment

```powershell
# Open terminal / PowerShell in project root
cd MuseGuide

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# (Linux / macOS)
# source venv/bin/activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Pipeline Execution (End-to-End)

If starting from scratch, the full pipeline can be reproduced with:

```powershell
# Step 1: Fetch raw records from Cleveland Museum of Art Open Access API
python src/data/fetch_cma.py --target 2000 --limit 1000 --delay 1.0

# Step 2: Clean, validate, and structure retrieval documents
python src/data/preprocess.py

# Step 3: Generate dense embeddings and construct FAISS vector index
python src/retrieval/build_index.py
```

*Note: The repository includes the curated, audited museum dataset in `data/processed/` and test samples. Because binary vector indexes are excluded from version control, build the local FAISS index once prior to launching the app (takes ~15 seconds on CPU/GPU):*

```powershell
python src/retrieval/build_index.py
```

### 4. Launch the Streamlit Web Application

```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your web browser.

---

## 🖥️ Command-Line Verification & Testing

### Run Semantic Search via CLI
```powershell
# Natural language semantic query
python src/retrieval/search.py "Ancient Egyptian bronze sculpture or gold jewelry" --top-k 3

# Discover related artworks by ID
python src/retrieval/search.py --similar-to 109291 --top-k 3
```

### Run Grounded Answer Generation (RAG) via CLI
```powershell
# Artwork-specific inquiry
python src/rag/answer.py "Who was Nathaniel Hurd and what portrait of him is in the collection?"

# Thematic inquiry
python src/rag/answer.py "What Ancient Egyptian artefacts with cobra or royal symbolism are in the museum?"

# Anti-hallucination refusal test (unsupported question)
python src/rag/answer.py "What is the capital city of planet Mars and how was its government founded?"
```

### Run Automated PyTest Suite
```powershell
python -m pytest tests/ -v
```

### Run RAG Benchmark Evaluation
```powershell
python tests/evaluate_rag.py
```

---

## 🔬 Evaluation Methodology & Measured Results

The RAG and retrieval engines were rigorously evaluated against a manually verified benchmark dataset ([`tests/benchmark_dataset.json`](tests/benchmark_dataset.json)) comprising:
1. **Specific Artwork Inquiries**: Questions tied to specific ground-truth artwork IDs (e.g. Copley's *Nathaniel Hurd*, Monet's *Water Lilies*, Corot's *Pond at Ville-d'Avray*).
2. **Thematic & Stylistic Queries**: Inquiries testing multi-document retrieval across Greek pottery, Egyptian funerary objects, and Chola bronzes.
3. **Negative / Unsupported Queries**: Out-of-domain questions with no museum counterpart testing anti-hallucination refusal.

### Quantitative Metrics (Recorded via `evaluate_rag.py`):

| Evaluation Dimension | Metric | Measured Result | Target Benchmark | Status |
|---|---|:---:|:---:|:---:|
| **Retrieval Accuracy** | **Recall@5** | **100.0%** (9/9) | $\ge 90.0\%$ | ✅ Exceeded |
| **Retrieval Quality** | **Mean Reciprocal Rank (MRR)** | **0.944** | $\ge 0.850$ | ✅ Exceeded |
| **Warm Retrieval Speed** | **Steady-State Retrieval Latency** | **14.7 ms** | $< 50\text{ ms}$ | ✅ Exceeded |
| **Cold-Start Latency** | **First-Query PyTorch/CUDA Warmup** | **347.3 ms** | $< 1000\text{ ms}$ | ✅ Fast |
| **Citation Integrity** | **Official CMA URL Validity** | **100.0%** (13/13) | $100.0\%$ | ✅ Perfect |
| **Citation Alignment** | **1:1 Text-Marker to Citation Match** | **100.0%** (9/9) | $100.0\%$ | ✅ Perfect |
| **Safety / Guardrails** | **Negative Refusal Accuracy** | **100.0%** (3/3) | $100.0\%$ | ✅ Perfect |
| **Warm End-to-End Latency** | **Full Generation Time (Steady-State)** | **14.8 ms** | $< 100\text{ ms}$ | ✅ Exceeded |

> [!NOTE]
> **Latency Methodology Clarification**:
> Initial naive benchmarks reported an average retrieval latency of **44.4 ms** alongside an average end-to-end latency of **11.6 ms**. This discrepancy occurred because the very first query incurred a **~206–258 ms** PyTorch CUDA JIT kernel compilation and VRAM allocation penalty (cold start). Averaged over 9 queries without pre-warming, that single spike skewed retrieval latency to 44.4 ms. By contrast, the subsequent end-to-end test ran *after* the GPU was already warm.
>
> In the updated 3-stage benchmark ([`tests/evaluate_rag.py`](tests/evaluate_rag.py)), cold start is explicitly isolated from warm steady state:
> * **Cold-Start (First Query)**: 347.3 ms (model weight loading and CUDA graph initialization).
> * **Steady-State Retrieval**: 14.7 ms (embedding computation + FAISS exact IP scan).
> * **Steady-State End-to-End**: 14.8 ms (retrieval + relevance gating + docent synthesis + citation formatting).
> * **Synthesis Overhead**: < 0.5 ms (in-memory extractive synthesis adds negligible overhead over raw retrieval).

---

## 🧪 Verification: Automated Checks vs. Manual Browser Testing

To ensure full transparency and auditability, MuseGuide distinguishes automated regression tests from visual UI checks:

### Automated Checks (Verified & Reproducible via CI/CLI):
* **Data Integrity**: 2,000 CC0 museum artworks audited with 0 missing images and 0 missing descriptions.
* **Vector Index**: 2,000 FAISS vectors indexed with normalized L2 cosine similarity (`IndexFlatIP`).
* **Unit & Regression Suite**: 16 automated tests passing in `tests/test_rag.py` and `tests/test_retrieval.py` (`python -m pytest tests/ -v`).
* **Evaluation Suite**: 12 benchmark cases (9 positive + 3 negative) validating Recall@5, MRR, 1:1 citation marker parity, and out-of-domain refusals (`python tests/evaluate_rag.py`).
* **HTTP Server Health**: Streamlit web server launches cleanly on `http://localhost:8501` returning HTTP 200 OK.

### Manual Browser Interactions (Requires Human Visual Testing):
* **Interactive UI Inspection**: Open `http://localhost:8501` in your web browser.
* **Tab Navigation**: Click through the 4 tabs: *🏛️ AI Museum Docent*, *🔍 Semantic Search*, *🖼️ Artwork Gallery*, and *📊 Collection Analytics*.
* **Filter Interactions**: Test dropdowns for Department, Culture, and the Century slider in the Gallery and Search tabs.
* **Related Artefact Drawer**: Click on an artwork card to expand its high-resolution view and view dynamically retrieved sister artworks.
* **External Citation Verification**: Click citation badges (e.g. `[1] Nathaniel Hurd`) to confirm they open the live Cleveland Museum of Art web portal.

---

## 🏛️ Dataset Source & Licensing

* **Institution**: The Cleveland Museum of Art (Cleveland, Ohio, USA).
* **Dataset Scope**: 2,000 public-domain artworks selected for curatorial description density and high-resolution imaging.
* **License**: **Creative Commons Zero (CC0)** — Public Domain Dedication. Artwork metadata and digital images can be shared, modified, and explored freely without permission.
* **API Documentation**: [Cleveland Museum of Art Open Access API](https://openaccess-api.clevelandart.org/)

---

## ⚙️ How the Offline Generator Works

To eliminate third-party API dependencies and guarantee 100% reproducibility in academic or portfolio grading environments, MuseGuide features an **Offline Curatorial Extractive Synthesizer**:

1. **Relevance Gating**: Queries with top retrieval score below `min_confidence = 0.35` are rejected with a helpful refusal message.
2. **Specific vs. Thematic Intent Gating**:
   * If a user asks about a specific person or artwork (e.g. *"Who was Nathaniel Hurd..."*), secondary retrieved documents (e.g. unrelated landscapes) are **suppressed**, preventing hallucinated connections.
   * If a user asks a thematic question (e.g. *"What Egyptian cobras are in the museum?"*), secondary works are only included if they pass strict score thresholds (`score >= 0.52`, `score >= top_score * 0.82`) and share key query entities.
3. **Curatorial Synthesis**: Extracts the most informative descriptive propositions from verified museum notes, formats them in a docent narrative, and maps every proposition to an exact `[1]`, `[2]` citation.
4. **Structured Citation Pairing**: Citation numbers in text match the structured `Citation` metadata objects 1:1, providing direct clickable links to official museum records.
5. **Optional Cloud LLM Mode**: If `GROQ_API_KEY` or `OPENAI_API_KEY` is set in `.env`, the pipeline automatically switches to cloud LLM generation while preserving the same evidence grounding and citation pipeline.

---

## ⚠️ Known Limitations & Future Roadmap

* **Collection Scope**: MuseGuide indexes 2,000 richly described CC0 works out of CMA's 64,000+ open access collection to keep local embedding storage compact (~15 MB). The data pipeline can scale to the entire collection with hierarchical FAISS indexes (e.g., `IndexIVFFlat` or `IndexHNSWFlat`).
* **Text-Centric Embeddings**: Retrieval currently operates on multi-field text records (title, creator, culture, date, medium, department, description). Direct visual image-to-image or text-to-image retrieval using multi-modal embeddings (e.g., CLIP / SigLIP) is planned for future iterations.
* **Stateless Q&A**: Each docent inquiry is answered independently. Multi-turn conversational memory with coreference resolution (e.g. *"Where was he born?"* following up on Copley) is a planned enhancement.
* **Single Institution Context**: Currently focused on the Cleveland Museum of Art CC0 Open Access collection; federating across other open collections (e.g., The Met, Rijksmuseum, Smithsonian) would broaden cross-cultural comparisons.
* **Headless Sandbox Limitations**: Automated browser interaction tests via Playwright/Selenium were skipped in this environment due to missing browser binaries in the Windows sandbox; all functional validation relies on pytest, HTTP endpoint probes, and documented manual testing steps.

---

## 📜 License

* Application source code: **MIT License**
* Museum metadata & images: **CC0 1.0 Universal (Public Domain Dedication)**
