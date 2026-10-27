# 🏛️ MuseGuide — AI Museum Guide Using RAG

An intelligent, portfolio-grade museum exploration application powered by **Retrieval-Augmented Generation (RAG)**, dense vector search, and verified evidence citations.

MuseGuide indexes 2,000 public-domain artworks from the **Cleveland Museum of Art (CMA)** Open Access collection, allowing users to browse high-resolution artefacts, perform natural-language semantic searches, and receive authoritative curatorial answers strictly grounded in museum records.

---

## 🌟 Key Features

* **🏛️ AI Museum Docent (RAG)**: Natural-language Q&A backed by curatorial evidence with inline `[1]`, `[2]` citations.
* **⚡ Dual Answer Generator Modes**:
  * **Groq Cloud LLM (Natural & Conversational)**: High-speed grounded answer generation powered by Groq's LPU inference (`qwen/qwen3.8-27b` or custom `GROQ_MODEL`) delivering direct, conversational docent answers free of repetitive catalog boilerplate.
  * **Offline Extractive Synthesizer (100% Offline)**: Local deterministic synthesizer requiring zero API keys or external dependencies, serving as a reliable fallback whenever offline or when an API key is not configured.
* **🎙️ Optional Voice Input (Groq Whisper)**: Ask museum questions by speaking directly into your microphone, with automatic speech-to-text powered by Groq Whisper (`whisper-large-v3-turbo`).
* **🔊 Manual Text-to-Speech (Local Piper & pyttsx3)**: Listen to direct answers with a dedicated `🔊 Speak Answer` button. Speech is generated locally on CPU using Piper ONNX neural voice with zero external audio APIs, citations cleanly omitted from speech, and zero autoplay.
* **🛡️ Anti-Hallucination Guardrails**: Out-of-domain and unsupported inquiries (e.g., questions about sci-fi, modern sports, or unrecorded topics) trigger an honest refusal rather than fabricating facts.
* **🔗 Structured Verified Citations**: Every factual claim links to official museum accession records (`https://clevelandart.org/art/...`) with high-resolution image thumbnails and relevance scores. Fabricated or out-of-range citation markers are strictly rejected.
* **🔍 Semantic Vector Search**: Dense semantic retrieval powered by Sentence Transformers and FAISS cosine similarity (`IndexFlatIP`).
* **✨ Related Artefact Discovery**: Explores conceptual and stylistic sister works using vector-space nearest neighbors (`find_similar`).
* **🎨 Interactive Gallery & Compound Filters**: Explores the 2,000-artefact collection by department, culture, medium, and date range.
* **📋 3-Tier Visual Response Hierarchy**: Clean separation between Direct Answer, Supporting Artwork Details, and Curatorial Sources & Citations.

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
  │   - Topical Keyword & Entity Overlap Verification           │
  └──────────────┬───────────────────────────────┬──────────────┘
                 │ (Score < 0.35 or Irrelevant)  │ (Score >= 0.35 & Relevant)
                 ▼                               ▼
     ┌───────────────────────┐       ┌─────────────────────────────────────┐
     │ 5a. HONEST REFUSAL    │       │ 5b. DUAL-MODE ANSWER GENERATOR      │
     │ "Insufficient evidence│       │                                     │
     │  in museum collection"│       │ • Groq Cloud LLM (Natural/Focused)  │
     └───────────────────────┘       │   (Official Groq SDK + Citations)   │
                                     │ • Extractive Synthesizer (Offline)  │
                                     │   (Zero-key deterministic fallback) │
                                     └──────────────────┬──────────────────┘
                                                        │
                                                        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         6. CITATION VERIFICATION & INTEGRITY GUARD          │
  │   - Validate [N] markers strictly against supplied records  │
  │   - Discard out-of-range or hallucinated citation markers   │
  │   - Official CMA portal links & thumbnail verification      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          7. INTERACTIVE STREAMLIT WEB APP (Port 8501)       │
  │   - Generator Mode Toggle • Docent Tab • Search • Gallery   │
  │   - 3-Tier Layout: Answer -> Artwork Details -> Citations   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 💻 Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.13 | High performance, modern typing, robust ecosystem |
| **Cloud LLM (Optional)** | Groq Python SDK (`groq`) | Ultra-fast LPU inference, natural conversational grounded docent answers |
| **Voice Input (STT)** | Groq Whisper (`whisper-large-v3-turbo`) | High-accuracy multilingual speech-to-text with sub-second API latency |
| **Voice Output (TTS)** | Piper TTS (`en_US-lessac-medium`) & `pyttsx3` | Fast local neural ONNX speech synthesis on CPU with zero cloud dependencies |
| **Offline Synthesizer** | Native Extractive Synthesizer | Deterministic docent answering with zero network calls or external API keys |
| **Data Engine** | Pandas & NumPy | Fast data transformation and array vector math |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Compact (384-dim), fast CPU/GPU inference, top semantic benchmark score |
| **Vector Index** | `faiss-cpu` (`IndexFlatIP`) | Exact inner product (cosine similarity), zero recall loss, fast lookup |
| **UI Framework** | Streamlit | Rapid interactive web interface with native layout & data visualization |
| **Data Source** | Cleveland Museum of Art API | 64,000+ open access artworks, CC0 public domain licensing |
| **Testing** | `pytest` | 38 automated unit, integration, mock voice, and RAG regression tests |

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
* **Unit & Regression Suite**: 38 automated tests passing in `tests/test_rag.py`, `tests/test_retrieval.py`, and `tests/test_voice.py` (`python -m pytest tests/ -v`).
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

## ⚡ Answer Generation Modes: Groq Cloud LLM & Offline Fallback

MuseGuide features dual answer generation engines designed for high-quality, grounded curatorial responses with zero hallucination:

### 1. Groq Cloud LLM Mode (Natural, Conversational & Focused)
* **Engine**: Powered by Groq's official API and Python SDK (`groq`).
* **Model**: Supported high-speed models such as `qwen/qwen3.8-27b` (configurable via `GROQ_MODEL`).
* **Behavior**:
  * Delivers direct answers in the very first sentence without repetitive catalog throat-clearing (e.g., skips *"Regarding your inquiry..."* or *"Curatorial records document the work as follows..."*).
  * Synthesizes natural, human docent narratives adapted to inquiry types: identification questions immediately state the classification and medium; historical questions narrate the story concisely.
  * Embeds inline `[1]`, `[2]` citation markers tied strictly to provided retrieved museum records.
  * Rejects fabricated or out-of-range citations (e.g., `[9]` when only 2 documents were retrieved).

### 2. Offline Extractive Mode (100% Local & Deterministic)
* **Engine**: Native rule-guided curatorial docent synthesizer with zero external dependencies.
* **Key Requirements**: Requires **no API key** and **no network connectivity**.
* **Behavior**:
  * Extracts verified factual propositions directly from curated curatorial notes.
  * Performs intent-based gating (artwork-specific vs. multi-work thematic inquiries).
  * Guaranteed 100% reproducible baseline for testing, grading, or offline environments.
  * Acts as the automatic, graceful fallback if a Groq request encounters a timeout, missing key, or rate limit.

---

## 🔑 Groq Configuration Guide

### Step 1: Obtain a Groq API Key
1. Sign up for a free account on the official Groq Console: [https://console.groq.com](https://console.groq.com).
2. Navigate to **API Keys** and click **Create API Key**.
3. Copy your key (starts with `gsk_...`).

> [!NOTE]
> Free-tier quotas, available models, and request limits are managed dynamically by Groq. Refer to the official [Groq Documentation & Limits](https://console.groq.com/docs) for current rate limits and model lifecycle updates.

### Step 2: Configure Environment Variables Locally
Copy the template configuration file:
```powershell
cp .env.example .env
```
Open `.env` in your text editor and set your configuration:
```env
# Cleveland Museum of Art Open Access API
CMA_API_BASE_URL=https://openaccess-api.clevelandart.org/api

# Groq Cloud LLM Configuration
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=qwen/qwen3.8-27b

# RAG Generator: 'auto', 'groq', or 'extractive'
RAG_GENERATOR=auto
```

> [!CAUTION]
> Never commit `.env` or your real API key to Git. MuseGuide's `.gitignore` explicitly excludes `.env` and `.env.*` to keep secrets secure.

### Step 3: Select Generation Mode in the Streamlit UI
* Open the **⚡ Docent Engine** section in the left sidebar of the Streamlit app.
* Toggle between **Groq Cloud LLM** and **Offline Extractive**.
* The docent response header explicitly labels the active engine (e.g. `groq (qwen/qwen3.8-27b)` or `extractive_synthesizer (offline fallback)`). The app never silently claims Groq was used if a fallback occurred.

---

## 🎙️ Voice Interaction & Local Text-to-Speech Guide

MuseGuide includes optional, multimodal voice capabilities while keeping the clean text-first experience as default:

### 1. Voice Input (Groq Whisper Speech-to-Text)
* **Recording Control**: A microphone widget beside the question input box enables browser-based audio recording (`st.audio_input`).
* **Transcription Pipeline**:
  1. The user clicks the microphone button and speaks their museum question.
  2. The recorded audio is transcribed by Groq's high-speed `whisper-large-v3-turbo` model.
  3. The recognized transcript is displayed in a dedicated card so the user can verify what was heard.
  4. The transcribed question is automatically dispatched to the **exact same RAG pipeline** as typed questions (`Retrieval -> Relevance gating -> Grounded answer -> Citations`).
  5. Session state deduplication prevents accidental duplicate submissions when Streamlit reruns.
* **Refusal & Grounding Parity**: If a voice query asks an out-of-domain or unsupported question (e.g., *"What is the capital of Mars?"*), the anti-hallucination guardrail triggers normally.

### 2. Voice Output (Local Text-to-Speech via Piper & pyttsx3)
* **Manual Playback Only**: Audio is **never played automatically** upon receiving an answer. A `🔊 Speak Answer` button appears only after a valid docent answer exists.
* **Clean Speech Synthesis**: The TTS cleaner strips inline citation markers (`[1]`, `[2]`), markdown bold/italics (`**...**`, `*...*`), and raw URLs from the spoken narration while preserving them in the displayed text.
* **Dual TTS Engine Strategy**:
  1. **Primary (Neural ONNX)**: Piper TTS using the articulate `en_US-lessac-medium` voice model for natural museum docent narration.
  2. **Automatic Fallback (Local OS)**: If the Piper model binary is not yet downloaded, MuseGuide seamlessly falls back to `pyttsx3` (Windows SAPI5 / macOS NSSpeechSynthesizer), generating audio 100% offline with zero downloads.

### 3. How to Download the Local Piper Voice Model
Run the following one-line helper in terminal:
```powershell
python -c "from src.voice import download_piper_model; download_piper_model()"
```
This downloads `en_US-lessac-medium.onnx` (~60 MB) and its configuration to `models/piper/`.
*Note: `models/` and all audio binaries (`*.wav`, `*.onnx`) are strictly excluded by `.gitignore` to maintain a clean repository.*

### 4. Running Without Voice (Offline & Graceful Degradation)
* **Text Remains Default**: Voice input and speech synthesis are completely optional. Users can type questions and read text answers without any microphone or speaker dependencies.
* **No API Key Behavior**: If `GROQ_API_KEY` is not configured, the app operates 100% offline using the extractive synthesizer for answers and `pyttsx3` for speech. If the microphone is clicked without an API key, the app displays a clear notification and allows typing.

### 5. Troubleshooting
* **Microphone Permissions**: Ensure your browser has permitted microphone access for `http://localhost:8501`.
* **Empty Audio Warnings**: Speak clearly for at least 1–2 seconds before stopping the recording widget.
* **Audio Duplicate Playback**: Session state locks prevent audio from re-playing on page interaction.

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
