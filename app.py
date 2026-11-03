"""MuseGuide — AI Museum Guide Using RAG.

Interactive Streamlit Web Application featuring:
- AI Museum Docent with Grounded Answer Generation & Verified Citations
- Semantic Vector Search with Compound Filtering
- Curated Collection Gallery
- Artefact Similarity Discovery
- Collection Analytics
"""

import os
import sys
import random
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.answer import AnswerGenerator
from src.retrieval.search import SemanticSearcher
from src.ui.components import (
    apply_custom_css,
    render_artwork_details,
    render_citation_card,
    render_docent_answer,
    render_header,
    render_section_banner,
    render_speak_control,
)
from src.voice import DocentTTS, VoiceTranscriber

# Page configuration
sidebar_initial = os.getenv("SIDEBAR_INITIAL_STATE", "collapsed").strip().lower()
st.set_page_config(
    page_title="MuseGuide — AI Museum Guide Using RAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed" if sidebar_initial != "expanded" else "expanded",
)


# Cached Resource Loaders
@st.cache_resource(show_spinner="Loading semantic retrieval engine...")
def load_searcher() -> SemanticSearcher:
    """Load and cache the SemanticSearcher with FAISS index."""
    index_path = "data/index/cma_index.faiss"
    metadata_path = "data/index/cma_metadata.json"

    # Auto-build index on cold deployment if files are missing
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        processed_json = "data/processed/cma_artworks.json"
        if os.path.exists(processed_json):
            with st.spinner("Building FAISS semantic vector index for initial deployment..."):
                from src.retrieval.build_index import build_faiss_index, load_records
                records = load_records(processed_json)
                build_faiss_index(records, output_dir="data/index")

    return SemanticSearcher(
        index_path=index_path,
        metadata_path=metadata_path,
    )


@st.cache_resource(show_spinner="Initializing AI Docent Generator...")
def load_generator() -> AnswerGenerator:
    """Load and cache the AnswerGenerator pipeline."""
    searcher = load_searcher()
    return AnswerGenerator(searcher=searcher, min_confidence=0.35, top_k=4)


@st.cache_resource(show_spinner="Initializing Local Speech Synthesizer...")
def load_tts_engine() -> DocentTTS:
    """Load and cache the local Piper / pyttsx3 speech synthesizer."""
    return DocentTTS()


@st.cache_resource(show_spinner="Initializing Speech Transcriber...")
def load_transcriber() -> VoiceTranscriber:
    """Load and cache Groq Whisper transcriber."""
    return VoiceTranscriber()


@st.cache_data
def load_metadata_df() -> pd.DataFrame:
    """Load artwork metadata into a pandas DataFrame."""
    df = pd.read_csv("data/processed/cma_artworks.csv")
    return df


def main():
    apply_custom_css()

    # Initialize engines
    try:
        searcher = load_searcher()
        generator = load_generator()
        artworks_df = load_metadata_df()
        tts_engine = load_tts_engine()
        transcriber = load_transcriber()
    except Exception as e:
        st.error(f"Error loading MuseGuide resources: {e}")
        st.info("Please verify that data/processed/cma_artworks.json and data/index/cma_index.faiss exist.")
        return

    # Sidebar: Collection Overview & Controls
    with st.sidebar:
        st.markdown("### 🏛️ MuseGuide Explorer")
        st.caption("AI-Powered Museum Exploration using RAG")

        # Key stats metrics
        total_items = len(searcher.metadata)
        st.metric("Curated Artefacts", f"{total_items:,}")
        st.metric("Collection Source", "Cleveland Museum of Art")
        st.metric("License", "CC0 Public Domain")

        st.divider()
        st.markdown("#### ⚡ Docent Engine")
        has_groq_key = bool(os.getenv("GROQ_API_KEY"))
        selected_mode = st.radio(
            "Generation Mode",
            options=["Groq Cloud LLM", "Offline Extractive"] if has_groq_key else ["Offline Extractive", "Groq Cloud LLM (No API Key)"],
            index=0,
            help="Groq Cloud LLM synthesizes natural, conversational docent answers strictly grounded in evidence. Offline mode generates deterministic answers without API keys or network access.",
        )
        backend_choice = "groq" if "Groq" in selected_mode and has_groq_key else "extractive"
        active_engine_name = f"Groq ({generator.model_name})" if backend_choice == "groq" else "Offline Extractive"
        st.caption(f"Active Engine: **{active_engine_name}**")
        st.caption(f"Speech Narration: **{tts_engine.engine_name}**")

        st.divider()
        st.markdown("#### 🎯 Filter Controls")
        departments = ["All Departments"] + searcher.get_departments()
        selected_dept = st.selectbox("Department", departments)

        cultures = ["All Cultures"] + searcher.get_cultures()[:50]
        selected_culture = st.selectbox("Culture", cultures)

        st.divider()
        st.markdown("#### 💡 Suggested Inquiries")
        example_prompts = [
            "Who was Nathaniel Hurd and what portrait of him is in the collection?",
            "What Ancient Egyptian artefacts with cobra or royal symbolism are in the museum?",
            "Are there French landscape paintings depicting ponds or water in the collection?",
            "Describe the ancient Greek black-figure amphora in the collection.",
            "What is the capital city of planet Mars?",
        ]
        for prompt in example_prompts:
            if st.button(prompt, key=f"ex_{prompt[:20]}"):
                st.session_state["user_query"] = prompt
                st.session_state["suggested_prompt_clicked"] = True

        st.divider()
        st.markdown(
            "<div style='font-size:0.75rem; color:#718096;'>"
            "MuseGuide RAG Architecture<br>"
            "• Embeddings: all-MiniLM-L6-v2 (384-dim)<br>"
            "• Index: FAISS IndexFlatIP (Cosine)<br>"
            "• Grounding: Anti-Hallucination Gated"
            "</div>",
            unsafe_allow_html=True,
        )

    # Main Brand Header
    render_header()

    # Primary Navigation Tabs
    tab_docent, tab_search, tab_gallery, tab_insights = st.tabs(
        [
            "🏛️ AI Museum Docent (RAG)",
            "🔍 Semantic Search",
            "🎨 Collection Gallery",
            "📊 Collection Insights",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: AI Museum Docent (RAG with Citations & Voice)
    # -------------------------------------------------------------
    with tab_docent:
        render_section_banner(
            "🏛️ AI Museum Docent (RAG)",
            "Answers are strictly grounded in verified museum records with linked citations.",
        )

        # Question input area: Primary text input + optional audio microphone
        col_text, col_mic = st.columns([5, 2])
        with col_text:
            default_q = st.session_state.get(
                "user_query",
                "Who was Nathaniel Hurd and what portrait of him is in the collection?",
            )
            query_input = st.text_input(
                "Enter your natural language question:",
                value=default_q,
                placeholder="e.g., What Ancient Egyptian cobras or royal symbols are in the museum?",
                key="docent_text_query",
            )
            col_ask, _ = st.columns([1, 5])
            with col_ask:
                ask_btn = st.button("Ask Docent", type="primary", key="btn_ask_docent")

        with col_mic:
            st.markdown(
                "<div style='font-size:0.83rem; font-weight:600; color:#d4af37; margin-bottom:4px;'>"
                "🎙️ Voice Input (Groq Whisper)"
                "</div>",
                unsafe_allow_html=True,
            )
            voice_recording = st.audio_input("Record voice question", label_visibility="collapsed", key="docent_audio_mic")

        # Determine if a query was triggered
        triggered_query = None

        # Check for new microphone recording
        if voice_recording is not None:
            audio_bytes = voice_recording.getvalue()
            audio_sig = hash(audio_bytes)
            if audio_bytes and st.session_state.get("last_audio_sig") != audio_sig:
                st.session_state["last_audio_sig"] = audio_sig
                with st.status("🎙️ Transcribing voice recording with Groq Whisper...", expanded=True) as status:
                    theme = os.getenv("UI_THEME", "vibrant").strip().lower()
                    if theme != "classic":
                        st.markdown(
                            """
                            <div class="audio-equalizer" style="margin: 6px 0 10px 0;">
                                <div class="eq-bar bar-2"></div>
                                <div class="eq-bar bar-4"></div>
                                <div class="eq-bar bar-6"></div>
                                <div class="eq-bar bar-8"></div>
                                <div class="eq-bar bar-10"></div>
                                <div class="eq-bar bar-12"></div>
                                <span class="eq-label">🎙️ Groq Whisper Listening & Transcribing...</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.write("Processing audio with `whisper-large-v3-turbo`...")
                    try:
                        transcribed = transcriber.transcribe(audio_bytes, filename=getattr(voice_recording, "name", "voice_query.wav"))
                        st.session_state["user_query"] = transcribed
                        st.session_state["last_transcript"] = transcribed
                        status.update(label=f"✓ Transcribed: \"{transcribed}\"", state="complete", expanded=False)
                        triggered_query = transcribed
                    except Exception as e:
                        status.update(label="⚠️ Voice transcription failed", state="error", expanded=True)
                        st.error(f"Voice Transcription Error: {e}")
                        st.info("You can still type your question in the text box on the left.")

        # Check for manual Ask Docent button
        if ask_btn:
            triggered_query = query_input
            st.session_state["user_query"] = query_input
        elif st.session_state.get("suggested_prompt_clicked"):
            triggered_query = st.session_state.get("user_query")
            st.session_state["suggested_prompt_clicked"] = False

        # First-load initialization so visitors immediately see a guided response
        if "active_docent_response" not in st.session_state and default_q:
            triggered_query = default_q

        # Execute generation if a query was triggered
        if triggered_query:
            dept_filter = selected_dept if selected_dept != "All Departments" else None
            culture_filter = selected_culture if selected_culture != "All Cultures" else None

            theme = os.getenv("UI_THEME", "vibrant").strip().lower()
            with st.spinner("Retrieving evidence and formulating guided answer..."):
                if theme != "classic":
                    st.markdown(
                        """
                        <div class="generation-pulse-banner">
                            <div class="generation-ring"></div>
                            <span>✦ Docent Neural Engine Formulating Grounded Answer...</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                response = generator.generate(
                    query=triggered_query,
                    department=dept_filter,
                    culture=culture_filter,
                    backend=backend_choice,
                )
                st.session_state["active_docent_response"] = response
                # Clear previous voice audio cache for the new answer
                st.session_state["tts_audio_cache"] = None

        # Display recognized voice transcript badge if present
        if st.session_state.get("last_transcript"):
            st.markdown(
                f'<div class="voice-box"><span style="color:#f6e05e; font-weight:600;">🎙️ Recognized Voice:</span> '
                f'<em>"{st.session_state["last_transcript"]}"</em></div>',
                unsafe_allow_html=True,
            )

        # Render the response
        response = st.session_state.get("active_docent_response")
        if response:
            # Section A: Main Direct Answer
            render_docent_answer(
                answer_text=response.answer,
                is_grounded=response.grounded,
                model_used=response.model_used,
            )

            # Voice Output: Speak Answer Button (Manual playback only, never autoplays)
            render_speak_control(response.answer, tts_engine)

            # Section B: Supporting Artwork Details (Compact Metadata Grid)
            if response.grounded and response.artwork_details:
                render_artwork_details(response.artwork_details)

            # Section C: Sources & Verified Citations
            if response.citations:
                st.markdown(f"#### 📚 Sources & Citations ({len(response.citations)})")
                st.caption("Curatorial evidence documents linking directly to official museum records.")
                for cit in response.citations:
                    render_citation_card(cit)

            # Expandable Raw Evidence Drawer
            with st.expander("🔍 View Raw Retrieved Context Documents"):
                if response.retrieved_documents:
                    for i, doc in enumerate(response.retrieved_documents, 1):
                        st.markdown(f"**Document {i}: {doc.get('title')}** (Ranking Score: {doc.get('score', 0.0):.4f} | Cosine: {doc.get('base_score', doc.get('score', 0.0)):.4f})")
                        st.text(doc.get("retrieval_text", ""))
                        st.divider()
                else:
                    st.info("No documents met the retrieval threshold.")

    # -------------------------------------------------------------
    # TAB 2: Semantic Search & Similarity Explorer
    # -------------------------------------------------------------
    with tab_search:
        render_section_banner(
            "🔍 Semantic Vector Search",
            "Search across 2,000 museum records using conceptual and natural-language queries.",
        )

        # List of curated exploration themes for shuffle discovery
        EXPLORATION_QUERIES = [
            "Ancient Greek ceramic amphora with mythological warriors",
            "Ancient Egyptian gold jewellery and royal cobra uraeus",
            "Medieval illuminated manuscript with vibrant gold leaf",
            "French Impressionist landscape oil paintings of rivers and gardens",
            "Japanese samurai armor and katana swords from Edo period",
            "Roman imperial marble portrait sculptures and deities",
            "Chinese Ming dynasty blue and white porcelain vases",
            "Renaissance religious paintings with rich lapis lazuli pigment",
            "American colonial silver craftsmanship and portraiture",
            "Indian miniature paintings of royal courts and nature",
            "19th century boxing and athletic club saloons in New York",
            "Art Nouveau decorative glassware and stained glass panels",
        ]

        # On initial load or refresh, choose a random exploration query
        if "semantic_search_query" not in st.session_state:
            st.session_state["semantic_search_query"] = random.choice(EXPLORATION_QUERIES)

        search_col1, search_col_btn, search_col2 = st.columns([5, 1.4, 1.6])
        with search_col1:
            search_query = st.text_input(
                "Search museum collection:",
                value=st.session_state.get("semantic_search_query", EXPLORATION_QUERIES[0]),
                placeholder="e.g., bronze sculpture, Buddhist altar, samurai sword...",
                key="semantic_query_input",
            )
        with search_col_btn:
            st.write("")
            st.write("")
            if st.button("🔀 Shuffle", key="btn_shuffle_semantic", help="Shuffle to a new theme & show 10 random artworks"):
                st.session_state["semantic_search_query"] = random.choice(EXPLORATION_QUERIES)
                st.rerun()
        with search_col2:
            top_k_select = st.slider("Results", min_value=3, max_value=15, value=10)

        if search_query:
            dept_filter = selected_dept if selected_dept != "All Departments" else None
            culture_filter = selected_culture if selected_culture != "All Cultures" else None

            results = searcher.search(
                query=search_query,
                top_k=top_k_select,
                department=dept_filter,
                culture=culture_filter,
            )

            if results:
                st.success(f"Retrieved {len(results)} semantically relevant artefacts")
                cols = st.columns(3)
                for i, item in enumerate(results):
                    with cols[i % 3]:
                        img_url = item.get("image_url")
                        if img_url:
                            try:
                                st.image(img_url, use_container_width=True)
                            except Exception:
                                st.caption("🖼️ [Image Preview Offline]")
                        st.markdown(f"**{item.get('title')}**")
                        creator_s = item.get("creator_name") or "Unknown"
                        date_s = item.get("creation_date") or "Unknown"
                        st.caption(f"{creator_s} • {date_s}")
                        st.markdown(
                            f'<span class="score-pill">Ranking Score: {item["score"]:.2f}</span> '
                            f'<span class="meta-tag">{item.get("department", "General")}</span>',
                            unsafe_allow_html=True,
                        )
                        if item.get("museum_url"):
                            st.markdown(f"[Museum Page]({item['museum_url']})")

                        # Related Artefacts Discovery
                        with st.expander("✨ Related Artefacts"):
                            try:
                                similar_items = searcher.find_similar(item["id"], top_k=2)
                                for sim in similar_items:
                                    st.markdown(f"• **{sim['title']}**")
                                    st.caption(f"{sim.get('culture', 'N/A')} • Cosine Sim: {sim['score']:.2f}")
                                    if sim.get("image_url"):
                                        try:
                                            st.image(sim["image_url"], width=120)
                                        except Exception:
                                            pass
                            except Exception as sim_err:
                                st.caption("Similar works unavailable.")

                        st.divider()
            else:
                st.warning("No records matched your search query and filters.")

    # -------------------------------------------------------------
    # TAB 3: Collection Gallery
    # -------------------------------------------------------------
    with tab_gallery:
        render_section_banner(
            "🎨 Curated Museum Gallery",
            "Explore high-resolution open-access artefacts from the Cleveland Museum of Art (CC0 Public Domain).",
        )

        # Random seed initialization for refresh
        if "gallery_seed" not in st.session_state:
            st.session_state["gallery_seed"] = random.randint(1, 999999)

        # Control bar: Filter input + Shuffle button + Revertible Random 10 Mode toggle
        col_f1, col_f2, col_f3 = st.columns([5, 1.4, 2.2])
        with col_f1:
            gallery_filter = st.text_input("Filter gallery by title or artist:", value="", placeholder="Type to filter...")
        with col_f2:
            st.write("")
            st.write("")
            if st.button("🔀 Shuffle", key="btn_shuffle_gallery", help="Shuffle and display 10 fresh random artworks"):
                st.session_state["gallery_seed"] = random.randint(1, 999999)
                st.rerun()
        with col_f3:
            st.write("")
            st.write("")
            shuffle_mode = st.toggle("🔀 Random 10 Mode", value=True, help="Toggle between Random 10 shuffle on refresh vs. In-Order catalog view (Revertible)")

        # Filter dataframe
        filtered_df = artworks_df
        if selected_dept != "All Departments":
            filtered_df = filtered_df[filtered_df["department"] == selected_dept]
        if selected_culture != "All Cultures":
            filtered_df = filtered_df[filtered_df["culture"] == selected_culture]
        if gallery_filter:
            mask = (
                filtered_df["title"].str.contains(gallery_filter, case=False, na=False)
                | filtered_df["creator_name"].str.contains(gallery_filter, case=False, na=False)
            )
            filtered_df = filtered_df[mask]

        # Sample 10 artworks (Random 10 vs In-Order 10)
        num_items = min(10, len(filtered_df))
        if shuffle_mode and len(filtered_df) > num_items:
            display_df = filtered_df.sample(n=num_items, random_state=st.session_state["gallery_seed"])
        else:
            display_df = filtered_df.head(num_items)

        st.caption(f"Showing {len(display_df)} of {len(filtered_df)} matching artworks (Mode: {'Random 10 (Shuffle)' if shuffle_mode else 'Sequential Catalog'})")

        gallery_cols = st.columns(3)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            with gallery_cols[idx % 3]:
                if pd.notna(row.get("image_url")):
                    try:
                        st.image(row["image_url"], use_container_width=True)
                    except Exception:
                        st.caption("🖼️ [Image Preview Offline]")
                st.markdown(f"**{row.get('title', 'Untitled')}**")
                creator_str = row.get("creator_name") if pd.notna(row.get("creator_name")) else "Unknown"
                date_str = row.get("creation_date") if pd.notna(row.get("creation_date")) else "Unknown"
                st.caption(f"{creator_str} • {date_str}")

                with st.expander("Curatorial Details & Related Works"):
                    st.write(row.get("description", "No description available."))
                    if pd.notna(row.get("culture")):
                        st.markdown(f"**Culture**: {row['culture']}")
                    if pd.notna(row.get("technique")):
                        st.markdown(f"**Medium**: {row['technique']}")
                    if pd.notna(row.get("museum_url")):
                        st.markdown(f"[Official CMA Record]({row['museum_url']})")

                    # Related Artefacts Discovery
                    st.markdown("---")
                    st.markdown("**✨ Semantically Related Artefacts:**")
                    try:
                        sim_gallery = searcher.find_similar(row["id"], top_k=2)
                        for sim in sim_gallery:
                            st.markdown(f"• **{sim['title']}** ({sim.get('culture', 'N/A')}) — *Cosine Sim: {sim['score']:.2f}*")
                    except Exception:
                        st.caption("Related works unavailable.")

                st.divider()

    # -------------------------------------------------------------
    # TAB 4: Collection Insights & Analytics
    # -------------------------------------------------------------
    with tab_insights:
        render_section_banner(
            "📊 Museum Collection Analytics",
            "Data distribution and curatorial statistics across the 2,000 indexed artefacts.",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(artworks_df):,}")
        c2.metric("Images Verified", "100%")
        c3.metric("Descriptions", "100%")
        c4.metric("Open Access", "CC0")

        st.markdown("#### Artefacts by Curatorial Department")
        dept_counts = artworks_df["department"].value_counts().head(12)
        st.bar_chart(dept_counts)

        st.markdown("#### Top Historical Cultures")
        culture_counts = artworks_df["culture"].value_counts().head(10)
        st.bar_chart(culture_counts)


if __name__ == "__main__":
    main()
