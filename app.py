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
from pathlib import Path
import pandas as pd
import streamlit as st

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
)

# Page configuration
st.set_page_config(
    page_title="MuseGuide — AI Museum Guide Using RAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Cached Resource Loaders
@st.cache_resource(show_spinner="Loading semantic retrieval engine...")
def load_searcher() -> SemanticSearcher:
    """Load and cache the SemanticSearcher with FAISS index."""
    return SemanticSearcher(
        index_path="data/index/cma_index.faiss",
        metadata_path="data/index/cma_metadata.json",
    )


@st.cache_resource(show_spinner="Initializing AI Docent Generator...")
def load_generator() -> AnswerGenerator:
    """Load and cache the AnswerGenerator pipeline."""
    searcher = load_searcher()
    return AnswerGenerator(searcher=searcher, min_confidence=0.35, top_k=4)


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
    # TAB 1: AI Museum Docent (RAG with Citations)
    # -------------------------------------------------------------
    with tab_docent:
        st.markdown("### Ask the AI Docent")
        st.caption("Answers are strictly grounded in verified museum records with linked citations.")

        default_q = st.session_state.get(
            "user_query",
            "Who was Nathaniel Hurd and what portrait of him is in the collection?",
        )

        query_input = st.text_input(
            "Enter your natural language question:",
            value=default_q,
            placeholder="e.g., What Ancient Egyptian cobras or royal symbols are in the museum?",
        )

        col_btn, col_clear = st.columns([1, 6])
        with col_btn:
            ask_btn = st.button("Ask Docent", type="primary")

        if ask_btn or query_input:
            dept_filter = selected_dept if selected_dept != "All Departments" else None
            culture_filter = selected_culture if selected_culture != "All Cultures" else None

            with st.spinner("Retrieving evidence and formulating guided answer..."):
                response = generator.generate(
                    query=query_input,
                    department=dept_filter,
                    culture=culture_filter,
                    backend=backend_choice,
                )

            # Section A: Main Direct Answer
            render_docent_answer(
                answer_text=response.answer,
                is_grounded=response.grounded,
                model_used=response.model_used,
            )

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
        st.markdown("### Semantic Vector Search")
        st.caption("Search across 2,000 museum records using conceptual and natural-language queries.")

        search_col1, search_col2 = st.columns([4, 1])
        with search_col1:
            search_query = st.text_input(
                "Search museum collection:",
                value="Ancient Greek ceramic amphora with mythological warriors",
                placeholder="e.g., bronze sculpture, Buddhist altar, samurai sword...",
            )
        with search_col2:
            top_k_select = st.slider("Results", min_value=3, max_value=12, value=6)

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
        st.markdown("### Browse Curated Museum Gallery")
        st.caption("Explore high-resolution open-access artefacts from the Cleveland Museum of Art.")

        gallery_filter = st.text_input("Filter gallery by title or artist:", value="")

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

        st.caption(f"Showing {min(18, len(filtered_df))} of {len(filtered_df)} matching artworks")

        gallery_cols = st.columns(3)
        for idx, (_, row) in enumerate(filtered_df.head(18).iterrows()):
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
        st.markdown("### Museum Collection Analytics")
        st.caption("Data distribution and curatorial statistics across the 2,000 indexed artefacts.")

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
