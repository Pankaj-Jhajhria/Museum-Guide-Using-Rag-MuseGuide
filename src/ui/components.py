"""Reusable UI components and custom CSS for the MuseGuide Streamlit app."""

import streamlit as st
from typing import Any, Dict, List, Optional
from src.rag.answer import Citation


def apply_custom_css():
    """Inject polished museum-themed CSS styling."""
    st.markdown(
        """
        <style>
        /* Base typography & colors */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@300;400;600;700&display=swap');

        .main-header {
            font-family: 'Cinzel', serif;
            color: #d4af37;
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 0.2rem;
            text-align: center;
        }
        .sub-header {
            font-family: 'Inter', sans-serif;
            color: #a0aec0;
            font-size: 1.05rem;
            text-align: center;
            margin-bottom: 1.8rem;
        }
        .badge-verified {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            display: inline-block;
        }
        .badge-refusal {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            display: inline-block;
        }
        .docent-box {
            background-color: #1a202c;
            border-left: 4px solid #d4af37;
            padding: 1.4rem;
            border-radius: 0 10px 10px 0;
            margin: 1.2rem 0;
            line-height: 1.7;
            font-size: 1.02rem;
        }
        .citation-card {
            background: #232a3b;
            border: 1px solid #3b4252;
            border-radius: 8px;
            padding: 0.9rem;
            margin-bottom: 0.8rem;
            transition: transform 0.2s, border-color 0.2s;
        }
        .citation-card:hover {
            border-color: #d4af37;
            transform: translateY(-2px);
        }
        .citation-num {
            background: #d4af37;
            color: #000;
            font-weight: 700;
            font-size: 0.85rem;
            padding: 2px 7px;
            border-radius: 4px;
            margin-right: 8px;
        }
        .meta-tag {
            background: #2d3748;
            color: #cbd5e0;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.78rem;
            margin-right: 6px;
        }
        .score-pill {
            background: rgba(212, 175, 55, 0.15);
            color: #f6e05e;
            border: 1px solid rgba(212, 175, 55, 0.3);
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        /* Sidebar metric styling to prevent text truncation */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"] > div,
        [data-testid="stSidebar"] [data-testid="stMetricValue"] * {
            font-size: 1.02rem !important;
            white-space: normal !important;
            text-overflow: unset !important;
            overflow: visible !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            line-height: 1.35 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
            font-size: 0.8rem !important;
            color: #a0aec0 !important;
        }
        .answer-section-title {
            font-family: 'Cinzel', serif;
            color: #d4af37;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .artwork-detail-box {
            background: #1e2533;
            border: 1px solid #3b4252;
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        .artwork-detail-title {
            color: #f6e05e;
            font-weight: 600;
            font-size: 0.96rem;
            margin-bottom: 0.35rem;
        }
        .meta-field {
            display: inline-block;
            margin-right: 14px;
            font-size: 0.86rem;
            color: #e2e8f0;
        }
        .meta-field strong {
            color: #a0aec0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render top brand header."""
    st.markdown('<div class="main-header">🏛️ MuseGuide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI Museum Exploration & Retrieval-Augmented Generation (RAG) | Grounded in Cleveland Museum of Art Open Access Records (CC0 Public Domain)</div>',
        unsafe_allow_html=True,
    )


def render_docent_answer(answer_text: str, is_grounded: bool, model_used: str):
    """Render prominent direct answer card with proper Markdown parsing and status badge."""
    if is_grounded:
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
            f'<span class="badge-verified">✓ Grounded in Verified Museum Records</span>'
            f'<span style="color:#718096; font-size:0.8rem;">Engine: {model_used}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="margin-bottom:8px;">'
            f'<span class="badge-refusal">⚠️ Anti-Hallucination Guardrail Active</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Main Section: Direct Answer (prominent card, native Markdown rendering for bold/italics)
    with st.container(border=True):
        st.markdown('<div class="answer-section-title">🏛️ Direct Answer</div>', unsafe_allow_html=True)
        st.markdown(answer_text)


def render_artwork_details(artwork_details: List[Dict[str, Any]]):
    """Render secondary Artwork Details section with compact metadata grid."""
    if not artwork_details:
        return

    st.markdown("#### 📋 Artwork Details")
    st.caption("Curatorial catalog metadata for cited collection records.")

    for art in artwork_details:
        cit_id = art.get("citation_id", 1)
        title = art.get("artwork", "Untitled")

        fields_html = []
        if art.get("artist"):
            fields_html.append(f'<span class="meta-field"><strong>Artist:</strong> {art["artist"]}</span>')
        if art.get("date"):
            fields_html.append(f'<span class="meta-field"><strong>Date:</strong> {art["date"]}</span>')
        if art.get("medium"):
            fields_html.append(f'<span class="meta-field"><strong>Medium:</strong> {art["medium"]}</span>')
        if art.get("department"):
            fields_html.append(f'<span class="meta-field"><strong>Department:</strong> {art["department"]}</span>')
        if art.get("culture"):
            fields_html.append(f'<span class="meta-field"><strong>Culture:</strong> {art["culture"]}</span>')

        metadata_line = " &nbsp;•&nbsp; ".join(fields_html) if fields_html else "<em>No catalog metadata recorded.</em>"

        box_html = (
            f'<div class="artwork-detail-box">'
            f'<div class="artwork-detail-title"><span class="citation-num">[{cit_id}]</span> {title}</div>'
            f'<div style="margin-top:4px;">{metadata_line}</div>'
            f'</div>'
        )
        st.markdown(box_html, unsafe_allow_html=True)


def render_citation_card(cit: Citation):
    """Render structured citation card with artwork details and official museum link."""
    with st.container():
        col_img, col_info = st.columns([1, 3])

        with col_img:
            if cit.image_url:
                st.image(cit.image_url, use_container_width=True)
            else:
                st.caption("No image available")

        with col_info:
            st.markdown(
                f'<span class="citation-num">[{cit.citation_id}]</span> '
                f'<strong>{cit.title}</strong> '
                f'<span class="score-pill">Ranking Score: {cit.similarity_score:.2f}</span>',
                unsafe_allow_html=True,
            )
            creator_text = cit.creator if cit.creator else "Unknown Artist"
            date_text = cit.creation_date if cit.creation_date else "Unknown Date"
            st.caption(f"{creator_text} • {date_text}")

            meta_html = ""
            if cit.culture:
                meta_html += f'<span class="meta-tag">{cit.culture}</span>'
            if cit.department:
                meta_html += f'<span class="meta-tag">{cit.department}</span>'
            if meta_html:
                st.markdown(f'<div style="margin: 4px 0;">{meta_html}</div>', unsafe_allow_html=True)

            if cit.evidence_snippet:
                st.markdown(f'<p style="font-size:0.85rem; color:#a0aec0; margin-top:6px;"><em>"{cit.evidence_snippet}"</em></p>', unsafe_allow_html=True)

            if cit.museum_url:
                st.markdown(f'[🏛️ View on Cleveland Museum of Art Portal]({cit.museum_url})')
        st.divider()
