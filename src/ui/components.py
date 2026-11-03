"""Reusable UI components and custom CSS for the MuseGuide Streamlit app."""

import os
import streamlit as st
from typing import Any, Dict, List, Optional
from src.rag.answer import Citation


def apply_custom_css():
    """Inject polished museum-themed CSS styling (supports 'vibrant' and 'classic' themes)."""
    theme = os.getenv("UI_THEME", "vibrant").strip().lower()
    if theme == "classic":
        _apply_classic_css()
    else:
        _apply_vibrant_css()


def _apply_vibrant_css():
    """Inject ultra-luxury 24K gold aesthetic, ambient lighting, and orbiting particle animations."""
    st.markdown(
        """
        <style>
        /* Base typography & fonts */
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Make top header bar transparent and remove Deploy button & three-dots menu */
        header[data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
            box-shadow: none !important;
            height: 3.4rem !important;
            z-index: 99999 !important;
        }

        /* Keep toolbar transparent and pointer events intact for the expand button */
        [data-testid="stToolbar"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            opacity: 1 !important;
            visibility: visible !important;
            display: flex !important;
        }

        /* Completely hide Deploy button, Three-Dots Menu, and top rainbow decoration line */
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stToolbar"] > div:not(:first-child) {
            display: none !important;
            visibility: hidden !important;
        }

        /* 24K Molten Gold Sidebar Expand Button (>>) on Top-Left */
        [data-testid="stExpandSidebarButton"] {
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            z-index: 999999 !important;
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.95) 0%, rgba(13, 19, 33, 0.98) 100%) !important;
            border: 1.5px solid rgba(255, 215, 0, 0.65) !important;
            border-radius: 10px !important;
            color: #FFD700 !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6), 0 0 16px rgba(255, 193, 7, 0.45) !important;
            transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            padding: 6px 10px !important;
            cursor: pointer !important;
        }
        [data-testid="stExpandSidebarButton"]:hover {
            background: linear-gradient(110deg, #FFE885 0%, #FFC107 40%, #D4AF37 80%) !important;
            color: #0A0E17 !important;
            border-color: #FFFFFF !important;
            transform: scale(1.08) !important;
            box-shadow: 0 0 24px rgba(255, 215, 0, 0.85) !important;
        }
        [data-testid="stExpandSidebarButton"] svg {
            fill: #FFD700 !important;
            stroke: #FFD700 !important;
            color: #FFD700 !important;
        }
        [data-testid="stExpandSidebarButton"]:hover svg {
            fill: #0A0E17 !important;
            stroke: #0A0E17 !important;
            color: #0A0E17 !important;
        }

        /* 24K Molten Gold Sidebar Collapse Button (<<) Inside Sidebar */
        [data-testid="stSidebarCollapseButton"] {
            display: inline-flex !important;
            visibility: visible !important;
            color: #FFD700 !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.92) 0%, rgba(13, 19, 33, 0.98) 100%) !important;
            border: 1.5px solid rgba(255, 215, 0, 0.55) !important;
            border-radius: 10px !important;
            color: #FFD700 !important;
            box-shadow: 0 2px 12px rgba(255, 193, 7, 0.3) !important;
            transition: all 0.25s ease !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: linear-gradient(110deg, #FFE885 0%, #FFC107 40%, #D4AF37 80%) !important;
            color: #0A0E17 !important;
            border-color: #FFFFFF !important;
            transform: scale(1.06) !important;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.75) !important;
        }
        [data-testid="stSidebarCollapseButton"] svg {
            fill: #FFD700 !important;
            stroke: #FFD700 !important;
        }
        [data-testid="stSidebarCollapseButton"] button:hover svg {
            fill: #0A0E17 !important;
            stroke: #0A0E17 !important;
        }

        .main .block-container {
            padding-top: 2rem !important;
        }

        /* Ambient Cyber-Gold Aurora Lightning Background */
        @keyframes nebulaBreath {
            0%, 100% {
                background-position: 0% 0%, 100% 0%, 50% 100%, 50% 50%;
            }
            50% {
                background-position: 8% 6%, 92% 10%, 48% 92%, 52% 48%;
            }
        }

        .stApp {
            background: radial-gradient(circle at 12% 15%, rgba(255, 193, 7, 0.15) 0%, transparent 42%),
                        radial-gradient(circle at 88% 18%, rgba(212, 175, 55, 0.13) 0%, transparent 38%),
                        radial-gradient(circle at 50% 85%, rgba(245, 158, 11, 0.10) 0%, transparent 48%),
                        radial-gradient(circle at 50% 45%, rgba(255, 215, 0, 0.08) 0%, transparent 52%),
                        #070A12 !important;
            background-size: 140% 140%, 130% 130%, 150% 150%, 100% 100% !important;
            animation: nebulaBreath 16s ease-in-out infinite !important;
            background-attachment: fixed !important;
        }

        /* Keyframe Animations for Gold Shimmer, Particle Sparks & Lightning Glow */
        @keyframes goldSweep {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes buttonShimmerBeam {
            0% { transform: translateX(-150%) skewX(-25deg); opacity: 0; }
            15% { opacity: 0.65; }
            35% { transform: translateX(250%) skewX(-25deg); opacity: 0; }
            100% { transform: translateX(250%) skewX(-25deg); opacity: 0; }
        }

        @keyframes goldParticleOrbit {
            0% {
                box-shadow: 
                    0 4px 18px rgba(255, 180, 0, 0.5),
                    0 0 10px rgba(255, 240, 160, 0.7),
                    -22px -12px 0 1.5px rgba(255, 215, 0, 0.95),
                    24px -14px 0 2px rgba(255, 170, 0, 0.85),
                    18px 16px 0 1px rgba(255, 255, 230, 0.98),
                    -20px 14px 0 2px rgba(255, 140, 0, 0.8);
            }
            25% {
                box-shadow: 
                    0 4px 22px rgba(255, 180, 0, 0.65),
                    0 0 14px rgba(255, 240, 160, 0.8),
                    -14px -22px 0 2px rgba(255, 255, 230, 0.98),
                    24px -6px 0 1px rgba(255, 215, 0, 0.9),
                    12px 22px 0 2.5px rgba(255, 160, 0, 0.85),
                    -24px 6px 0 1.5px rgba(255, 140, 0, 0.85);
            }
            50% {
                box-shadow: 
                    0 4px 28px rgba(255, 180, 0, 0.75),
                    0 0 18px rgba(255, 240, 160, 0.9),
                    20px -16px 0 1.5px rgba(255, 160, 0, 0.9),
                    16px 20px 0 2px rgba(255, 255, 230, 0.98),
                    -18px 18px 0 1.5px rgba(255, 215, 0, 0.95),
                    -16px -18px 0 2px rgba(255, 140, 0, 0.85);
            }
            75% {
                box-shadow: 
                    0 4px 22px rgba(255, 180, 0, 0.65),
                    0 0 14px rgba(255, 240, 160, 0.8),
                    22px 12px 0 2px rgba(255, 140, 0, 0.85),
                    -14px 22px 0 1.5px rgba(255, 255, 230, 0.98),
                    -24px -10px 0 2px rgba(255, 215, 0, 0.9),
                    14px -20px 0 1px rgba(255, 160, 0, 0.85);
            }
            100% {
                box-shadow: 
                    0 4px 18px rgba(255, 180, 0, 0.5),
                    0 0 10px rgba(255, 240, 160, 0.7),
                    -22px -12px 0 1.5px rgba(255, 215, 0, 0.95),
                    24px -14px 0 2px rgba(255, 170, 0, 0.85),
                    18px 16px 0 1px rgba(255, 255, 230, 0.98),
                    -20px 14px 0 2px rgba(255, 140, 0, 0.8);
            }
        }

        @keyframes secondaryOrbitSparks {
            0% {
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 215, 0, 0.35),
                            -16px -8px 0 1px rgba(255, 215, 0, 0.65), 16px 10px 0 1px rgba(255, 193, 7, 0.65);
            }
            50% {
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 16px rgba(255, 215, 0, 0.55),
                            16px -10px 0 1.5px rgba(255, 215, 0, 0.8), -16px 8px 0 1.5px rgba(255, 193, 7, 0.8);
            }
            100% {
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 215, 0, 0.35),
                            -16px -8px 0 1px rgba(255, 215, 0, 0.65), 16px 10px 0 1px rgba(255, 193, 7, 0.65);
            }
        }

        @keyframes electricGoldPulse {
            0%, 100% {
                box-shadow: 0 0 20px rgba(255, 193, 7, 0.35), inset 0 0 14px rgba(255, 193, 7, 0.08);
                border-color: rgba(255, 215, 0, 0.5);
            }
            50% {
                box-shadow: 0 0 35px rgba(255, 152, 0, 0.55), inset 0 0 22px rgba(255, 215, 0, 0.14);
                border-color: rgba(255, 240, 160, 0.8);
            }
        }

        @keyframes pulseGlow {
            0%, 100% {
                box-shadow: 0 0 12px rgba(16, 185, 129, 0.35);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 0 22px rgba(16, 185, 129, 0.65);
                transform: scale(1.02);
            }
        }

        @keyframes goldAuraLight {
            0%, 100% {
                filter: drop-shadow(0 0 22px rgba(255, 215, 0, 0.6)) drop-shadow(0 0 45px rgba(255, 152, 0, 0.4));
            }
            50% {
                filter: drop-shadow(0 0 35px rgba(255, 235, 150, 0.85)) drop-shadow(0 0 60px rgba(212, 175, 55, 0.55));
            }
        }

        /* 24K Molten Gold Brand Header */
        .main-header {
            font-family: 'Cinzel', serif;
            background: linear-gradient(270deg, #FFE885 0%, #FFC107 25%, #FF9800 50%, #D4AF37 75%, #FFE885 100%);
            background-size: 400% 400%;
            animation: goldSweep 8s ease infinite, goldAuraLight 6s ease-in-out infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.1rem;
            font-weight: 900;
            letter-spacing: 3.5px;
            text-align: center;
            margin-bottom: 0.15rem;
        }
        .sub-header {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #94A3B8;
            font-size: 0.96rem;
            text-align: center;
            margin-bottom: 0.8rem;
            letter-spacing: 0.5px;
            text-shadow: 0 0 12px rgba(148, 163, 184, 0.25);
        }

        /* Neoclassical Gold Ornament Divider */
        .gold-ornament {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0.4rem auto 1.8rem auto;
            max-width: 580px;
        }
        .gold-ornament-line {
            flex: 1;
            height: 1.5px;
            background: linear-gradient(90deg, transparent 0%, #FFD700 50%, #FF9800 100%);
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
        }
        .gold-ornament-line:last-child {
            background: linear-gradient(90deg, #FF9800 0%, #FFD700 50%, transparent 100%);
        }
        .gold-ornament-gem {
            padding: 0 16px;
            font-size: 0.95rem;
            font-weight: 800;
            color: #FFD700;
            letter-spacing: 2px;
            text-shadow: 0 0 14px rgba(255, 215, 0, 0.8), 0 0 25px rgba(255, 152, 0, 0.5);
            animation: pulseGlow 3.5s ease-in-out infinite;
        }

        /* 24K Molten Gold Primary Buttons (Ask Docent) with Orbiting Particles */
        button[kind="primary"] {
            position: relative !important;
            overflow: hidden !important;
            background: linear-gradient(110deg, #FFE885 0%, #FFC107 20%, #FF9800 45%, #D4AF37 70%, #FFF3B0 85%, #FFC107 100%) !important;
            background-size: 250% 250% !important;
            animation: goldSweep 4.5s ease infinite, goldParticleOrbit 3.5s ease-in-out infinite !important;
            color: #0A0E17 !important;
            border: 1.5px solid rgba(255, 245, 180, 0.8) !important;
            border-radius: 12px !important;
            font-weight: 900 !important;
            font-size: 1.02rem !important;
            letter-spacing: 0.8px !important;
            padding: 0.65rem 1.6rem !important;
            text-transform: uppercase !important;
            text-shadow: 0 1px 1px rgba(255, 255, 255, 0.6) !important;
            transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        }
        button[kind="primary"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 70%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.55), transparent);
            animation: buttonShimmerBeam 4.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
            pointer-events: none;
        }
        button[kind="primary"]:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 
                0 8px 35px rgba(255, 193, 7, 0.95),
                0 0 55px rgba(255, 152, 0, 0.8),
                0 0 90px rgba(255, 235, 150, 0.65) !important;
            border-color: #FFFFFF !important;
        }
        button[kind="primary"]:active {
            transform: translateY(1px) scale(0.98) !important;
        }

        /* Glowing Secondary & Shuffle Buttons with Gold Vibe & Orbiting Sparks */
        div.stButton > button {
            position: relative !important;
            overflow: hidden !important;
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.88) 0%, rgba(13, 19, 33, 0.96) 100%) !important;
            color: #F8FAFC !important;
            border: 1.5px solid rgba(255, 215, 0, 0.55) !important;
            border-radius: 11px !important;
            font-weight: 750 !important;
            letter-spacing: 0.4px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 12px rgba(255, 193, 7, 0.35) !important;
            animation: secondaryOrbitSparks 4s ease-in-out infinite !important;
            transition: all 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        }
        div.stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 60%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.3), transparent);
            animation: buttonShimmerBeam 5.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
            pointer-events: none;
        }
        div.stButton > button:hover {
            background: linear-gradient(110deg, #FFE885 0%, #FFC107 30%, #D4AF37 70%, #FFB300 100%) !important;
            color: #0A0E17 !important;
            border-color: #FFFFFF !important;
            font-weight: 850 !important;
            transform: translateY(-2.5px) scale(1.035) !important;
            animation: goldParticleOrbit 3s ease-in-out infinite !important;
            box-shadow: 0 6px 28px rgba(255, 193, 7, 0.75), 0 0 45px rgba(255, 152, 0, 0.55) !important;
        }
        div.stButton > button:active {
            transform: translateY(1px) scale(0.98) !important;
        }

        /* Glassmorphism Docent Answer Box with Electric Gold Glow */
        .docent-box {
            background: linear-gradient(135deg, rgba(20, 26, 44, 0.88) 0%, rgba(10, 14, 25, 0.95) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 215, 0, 0.45);
            border-left: 6px solid #FFC107;
            animation: electricGoldPulse 5s ease-in-out infinite;
            padding: 1.7rem;
            border-radius: 14px;
            margin: 1.2rem 0;
            line-height: 1.8;
            font-size: 1.05rem;
            color: #F8FAFC;
            box-shadow: 0 10px 35px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 235, 150, 0.15);
        }

        /* Stamped 24K Gold Citation Medallions */
        .citation-num {
            background: radial-gradient(circle at 35% 30%, #FFF3B0 0%, #FFC107 40%, #D4AF37 80%, #996515 100%);
            color: #0A0E17;
            font-weight: 900;
            font-size: 0.88rem;
            padding: 3px 11px;
            border-radius: 7px;
            margin-right: 8px;
            box-shadow: 0 0 15px rgba(255, 193, 7, 0.65), inset 0 1px 2px rgba(255, 255, 255, 0.8);
            display: inline-block;
            border: 1px solid rgba(255, 255, 255, 0.4);
        }

        /* Floating Citation Cards with Gold Edge Lighting */
        .citation-card {
            background: linear-gradient(140deg, rgba(26, 34, 52, 0.75) 0%, rgba(13, 18, 32, 0.9) 100%);
            backdrop-filter: blur(12px);
            border: 1.5px solid rgba(212, 175, 55, 0.3);
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .citation-card:hover {
            border-color: #FFD700;
            transform: translateY(-5px) scale(1.018);
            box-shadow: 0 15px 40px -5px rgba(255, 193, 7, 0.4), 0 0 25px rgba(255, 152, 0, 0.25);
        }

        .meta-tag {
            background: rgba(51, 65, 85, 0.65);
            color: #E2E8F0;
            border: 1px solid rgba(212, 175, 55, 0.3);
            padding: 3px 11px;
            border-radius: 6px;
            font-size: 0.81rem;
            margin-right: 6px;
            display: inline-block;
            margin-bottom: 4px;
            transition: all 0.2s ease;
        }
        .meta-tag:hover {
            border-color: #FFD700;
            color: #FFE082;
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.35);
        }

        .score-pill {
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.25) 0%, rgba(212, 175, 55, 0.15) 100%);
            color: #FFE082;
            border: 1px solid #FFC107;
            padding: 3px 12px;
            border-radius: 9999px;
            font-size: 0.80rem;
            font-weight: 700;
            box-shadow: 0 0 12px rgba(255, 193, 7, 0.3);
            display: inline-block;
        }

        .badge-verified {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.2) 100%);
            color: #34D399;
            border: 1px solid #10B981;
            padding: 5px 15px;
            border-radius: 9999px;
            font-size: 0.83rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.4), inset 0 0 8px rgba(16, 185, 129, 0.15);
            display: inline-block;
            animation: pulseGlow 4s ease-in-out infinite;
        }
        .badge-refusal {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(220, 38, 38, 0.2) 100%);
            color: #F87171;
            border: 1px solid #EF4444;
            padding: 5px 15px;
            border-radius: 9999px;
            font-size: 0.83rem;
            font-weight: 700;
            box-shadow: 0 0 16px rgba(239, 68, 68, 0.4);
            display: inline-block;
        }

        /* Sidebar Cyber-Dark Gold Architecture */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060911 0%, #0C101D 100%) !important;
            border-right: 1px solid rgba(212, 175, 55, 0.3) !important;
            box-shadow: 4px 0 25px rgba(0, 0, 0, 0.5) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(22, 30, 49, 0.85) 0%, rgba(13, 19, 33, 0.95) 100%) !important;
            border: 1px solid rgba(212, 175, 55, 0.35) !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            margin-bottom: 10px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(212, 175, 55, 0.08) !important;
            transition: all 0.25s ease !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"]:hover {
            border-color: #FFD700 !important;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.4) !important;
            transform: translateY(-2px) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"] > div,
        [data-testid="stSidebar"] [data-testid="stMetricValue"] * {
            font-size: 1.08rem !important;
            font-weight: 800 !important;
            color: #FFE082 !important;
            white-space: normal !important;
            text-overflow: unset !important;
            overflow: visible !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            line-height: 1.35 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
            font-size: 0.80rem !important;
            color: #94A3B8 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.6px !important;
        }

        /* Gold Framed Artwork Images */
        [data-testid="stImage"] {
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6), 0 0 1px 1.5px rgba(212, 175, 55, 0.35) !important;
            transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        }
        [data-testid="stImage"]:hover {
            transform: translateY(-5px) scale(1.025) !important;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.75), 0 0 28px rgba(255, 215, 0, 0.55), 0 0 2px 2px #FFD700 !important;
        }
        [data-testid="stImage"] img {
            border-radius: 12px !important;
            transition: transform 0.4s ease !important;
        }
        [data-testid="stImage"]:hover img {
            transform: scale(1.04) !important;
        }

        /* Royal Gold Navigation Tabs */
        [data-testid="stTabs"] [role="tablist"] {
            gap: 8px !important;
            border-bottom: 2px solid rgba(212, 175, 55, 0.25) !important;
            padding-bottom: 4px !important;
        }
        [data-testid="stTabs"] button[role="tab"],
        button[data-baseweb="tab"] {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            color: #94A3B8 !important;
            border-radius: 12px 12px 0 0 !important;
            transition: all 0.28s ease !important;
            padding: 10px 22px !important;
            background: transparent !important;
            border: none !important;
        }
        [data-testid="stTabs"] button[role="tab"]:hover,
        button[data-baseweb="tab"]:hover {
            color: #FFE082 !important;
            background: rgba(255, 215, 0, 0.08) !important;
            text-shadow: 0 0 12px rgba(255, 215, 0, 0.5) !important;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #FFD700 !important;
            border-bottom: 3.5px solid #FFD700 !important;
            background: linear-gradient(180deg, rgba(255, 215, 0, 0.16) 0%, rgba(255, 215, 0, 0.02) 100%) !important;
            box-shadow: 0 6px 20px -2px rgba(255, 215, 0, 0.5) !important;
            text-shadow: 0 0 15px rgba(255, 215, 0, 0.8) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #FFD700 !important;
            box-shadow: 0 0 12px #FFD700, 0 0 20px #FF9800 !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
            background-color: rgba(212, 175, 55, 0.3) !important;
        }

        /* 24K Gold Molten Slider */
        [data-baseweb="slider"] div[role="slider"] {
            background-color: #FFD700 !important;
            border: 2px solid #FFF3B0 !important;
            box-shadow: 0 0 16px rgba(255, 215, 0, 0.95), 0 0 28px rgba(255, 152, 0, 0.7) !important;
            transform: scale(1.15) !important;
        }
        [data-baseweb="slider"] div[data-testid="stSliderTickBar"] {
            background-color: rgba(212, 175, 55, 0.3) !important;
        }
        [data-baseweb="slider"] div > div {
            background: linear-gradient(90deg, #FFC107 0%, #FFD700 100%) !important;
        }

        /* Sleek Glassmorphic Alert & Success Containers */
        [data-testid="stAlert"] {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(13, 148, 136, 0.09) 100%) !important;
            border: 1px solid rgba(16, 185, 129, 0.5) !important;
            border-left: 5px solid #10B981 !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.22), inset 0 0 12px rgba(16, 185, 129, 0.08) !important;
            backdrop-filter: blur(10px) !important;
            color: #E6FFFA !important;
        }

        /* Glowing Text Input Fields with Gold Focus Ring */
        [data-testid="stTextInput"] input {
            background: rgba(15, 23, 42, 0.85) !important;
            border: 1.5px solid rgba(212, 175, 55, 0.4) !important;
            border-radius: 11px !important;
            color: #F8FAFC !important;
            padding: 0.65rem 1rem !important;
            transition: all 0.25s ease !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #FFD700 !important;
            box-shadow: 0 0 24px rgba(255, 215, 0, 0.6) !important;
        }

        /* Smooth Page Scrolling & 24K Molten Gold Custom Scrollbar */
        html {
            scroll-behavior: smooth !important;
        }
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #060911;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #FFE885 0%, #FFC107 40%, #FF9800 80%, #D4AF37 100%);
            border-radius: 9999px;
            border: 2px solid #060911;
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFD700 60%, #FF9800 100%);
            box-shadow: 0 0 18px rgba(255, 215, 0, 0.9);
        }

        /* Section Fade-In Animation on Scroll & Tab Switching */
        @keyframes sectionFadeInUp {
            0% {
                opacity: 0;
                transform: translateY(16px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        [data-testid="stTabs"] [data-testid="stVerticalBlock"] > div {
            animation: sectionFadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        /* Audio Equalizer Frequency Waveform Animations */
        @keyframes eqBounce {
            0%, 100% { height: 5px; opacity: 0.6; }
            50% { height: 24px; opacity: 1; }
        }
        .audio-equalizer {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            height: 32px;
            padding: 6px 16px;
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.9) 0%, rgba(13, 19, 33, 0.95) 100%);
            border: 1px solid rgba(255, 215, 0, 0.55);
            border-radius: 9999px;
            box-shadow: 0 0 20px rgba(255, 193, 7, 0.4);
            margin: 8px 0;
        }
        .eq-bar {
            width: 3.5px;
            border-radius: 9999px;
            background: linear-gradient(180deg, #FFE885 0%, #FFC107 45%, #FF9800 100%);
            animation: eqBounce 1.2s ease-in-out infinite;
            box-shadow: 0 0 6px rgba(255, 215, 0, 0.6);
        }
        .bar-1 { animation-delay: 0.0s; }
        .bar-2 { animation-delay: 0.18s; }
        .bar-3 { animation-delay: 0.36s; }
        .bar-4 { animation-delay: 0.54s; }
        .bar-5 { animation-delay: 0.22s; }
        .bar-6 { animation-delay: 0.4s; }
        .bar-7 { animation-delay: 0.12s; }
        .bar-8 { animation-delay: 0.48s; }
        .bar-9 { animation-delay: 0.28s; }
        .bar-10 { animation-delay: 0.6s; }
        .bar-11 { animation-delay: 0.15s; }
        .bar-12 { animation-delay: 0.33s; }
        .eq-label {
            margin-left: 10px;
            font-size: 0.83rem;
            font-weight: 750;
            color: #FFE082;
            letter-spacing: 0.4px;
            text-shadow: 0 0 8px rgba(255, 215, 0, 0.5);
        }

        /* Generation Status Spinner & Pulse Animations */
        [data-testid="stSpinner"] > div {
            border-top-color: #FFD700 !important;
            border-right-color: #FF9800 !important;
            border-bottom-color: rgba(255, 215, 0, 0.2) !important;
            border-left-color: rgba(255, 215, 0, 0.2) !important;
            filter: drop-shadow(0 0 14px rgba(255, 215, 0, 0.85));
        }
        .generation-pulse-banner {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 18px;
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.08) 100%);
            border: 1.5px solid rgba(255, 215, 0, 0.55);
            border-radius: 12px;
            color: #FFE082;
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            box-shadow: 0 0 25px rgba(255, 193, 7, 0.35);
            animation: electricGoldPulse 2.5s ease-in-out infinite;
            margin-bottom: 12px;
        }
        .generation-ring {
            width: 16px;
            height: 16px;
            border: 2.5px solid rgba(255, 215, 0, 0.3);
            border-top: 2.5px solid #FFD700;
            border-radius: 50%;
            animation: ringSpin 0.9s linear infinite;
        }
        @keyframes ringSpin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Voice Audio Input Microphone Animation & Design */
        [data-testid="stAudioInput"] {
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.85) 0%, rgba(13, 19, 33, 0.95) 100%) !important;
            border: 1.5px solid rgba(255, 215, 0, 0.45) !important;
            border-radius: 14px !important;
            padding: 8px 12px !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4), inset 0 0 12px rgba(255, 215, 0, 0.06) !important;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        }
        [data-testid="stAudioInput"]:hover {
            border-color: #FFD700 !important;
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.5), inset 0 0 15px rgba(255, 215, 0, 0.12) !important;
        }
        [data-testid="stAudioInput"] button {
            border-radius: 9999px !important;
            background: linear-gradient(135deg, #FFC107 0%, #FF9800 100%) !important;
            color: #0A0E17 !important;
            box-shadow: 0 0 16px rgba(255, 193, 7, 0.7) !important;
            animation: micRecordingPulse 2.5s ease-in-out infinite !important;
        }
        @keyframes micRecordingPulse {
            0%, 100% {
                box-shadow: 0 0 14px rgba(255, 193, 7, 0.6);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 0 28px rgba(239, 68, 68, 0.85), 0 0 45px rgba(255, 193, 7, 0.6);
                transform: scale(1.06);
            }
        }

        /* Docent Audio Narration Player Styling */
        @keyframes timelinePulseGlow {
            0% {
                background-position: 0% 50%;
                filter: drop-shadow(0 0 4px rgba(255, 215, 0, 0.6));
            }
            50% {
                background-position: 100% 50%;
                filter: drop-shadow(0 0 12px rgba(255, 152, 0, 0.95));
            }
            100% {
                background-position: 0% 50%;
                filter: drop-shadow(0 0 4px rgba(255, 215, 0, 0.6));
            }
        }
        @keyframes timelineTravel {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .audio-timeline-track {
            width: 100%;
            height: 7px;
            background: rgba(255, 215, 0, 0.12);
            border-radius: 9999px;
            position: relative;
            overflow: hidden;
            margin-top: 6px;
            margin-bottom: 8px;
            border: 1px solid rgba(255, 215, 0, 0.35);
            box-shadow: 0 0 12px rgba(255, 193, 7, 0.25);
        }
        .audio-timeline-wave {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background: linear-gradient(90deg, 
                transparent 0%, 
                rgba(255, 215, 0, 0.3) 20%, 
                #FFD700 45%, 
                #FFFFFF 50%, 
                #FFD700 55%, 
                rgba(255, 152, 0, 0.3) 80%, 
                transparent 100%);
            background-size: 200% 100%;
            animation: timelineTravel 2.2s linear infinite;
        }

        audio {
            width: 100% !important;
            border-radius: 9999px !important;
            border: 1.5px solid rgba(255, 215, 0, 0.5) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 18px rgba(255, 193, 7, 0.35) !important;
            background: #0E1424 !important;
            transition: all 0.25s ease !important;
        }
        audio:hover {
            border-color: #FFD700 !important;
            box-shadow: 0 6px 26px rgba(255, 215, 0, 0.6) !important;
        }
        audio::-webkit-media-controls-timeline {
            border-radius: 9999px !important;
            background: linear-gradient(90deg, #FFE885 0%, #FFC107 25%, #FF9800 50%, #D4AF37 75%, #FFE885 100%) !important;
            background-size: 300% 100% !important;
            animation: timelinePulseGlow 3s ease infinite !important;
            height: 7px !important;
            box-shadow: 0 0 12px rgba(255, 193, 7, 0.6) !important;
        }
        audio::-webkit-media-controls-current-time-display,
        audio::-webkit-media-controls-time-remaining-display {
            color: #FFE082 !important;
            font-weight: 750 !important;
            text-shadow: 0 0 6px rgba(255, 215, 0, 0.5) !important;
        }
        audio::-webkit-media-controls-play-button {
            filter: invert(85%) sepia(85%) saturate(1200%) hue-rotate(5deg) !important;
            transform: scale(1.1) !important;
            transition: transform 0.2s ease !important;
        }
        audio::-webkit-media-controls-play-button:hover {
            transform: scale(1.25) !important;
        }

        /* Curatorial Section Badges & Box Dividers */
        .section-hero-banner {
            background: linear-gradient(135deg, rgba(22, 30, 49, 0.7) 0%, rgba(13, 19, 33, 0.85) 100%);
            border-left: 4px solid #FFC107;
            border-radius: 0 12px 12px 0;
            padding: 10px 18px;
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .section-hero-title {
            font-family: 'Cinzel', serif;
            font-size: 1.35rem;
            font-weight: 800;
            color: #FFD700;
            letter-spacing: 0.5px;
            text-shadow: 0 0 12px rgba(255, 215, 0, 0.5);
        }
        .section-hero-caption {
            color: #94A3B8;
            font-size: 0.88rem;
            margin-top: 3px;
        }

        /* Expanders with Frosted Glass and Gold Border */
        [data-testid="stExpander"] {
            border: 1px solid rgba(212, 175, 55, 0.35) !important;
            border-radius: 12px !important;
            background: rgba(16, 23, 41, 0.6) !important;
            backdrop-filter: blur(8px) !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.25s ease !important;
        }
        [data-testid="stExpander"]:hover {
            border-color: #FFD700 !important;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3) !important;
        }
        [data-testid="stExpander"] summary {
            color: #FFE082 !important;
            font-weight: 700 !important;
        }

        /* Toggle Switches with Glowing Gold Track */
        [data-testid="stToggle"] span[role="checkbox"][aria-checked="true"] {
            background-color: #FFD700 !important;
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.8) !important;
        }

        /* Artwork Details Grid */
        .artwork-detail-box {
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.75) 0%, rgba(13, 18, 32, 0.9) 100%);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-radius: 12px;
            padding: 1.0rem 1.3rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            transition: all 0.25s ease;
        }
        .artwork-detail-box:hover {
            border-color: #FFD700;
            box-shadow: 0 0 22px rgba(255, 215, 0, 0.35);
        }
        .artwork-detail-title {
            color: #FBBF24;
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.4rem;
        }
        .meta-field {
            display: inline-block;
            margin-right: 14px;
            font-size: 0.86rem;
            color: #E2E8F0;
        }
        .meta-field strong {
            color: #94A3B8;
        }

        .answer-section-title {
            font-family: 'Cinzel', serif;
            color: #FFD700;
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: 0.6px;
            text-shadow: 0 0 12px rgba(255, 215, 0, 0.45);
        }

        .voice-box {
            background: linear-gradient(135deg, rgba(26, 34, 52, 0.8) 0%, rgba(13, 18, 32, 0.9) 100%);
            border: 1px solid rgba(255, 215, 0, 0.45);
            border-radius: 12px;
            padding: 0.75rem 1.1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 4px 18px rgba(255, 193, 7, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_classic_css():
    """Inject classic museum-themed CSS styling (revertible baseline)."""
    st.markdown(
        """
        <style>
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
        .voice-box {
            background: #161c28;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render top brand header with optional gold architectural ornament."""
    theme = os.getenv("UI_THEME", "vibrant").strip().lower()
    st.markdown('<div class="main-header">🏛️ MuseGuide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI Museum Exploration & Retrieval-Augmented Generation (RAG) | Grounded in Cleveland Museum of Art Open Access Records (CC0 Public Domain)</div>',
        unsafe_allow_html=True,
    )
    if theme != "classic":
        st.markdown(
            '<div class="gold-ornament"><span class="gold-ornament-line"></span><span class="gold-ornament-gem">✦ &nbsp; 🏛️ &nbsp; ✦</span><span class="gold-ornament-line"></span></div>',
            unsafe_allow_html=True,
        )


def render_section_banner(title: str, caption: str):
    """Render styled section header banner with revertible fallback."""
    theme = os.getenv("UI_THEME", "vibrant").strip().lower()
    if theme != "classic":
        st.markdown(
            f'<div class="section-hero-banner">'
            f'<div class="section-hero-title">{title}</div>'
            f'<div class="section-hero-caption">{caption}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### {title}")
        st.caption(caption)


def render_docent_answer(answer_text: str, is_grounded: bool, model_used: str):
    """Render prominent direct answer card with proper Markdown parsing and status badge."""
    theme = os.getenv("UI_THEME", "vibrant").strip().lower()
    if is_grounded:
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
            f'<span class="badge-verified">✓ Grounded in Verified Museum Records</span>'
            f'<span style="color:#FFE082; font-size:0.82rem; font-weight:600;">⚡ Engine: {model_used}</span>'
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


def render_speak_control(answer_text: str, tts_engine):
    """Render manual '🔊 Speak Answer' button and audio player. Never autoplays."""
    if not answer_text or not answer_text.strip():
        return

    theme = os.getenv("UI_THEME", "vibrant").strip().lower()
    col_spk, col_info = st.columns([1.3, 3.7])
    with col_spk:
        speak_btn = st.button(
            "🔊 Speak Answer",
            key="speak_docent_btn",
            help="Listen to a voice narration of the direct answer. Citations and metadata are omitted from spoken audio.",
        )
    with col_info:
        engine_label = tts_engine.engine_name if tts_engine else "Offline Speech"
        st.caption(f"Speech Engine: **{engine_label}** (Rate: {os.getenv('DOCENT_VOICE_RATE', '135')} WPM)")

    if speak_btn:
        status_placeholder = st.empty()
        with st.spinner("Synthesizing docent voice..."):
            if theme != "classic":
                status_placeholder.markdown(
                    """
                    <div class="audio-equalizer" style="margin-bottom:12px;">
                        <div class="eq-bar bar-1"></div>
                        <div class="eq-bar bar-3"></div>
                        <div class="eq-bar bar-5"></div>
                        <div class="eq-bar bar-7"></div>
                        <div class="eq-bar bar-9"></div>
                        <div class="eq-bar bar-11"></div>
                        <span class="eq-label">⚡ Neural Voice Engine Synthesizing Narration...</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            try:
                audio_bytes = tts_engine.synthesize_wav(answer_text)
                st.session_state["tts_audio_cache"] = audio_bytes
            except Exception as e:
                st.error(f"Speech synthesis error: {e}")
            finally:
                status_placeholder.empty()

    # Display audio player if audio was synthesized
    cached_audio = st.session_state.get("tts_audio_cache")
    if cached_audio:
        if theme != "classic":
            st.markdown(
                """
                <div class="audio-equalizer" style="margin-top:6px; margin-bottom:6px;">
                    <div class="eq-bar bar-1"></div>
                    <div class="eq-bar bar-2"></div>
                    <div class="eq-bar bar-3"></div>
                    <div class="eq-bar bar-4"></div>
                    <div class="eq-bar bar-5"></div>
                    <div class="eq-bar bar-6"></div>
                    <div class="eq-bar bar-7"></div>
                    <div class="eq-bar bar-8"></div>
                    <div class="eq-bar bar-9"></div>
                    <div class="eq-bar bar-10"></div>
                    <div class="eq-bar bar-11"></div>
                    <div class="eq-bar bar-12"></div>
                    <span class="eq-label">🎧 Docent Voice Narration Active</span>
                </div>
                <div class="audio-timeline-track">
                    <div class="audio-timeline-wave"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.audio(cached_audio, format="audio/wav", autoplay=False)

