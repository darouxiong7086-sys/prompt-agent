"""
Vibe-to-Prompt Agent - Streamlit Frontend v4.0 "Mainframe" Edition
==================================================================

Complete CodeNest visual refactor - all backend threading,
streaming, HITL, and DeepSeek/Claude routing logic preserved intact.

Execution model:
1. Generate a frontend-owned thread_id.
2. Start run_pipeline(..., thread_id=thread_id) to let DeepSeek produce dynamic directions.
3. Lock those directions in session_state for a HITL selection step.
4. Resume with the selected direction, stream Claude chit-chat, then render the final XML prompt.
5. Clear backend streaming/reasoning caches only after final delivery.
"""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import threading
import time
import uuid
from datetime import datetime
from html import escape
from typing import Any, Dict, Iterator, List, Optional

import streamlit as st
import streamlit.components.v1 as components

import graph
from vibe_config import VIBE_TECH_MAPPING


st.set_page_config(
    page_title="CodeNest - Vibe-to-Prompt Agent v4.0",
    page_icon="CN",
    layout="centered",
)


# ============================================================================
# Constants
# ============================================================================

POSITIVE_SAMPLES_FILE = "positive_samples.json"
_POSITIVE_SAMPLE_LOCK = threading.Lock()

POLL_INTERVAL_SECONDS = 0.055
STREAM_TIMEOUT_SECONDS = 240
STREAM_GRACE_SECONDS = 0.35

OFFICE_QUOTES: List[str] = [
    "Good design is as little design as possible.",
    "Less, but better - because it concentrates on the essential.",
    "A designer knows he has achieved perfection not when there is nothing left to add, but when there is nothing left to take away.",
    "The mainframe is not a machine; it is an attitude.",
    "Form follows function. And function this time is prompt engineering.",
    "Not busy. Not minimal. Just right.",
    "Systems, not surfaces. Contracts, not copy.",
    "Swiss precision: every pixel, every token, every boundary condition.",
]

COMPLEXITY_SIGNALS: List[str] = [
    "like", "as if", "resembles", "movie", "anime", "game", "novel",
    "apocalyptic", "oppressive", "suffocating", "dream", "surreal",
    "wasteland", "gothic", "dark fantasy", "cyber", "full-stack",
    "holographic", "neon", "blade runner", "MOSS", "wandering earth",
    "texture", "synesthesia",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

@font-face {
    font-family: 'Helvetica Now Display';
    src: local('Helvetica Neue'), local('Helvetica'), local('Arial');
    font-display: swap;
}

:root {
    --mf-bg: #000000;
    --mf-text: #ffffff;
    --mf-muted: rgba(255, 255, 255, 0.55);
    --mf-line: rgba(255, 255, 255, 0.08);
    --mf-glass: rgba(255, 255, 255, 0.02);
    --mf-glass-hover: rgba(255, 255, 255, 0.06);
    --mf-border: rgba(255, 255, 255, 0.12);
    --mf-font-display: 'Helvetica Now Display', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mf-font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-body: var(--mf-font-body);
    --mf-font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--mf-bg) !important;
    color: var(--mf-text) !important;
    font-family: var(--mf-font-body) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

#MainMenu, footer {
    display: none !important;
}

section.main > div, .block-container {
    width: min(820px, calc(100vw - 3rem)) !important;
    max-width: min(820px, calc(100vw - 3rem)) !important;
    padding-top: 1.6rem !important;
    padding-bottom: 4rem !important;
    position: relative;
    z-index: 2;
}

/* ---- Mainframe Navbar ---- */
.mf-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 0.8rem;
    border-bottom: 1px solid var(--mf-line);
    margin-bottom: 1.6rem;
}

.mf-logo {
    font-family: var(--mf-font-display);
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #000;
    background: #fff;
    display: inline-block;
    padding: 0.04em 0.35em 0.04em 0.5em;
    line-height: 1.2;
}

.mf-logo .star {
    font-size: 22px;
    vertical-align: middle;
    margin-left: 0.06em;
}

/* ---- Hero Blur Intro ---- */
.mf-blur-intro {
    margin: 1.2rem 0 1.8rem;
}

.mf-blur-line {
    font-family: var(--mf-font-display);
    font-size: clamp(24px, 4vw, 36px);
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--mf-text);
    filter: blur(4px);
    line-height: 1.2;
    margin: 0;
}

/* ---- Typewriter ---- */
.mf-typewriter {
    font-family: var(--mf-font-body);
    font-size: 15px;
    font-weight: 400;
    color: var(--mf-muted);
    min-height: 1.6em;
    margin: 0.2rem 0 1.4rem;
    letter-spacing: 0.01em;
}

.mf-cursor {
    display: inline-block;
    width: 1px;
    height: 1.1em;
    background: #000;
    vertical-align: text-bottom;
    margin-left: 2px;
    animation: blink 1s step-end infinite;
}

.mf-cursor.done {
    animation: none;
    opacity: 0;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* ---- Cards & Panels ---- */
.mf-panel {
    position: relative;
    background: var(--mf-glass) !important;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid var(--mf-border) !important;
    border-radius: 12px;
    padding: 1.1rem 1.15rem 1.25rem;
    margin: 1rem 0 1.15rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.mf-panel-thin {
    padding: 0.85rem 1rem;
    margin: 0.85rem 0;
}

/* ---- Text Input ---- */
.stTextArea label, .stRadio label, .stFormSubmitButton label {
    color: var(--mf-muted) !important;
    font-family: var(--mf-font-body) !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600 !important;
}

.stTextArea textarea {
    background: rgba(0, 0, 0, 0.25) !important;
    border: 1px solid var(--mf-border) !important;
    color: var(--mf-text) !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    line-height: 1.62 !important;
    font-family: var(--mf-font-body) !important;
}

.stTextArea textarea:focus {
    border-color: rgba(255, 255, 255, 0.35) !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08) !important;
}

.stTextArea textarea::placeholder {
    color: rgba(255, 255, 255, 0.20) !important;
}

/* ---- Pill Buttons ---- */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 999px !important;
    font-family: var(--mf-font-body) !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease !important;
    background: #fff !important;
    color: #000 !important;
    border: 1px solid rgba(0, 0, 0, 0.10) !important;
    box-shadow: none !important;
    padding: 0.3em 1.25em !important;
    min-height: 2.25rem !important;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #000 !important;
    color: #fff !important;
    border-color: rgba(255, 255, 255, 0.20) !important;
    transform: none !important;
    box-shadow: none !important;
}

div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: #fff !important;
    color: #000 !important;
    border-color: rgba(0, 0, 0, 0.15) !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: #000 !important;
    color: #fff !important;
    border-color: rgba(255, 255, 255, 0.20) !important;
}

/* ---- Radio (Pill-style) ---- */
div[data-testid="stRadio"] > div {
    position: relative;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 8px;
}

div[data-testid="stRadio"] label {
    display: inline-flex !important;
    align-items: center !important;
    padding: 0.3em 1.25em !important;
    border-radius: 999px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    background: #fff !important;
    color: #000 !important;
    border: 1px solid rgba(0, 0, 0, 0.10) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    min-height: unset !important;
}

div[data-testid="stRadio"] label:hover {
    background: #000 !important;
    color: #fff !important;
    border-color: rgba(255, 255, 255, 0.20) !important;
}

div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] input:checked + div,
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] + div[data-checked="true"] {
    background: #000 !important;
    color: #fff !important;
    border-color: rgba(255, 255, 255, 0.20) !important;
}

div[data-testid="stRadio"] input {
    display: none !important;
}

/* ---- Code Blocks ---- */
.stCodeBlock {
    border-radius: 8px !important;
    border: 1px solid var(--mf-border) !important;
    overflow: hidden;
    background: rgba(0, 0, 0, 0.4) !important;
}

.stCodeBlock pre {
    background: rgba(0, 0, 0, 0.5) !important;
}

.stCodeBlock code {
    color: var(--mf-text) !important;
    font-family: var(--mf-font-mono) !important;
    font-size: 13px !important;
}

/* ---- Dividers ---- */
hr {
    border-color: var(--mf-line) !important;
    margin: 1.4rem 0 !important;
}

/* ---- Metric / Quality ---- */
div[data-testid="stMetric"] {
    background: var(--mf-glass) !important;
    border: 1px solid var(--mf-line) !important;
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
}

div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
    color: var(--mf-muted) !important;
}

/* ---- Direction Detail Cards ---- */
.mf-direction-card {
    padding: 0.75rem 0.9rem;
    margin: 0.5rem 0;
    border: 1px solid var(--mf-line);
    border-radius: 8px;
    background: var(--mf-glass);
}

.mf-direction-card strong {
    display: block;
    font-size: 0.96rem;
    font-weight: 600;
    color: var(--mf-text);
    margin-bottom: 0.2rem;
}

.mf-direction-card ul {
    margin: 0.3rem 0 0;
    padding-left: 1rem;
    color: var(--mf-muted);
    font-size: 0.82rem;
    list-style: none;
}

.mf-direction-card ul li::before {
    content: "- ";
}

/* ---- Chips & Badges ---- */
.mf-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    margin: 0 0.3rem 0.3rem 0;
    font-size: 0.72rem;
    font-weight: 500;
    border: 1px solid var(--mf-line);
    background: var(--mf-glass);
    color: var(--mf-muted);
}

.mf-chip.ok {
    color: var(--mf-text);
    border-color: rgba(255, 255, 255, 0.20);
}

/* ---- Copy Pill ---- */
.mf-copy-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 0.4em 1em;
    font-size: 14px;
    font-weight: 500;
    background: #000;
    color: #fff;
    border: 1px solid var(--mf-border);
    cursor: pointer;
    transition: all 0.2s ease;
}

.mf-copy-pill:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #fff;
}

.mf-copy-pill svg {
    width: 14px;
    height: 14px;
}

.mf-direction-actions {
    margin-top: 0.5rem;
}

/* -------------------------------------------------------------------------
   CodeNest Dark Green Liquid Glass Skin
   Visual override only: backend state machine remains untouched.
------------------------------------------------------------------------- */
:root {
    --cn-bg: #070b0a;
    --cn-panel: rgba(255, 255, 255, 0.01);
    --cn-green: #5ed29c;
    --cn-cyan: #61f3e8;
    --cn-deep: #0f6f51;
    --cn-ink: #e8fff5;
    --cn-muted: rgba(223, 255, 241, 0.68);
    --cn-line: rgba(255, 255, 255, 0.10);
    --cn-glow: rgba(94, 210, 156, 0.18);
    --font-body: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

html,
body,
[data-testid="stAppViewContainer"],
.stApp {
    background: #070b0a !important;
    color: var(--cn-ink) !important;
    font-family: var(--font-body) !important;
}

section.main > div,
.block-container {
    width: min(960px, calc(100vw - 2rem)) !important;
    max-width: min(960px, calc(100vw - 2rem)) !important;
    padding-top: 3.4rem !important;
}

.codenest-video {
    position: fixed;
    inset: 0;
    z-index: 0;
    width: 100vw;
    height: 100vh;
    object-fit: cover;
    opacity: 0.6;
    pointer-events: none;
    filter: saturate(1.08) contrast(1.08) brightness(0.62);
}

.codenest-mask {
    position: fixed;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background:
        linear-gradient(90deg, #070b0a 0%, rgba(7, 11, 10, 0.82) 28%, rgba(7, 11, 10, 0.16) 100%),
        linear-gradient(0deg, #070b0a 0%, rgba(7, 11, 10, 0.82) 18%, rgba(7, 11, 10, 0.16) 58%);
}

.codenest-grid {
    position: fixed;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background:
        linear-gradient(90deg, transparent 24.94%, rgba(255,255,255,0.10) 25%, transparent 25.06%),
        linear-gradient(90deg, transparent 49.94%, rgba(255,255,255,0.10) 50%, transparent 50.06%),
        linear-gradient(90deg, transparent 74.94%, rgba(255,255,255,0.10) 75%, transparent 75.06%);
}

.codenest-glow {
    position: fixed;
    top: -92px;
    left: 50%;
    z-index: 1;
    width: min(780px, 88vw);
    height: 230px;
    transform: translateX(-50%);
    pointer-events: none;
    filter: blur(25px);
    opacity: 0.88;
}

.codenest-hero {
    position: relative;
    z-index: 2;
    margin: 0.2rem 0 1.55rem;
}

.cn-kicker,
.codenest-kicker {
    font-family: "Plus Jakarta Sans", var(--font-body) !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase;
    color: var(--cn-green) !important;
}

.codenest-title {
    margin: 0.42rem 0 0;
    max-width: 920px;
    color: #f3fbff;
    font-family: "Inter", var(--font-body);
    font-size: clamp(42px, 8vw, 78px);
    font-weight: 900;
    line-height: 0.92;
    letter-spacing: -0.055em;
    text-transform: uppercase;
}

.codenest-title .green-dot {
    color: var(--cn-green);
    text-shadow: 0 0 28px rgba(94, 210, 156, 0.58);
}

.cn-floating-card {
    width: 200px;
    height: 200px;
    transform: translateY(-50px);
    margin-bottom: -38px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    gap: 0.42rem;
}

.cn-floating-card .status-orbit {
    width: 48px;
    height: 48px;
    border-radius: 999px;
    border: 1px solid rgba(94, 210, 156, 0.44);
    box-shadow: 0 0 28px rgba(94, 210, 156, 0.18), inset 0 0 24px rgba(94, 210, 156, 0.08);
}

.cn-floating-card strong {
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #f5fbff;
}

.cn-floating-card span {
    color: var(--cn-muted);
    font-size: 0.78rem;
    line-height: 1.42;
}

.liquid-glass-card,
.mf-panel,
.mf-panel-thin,
.mf-direction-card {
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.01) !important;
    background-blend-mode: luminosity;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    border: 0 !important;
    border-radius: 18px;
    color: var(--cn-ink);
    box-shadow:
        inset 0 1px 1px rgba(255, 255, 255, 0.1),
        0 4px 20px rgba(94, 210, 156, 0.05),
        0 28px 90px rgba(0, 0, 0, 0.34);
}

.liquid-glass-card::before,
.mf-panel::before,
.mf-panel-thin::before,
.mf-direction-card::before,
.cn-floating-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1.4px;
    background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.08) 44%, rgba(94,210,156,0.36));
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.mf-direction-card {
    border-radius: 14px;
}

.mf-direction-card strong {
    color: #f4fbff;
}

.mf-direction-card,
.mf-direction-card ul,
.mf-direction-card li,
.stCaptionContainer,
.stMarkdown p {
    color: var(--cn-muted) !important;
}

.stTextArea label,
.stRadio label,
.stFormSubmitButton label {
    color: var(--cn-muted) !important;
    font-family: var(--font-body) !important;
}

.stTextArea textarea {
    background: rgba(3, 8, 7, 0.58) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    color: #f4fbff !important;
    border-radius: 14px !important;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.06);
}

.stTextArea textarea:focus {
    border-color: rgba(94, 210, 156, 0.72) !important;
    box-shadow: 0 0 0 1px rgba(94, 210, 156, 0.20), 0 0 30px rgba(94, 210, 156, 0.08) !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #ffffff !important;
    color: #06100c !important;
    border: 1px solid rgba(0, 0, 0, 0.10) !important;
    border-radius: 999px !important;
    font-size: 15px !important;
    padding: 0.3em 1.25em !important;
    transition: background 200ms ease, color 200ms ease, box-shadow 200ms ease, border-color 200ms ease !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: var(--cn-green) !important;
    color: #06100c !important;
    border-color: rgba(94, 210, 156, 0.76) !important;
    box-shadow: 0 0 26px rgba(94, 210, 156, 0.20) !important;
}

div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: #ffffff !important;
    color: #06100c !important;
}

.stCodeBlock {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 24px 70px rgba(0,0,0,0.34);
}

.stCodeBlock pre {
    background: rgba(3, 8, 7, 0.88) !important;
}

.stCodeBlock code {
    color: #d8ffed !important;
}

.mf-chip {
    color: rgba(223, 255, 241, 0.78);
    border-color: rgba(94, 210, 156, 0.16);
    background: rgba(94, 210, 156, 0.045);
}

.mf-chip.ok {
    color: #d8ffed;
    border-color: rgba(94, 210, 156, 0.30);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.018) !important;
    border-color: rgba(255,255,255,0.10) !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: #dfffee !important;
}

/* ---- CodeNest Readability Lock: every text layer must stay legible ---- */
.stApp,
.stApp *,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.42);
}

.stMarkdown,
.stMarkdown *,
.stCaptionContainer,
.stCaptionContainer *,
.stText,
.stText *,
.stStatusWidget,
.stStatusWidget *,
[data-testid="stNotificationContent"],
[data-testid="stNotificationContent"] *,
[data-testid="stAlert"],
[data-testid="stAlert"] *,
[data-testid="stExpander"],
[data-testid="stExpander"] *,
[data-testid="stMetric"],
[data-testid="stMetric"] *,
label,
p,
span,
small,
li,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] * {
    color: #f0fff8 !important;
}

.stCaptionContainer,
.stCaptionContainer *,
.mf-direction-card,
.mf-direction-card *,
.mf-panel,
.mf-panel *,
.mf-panel-thin,
.mf-panel-thin * {
    color: #e1fff1 !important;
}

.cn-kicker,
.codenest-kicker,
.codenest-title .green-dot,
.quality-title {
    color: #5ed29c !important;
    text-shadow: 0 0 24px rgba(94, 210, 156, 0.36);
}

.codenest-title,
.codenest-title * {
    color: #f7fffb !important;
}

.mf-typewriter {
    color: #e8fff5 !important;
    font-weight: 600;
}

.mf-cursor {
    background: #5ed29c !important;
}

.stTextArea textarea,
.stTextArea textarea::placeholder,
.stTextInput input,
.stSelectbox div,
.stRadio label,
.stRadio label *,
.stCheckbox label,
.stCheckbox label * {
    color: #f2fff9 !important;
}

.stTextArea textarea::placeholder {
    color: rgba(232, 255, 245, 0.56) !important;
}

.stButton > button,
.stFormSubmitButton > button,
.stButton > button *,
.stFormSubmitButton > button * {
    color: #06100c !important;
    text-shadow: none !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stButton > button:hover *,
.stFormSubmitButton > button:hover * {
    color: #06100c !important;
}

.stCodeBlock,
.stCodeBlock *,
.stCodeBlock pre,
.stCodeBlock code {
    color: #e8fff5 !important;
    text-shadow: none !important;
}

.mf-chip,
.mf-chip *,
.vibe-chip,
.quality-badge {
    color: #e8fff5 !important;
}

/* ---- Final CodeNest Legibility Override ---- */
html body .stApp,
html body .stApp p,
html body .stApp span,
html body .stApp div,
html body .stApp label,
html body .stApp small,
html body .stApp strong,
html body .stApp li,
html body .stApp h1,
html body .stApp h2,
html body .stApp h3,
html body .stApp h4,
html body .stApp [data-testid="stMarkdownContainer"],
html body .stApp [data-testid="stMarkdownContainer"] * {
    opacity: 1 !important;
    filter: none !important;
    color: #f4fff9 !important;
    -webkit-text-fill-color: #f4fff9 !important;
}

html body .stApp .codenest-title,
html body .stApp .codenest-title * {
    color: #f7fffb !important;
    -webkit-text-fill-color: #f7fffb !important;
}

html body .stApp .codenest-title .green-dot,
html body .stApp .cn-kicker,
html body .stApp .codenest-kicker,
html body .stApp .quality-title {
    color: #5ed29c !important;
    -webkit-text-fill-color: #5ed29c !important;
}

html body .stApp .mf-panel,
html body .stApp .mf-panel *,
html body .stApp .mf-panel-thin,
html body .stApp .mf-panel-thin *,
html body .stApp .mf-direction-card,
html body .stApp .mf-direction-card *,
html body .stApp .liquid-glass-card,
html body .stApp .liquid-glass-card * {
    opacity: 1 !important;
    color: #eafff6 !important;
    -webkit-text-fill-color: #eafff6 !important;
}

html body .stApp .mf-panel .cn-kicker,
html body .stApp .mf-panel-thin .cn-kicker,
html body .stApp .liquid-glass-card .cn-kicker {
    color: #5ed29c !important;
    -webkit-text-fill-color: #5ed29c !important;
}

html body .stApp .stTextArea label,
html body .stApp .stTextInput label,
html body .stApp .stRadio label,
html body .stApp .stCheckbox label,
html body .stApp .stSelectbox label,
html body .stApp [data-testid="stWidgetLabel"],
html body .stApp [data-testid="stWidgetLabel"] * {
    color: #e8fff5 !important;
    -webkit-text-fill-color: #e8fff5 !important;
    opacity: 1 !important;
}

html body .stApp textarea,
html body .stApp input,
html body .stApp textarea::placeholder,
html body .stApp input::placeholder {
    color: #f7fffb !important;
    -webkit-text-fill-color: #f7fffb !important;
    opacity: 1 !important;
}

html body .stApp textarea::placeholder,
html body .stApp input::placeholder {
    color: rgba(232, 255, 245, 0.68) !important;
    -webkit-text-fill-color: rgba(232, 255, 245, 0.68) !important;
}

html body .stApp .stButton > button,
html body .stApp .stFormSubmitButton > button,
html body .stApp .stButton > button *,
html body .stApp .stFormSubmitButton > button * {
    color: #06100c !important;
    -webkit-text-fill-color: #06100c !important;
    text-shadow: none !important;
}

html body .stApp .stCodeBlock,
html body .stApp .stCodeBlock *,
html body .stApp pre,
html body .stApp code {
    color: #e8fff5 !important;
    -webkit-text-fill-color: #e8fff5 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

@media (max-width: 640px) {
    .block-container {
        padding-top: 1rem !important;
        width: calc(100vw - 2rem) !important;
    }
    .mf-logo {
        font-size: 20px;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        flex-direction: column;
    }
}
</style>
"""

BACKDROP_HTML = """
<video class="codenest-video" muted autoplay loop playsinline preload="auto" data-hls-src="https://stream.mux.com/tLkHO1qZoaaQOUeVWo8hEBeGQfySP02EPS02BmnNFyXys.m3u8"></video>
<div class="codenest-mask"></div>
<div class="codenest-grid"></div>
<svg class="codenest-glow" viewBox="0 0 780 230" aria-hidden="true">
  <defs>
    <radialGradient id="codenestCyberGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#61f3e8" stop-opacity="0.82"/>
      <stop offset="42%" stop-color="#0f6f51" stop-opacity="0.48"/>
      <stop offset="100%" stop-color="#070b0a" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <ellipse cx="390" cy="92" rx="330" ry="76" fill="url(#codenestCyberGlow)"/>
</svg>
"""

INTERACTION_SCRIPT_HTML = """
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
(function(){
  const parentDoc = window.parent.document;
  const parentWin = window.parent;

  function installHlsVideo(){
    const video = parentDoc.querySelector('video.codenest-video');
    if (!video) {
      parentWin.setTimeout(installHlsVideo, 120);
      return;
    }
    if (video.dataset.codenestHlsInstalled === 'true') return;
    video.dataset.codenestHlsInstalled = 'true';

    const source = video.dataset.hlsSrc;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;

    if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = source;
    } else if (window.Hls && window.Hls.isSupported()) {
      const hls = new window.Hls({
        enableWorker: false,
        lowLatencyMode: false,
        backBufferLength: 30
      });
      hls.loadSource(source);
      hls.attachMedia(video);
      video.dataset.codenestHlsWorker = 'false';
    }
    video.play().catch(function(){});
  }

  function installTypewriter(){
    const target = parentDoc.getElementById('mf-typewriter');
    if (!target) {
      parentWin.setTimeout(installTypewriter, 120);
      return;
    }
    if (target.dataset.mfTyped === 'true') return;
    target.dataset.mfTyped = 'true';
    target.textContent = '';

    const text = "Describe the vibe. DeepSeek shapes the directions. Claude forges the XML prompt.";
    const cursor = parentDoc.createElement('span');
    cursor.className = 'mf-cursor';
    target.appendChild(cursor);
    let index = 0;

    function tick(){
      if (index < text.length) {
        target.insertBefore(parentDoc.createTextNode(text.charAt(index)), cursor);
        const current = text.charAt(index);
        index += 1;
        const delay = current === '.' || current === ',' || current === '?' ? 170 : 26 + Math.random() * 22;
        parentWin.setTimeout(tick, delay);
      } else {
        cursor.classList.add('done');
      }
    }
    parentWin.setTimeout(tick, 360);
  }

  installHlsVideo();
  installTypewriter();
})();
</script>
"""

# ============================================================================
# Session State
# ============================================================================

def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "current_stage": "idle",
        "saved_response": None,
        "thread_id": None,
        "user_input": "",
        "clarification_question": "",
        "clarification_options": [],
        "generated_prompt": "",
        "matched_vibes": [],
        "dynamic_directions": [],
        "selected_dynamic_direction": {},
        "direction_choice_id": "",
        "quality_snapshot": {},
        "last_streamed_chit": "",
        "liked": False,
        "error": None,
        "office_quote": random.choice(OFFICE_QUOTES),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_all() -> None:
    st.session_state.current_stage = "idle"
    st.session_state.saved_response = None
    st.session_state.thread_id = None
    st.session_state.user_input = ""
    st.session_state.clarification_question = ""
    st.session_state.clarification_options = []
    st.session_state.generated_prompt = ""
    st.session_state.matched_vibes = []
    st.session_state.dynamic_directions = []
    st.session_state.selected_dynamic_direction = {}
    st.session_state.direction_choice_id = ""
    st.session_state.quality_snapshot = {}
    st.session_state.last_streamed_chit = ""
    st.session_state.liked = False
    st.session_state.error = None
    st.session_state.office_quote = random.choice(OFFICE_QUOTES)


def _apply_graph_response(result: Dict[str, Any]) -> None:
    st.session_state.saved_response = result
    st.session_state.thread_id = result.get("thread_id") or st.session_state.thread_id
    st.session_state.clarification_question = result.get("clarification_question", "") or ""
    st.session_state.clarification_options = result.get("clarification_options", []) or []
    st.session_state.generated_prompt = result.get("generated_prompt", "") or ""
    st.session_state.matched_vibes = result.get("matched_vibes", []) or []
    st.session_state.dynamic_directions = result.get("dynamic_directions", []) or st.session_state.dynamic_directions or []
    st.session_state.selected_dynamic_direction = result.get("selected_dynamic_direction", {}) or st.session_state.selected_dynamic_direction or {}
    st.session_state.direction_choice_id = result.get("direction_choice_id", "") or st.session_state.direction_choice_id or ""
    st.session_state.quality_snapshot = _build_quality_snapshot(result)

    if result.get("needs_direction_choice"):
        st.session_state.current_stage = "direction_select"
    elif result.get("needs_clarification"):
        st.session_state.current_stage = "clarifying"
    elif st.session_state.generated_prompt:
        st.session_state.current_stage = "complete"
    else:
        st.session_state.current_stage = result.get("stage", "idle") or "idle"


# ============================================================================
# Streaming Hook-up
# ============================================================================

def _looks_complex(user_text: str) -> bool:
    return any(signal.lower() in user_text.lower() for signal in COMPLEXITY_SIGNALS)


def _safe_clear_streaming_state(thread_id: Optional[str]) -> None:
    if not thread_id:
        return
    clear_fn = getattr(graph, "clear_streaming_state", None)
    if callable(clear_fn):
        try:
            clear_fn(thread_id)
        except Exception:
            pass


def _safe_clear_streaming_chit_only(thread_id: Optional[str]) -> None:
    if not thread_id:
        return
    clear_fn = getattr(graph, "clear_streaming_chit_chat", None)
    if callable(clear_fn):
        try:
            clear_fn(thread_id)
            return
        except Exception:
            pass


def _safe_get_streaming_chit_chat(thread_id: str) -> str:
    get_fn = getattr(graph, "get_streaming_chit_chat", None)
    if not callable(get_fn):
        return ""
    try:
        return get_fn(thread_id) or ""
    except Exception:
        return ""


def _call_run_pipeline(user_input: str, thread_id: str) -> Dict[str, Any]:
    params = inspect.signature(graph.run_pipeline).parameters
    if "thread_id" in params:
        return graph.run_pipeline(user_input=user_input, thread_id=thread_id)
    result = graph.run_pipeline(user_input=user_input)
    if isinstance(result, dict):
        result.setdefault("thread_id", thread_id)
    return result


def _call_resume_after_clarify(selected_option: str, thread_id: str, user_input: str) -> Dict[str, Any]:
    params = inspect.signature(graph.resume_after_clarify).parameters
    if "thread_id" in params or "user_choice" in params:
        return graph.resume_after_clarify(
            thread_id=thread_id,
            user_choice=selected_option,
            user_input=user_input,
        )
    result = graph.resume_after_clarify(selected_option)
    if isinstance(result, dict):
        result.setdefault("thread_id", thread_id)
    return result


def _call_resume_after_direction(direction_choice_id: str, thread_id: str, user_input: str) -> Dict[str, Any]:
    resume_fn = getattr(graph, "resume_after_direction", None)
    if not callable(resume_fn):
        raise RuntimeError("Backend missing resume_after_direction - cannot proceed to v4.0 dynamic direction finalization.")
    return resume_fn(
        thread_id=thread_id,
        direction_choice_id=direction_choice_id,
        user_input=user_input,
    )

def _start_pipeline_worker(user_input: str, thread_id: str) -> tuple[threading.Event, Dict[str, Any]]:
    done_event = threading.Event()
    holder: Dict[str, Any] = {"result": None, "error": None}

    def worker() -> None:
        try:
            holder["result"] = _call_run_pipeline(user_input, thread_id)
        except Exception as exc:
            holder["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(
        target=worker,
        name=f"vibe-run-{thread_id[:8]}",
        daemon=True,
    )
    thread.start()
    return done_event, holder


def _start_resume_worker(selected_option: str, thread_id: str, user_input: str) -> tuple[threading.Event, Dict[str, Any]]:
    done_event = threading.Event()
    holder: Dict[str, Any] = {"result": None, "error": None}

    def worker() -> None:
        try:
            holder["result"] = _call_resume_after_clarify(selected_option, thread_id, user_input)
        except Exception as exc:
            holder["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(
        target=worker,
        name=f"vibe-resume-{thread_id[:8]}",
        daemon=True,
    )
    thread.start()
    return done_event, holder


def _start_direction_resume_worker(direction_choice_id: str, thread_id: str, user_input: str) -> tuple[threading.Event, Dict[str, Any]]:
    done_event = threading.Event()
    holder: Dict[str, Any] = {"result": None, "error": None}

    def worker() -> None:
        try:
            holder["result"] = _call_resume_after_direction(direction_choice_id, thread_id, user_input)
        except Exception as exc:
            holder["error"] = exc
        finally:
            done_event.set()

    thread = threading.Thread(
        target=worker,
        name=f"vibe-direction-{thread_id[:8]}",
        daemon=True,
    )
    thread.start()
    return done_event, holder


def _poll_chit_tokens(
    thread_id: str,
    done_event: threading.Event,
    holder: Dict[str, Any],
    status: Any,
) -> Iterator[str]:
    """Yield deltas from graph.get_streaming_chit_chat(thread_id).

    The backend hook returns the full accumulated string, so the frontend emits
    only the unseen suffix. The loop exits when the worker is done and the cache
    has stopped growing for a short grace period.
    """

    emitted_len = 0
    start_time = time.monotonic()
    last_growth = start_time
    last_status_tick = 0.0
    saw_token = False

    while True:
        now = time.monotonic()
        if now - start_time > STREAM_TIMEOUT_SECONDS:
            holder["error"] = TimeoutError("Stream poll timeout - backend chain may be stuck.")
            done_event.set()
            break

        full_text = _safe_get_streaming_chit_chat(thread_id)
        if full_text and len(full_text) < emitted_len:
            emitted_len = 0
            yield "\n\n"

        if len(full_text) > emitted_len:
            delta = full_text[emitted_len:]
            emitted_len = len(full_text)
            last_growth = now
            saw_token = True
            status.update(label="Token stream arriving from backend...", state="running", expanded=False)
            yield delta
            continue

        if holder.get("error") is not None:
            break

        if done_event.is_set():
            final_text = _safe_get_streaming_chit_chat(thread_id)
            if len(final_text) > emitted_len:
                delta = final_text[emitted_len:]
                emitted_len = len(final_text)
                yield delta
                continue
            if now - last_growth >= STREAM_GRACE_SECONDS:
                break

        if now - last_status_tick > 1.1:
            if saw_token:
                status.update(label="Claude is assembling the industrial prompt...", state="running", expanded=False)
            else:
                status.update(label="Waiting for first token from backend...", state="running", expanded=False)
            last_status_tick = now

        time.sleep(POLL_INTERVAL_SECONDS)


def _stream_backend_chit(
    thread_id: str,
    done_event: threading.Event,
    holder: Dict[str, Any],
    status: Any,
) -> str:
    st.markdown('<div class="liquid-glass-card mf-panel mf-panel-thin"><div class="cn-kicker" style="margin-bottom:0.4rem;">Developer Chit-chat</div>', unsafe_allow_html=True)
    streamed = st.write_stream(_poll_chit_tokens(thread_id, done_event, holder, status))
    st.markdown("</div>", unsafe_allow_html=True)
    return str(streamed or "")


# ============================================================================
# Parsing & Quality Helpers
# ============================================================================

def _split_two_part_output(generated_text: str) -> Dict[str, str]:
    text = (generated_text or "").strip()
    if not text:
        return {"chit_chat": "", "cursor_prompt": ""}

    cursor_markers = [
        "### 🤖 复制去投喂 Cursor",
        "### 复制去投喂 Cursor",
        "## 🤖 复制去投喂 Cursor",
        "## 复制去投喂 Cursor",
        "复制去投喂 Cursor",
    ]
    cursor_index = -1
    cursor_marker = ""
    for marker in cursor_markers:
        idx = text.find(marker)
        if idx != -1 and (cursor_index == -1 or idx < cursor_index):
            cursor_index = idx
            cursor_marker = marker

    if cursor_index == -1:
        xml_index = text.find("<system_role>")
        if xml_index != -1:
            return {
                "chit_chat": text[:xml_index].replace("---", "").strip(),
                "cursor_prompt": text[xml_index:].strip(),
            }
        context_index = text.find("[Context]")
        if context_index != -1:
            return {
                "chit_chat": text[:context_index].replace("---", "").strip(),
                "cursor_prompt": text[context_index:].strip(),
            }
        return {"chit_chat": "", "cursor_prompt": text}

    before_cursor = text[:cursor_index].strip()
    cursor_prompt = text[cursor_index + len(cursor_marker):].strip()

    for marker in [
        "### 💬 开发者碎碎念",
        "### 开发者碎碎念",
        "## 💬 开发者碎碎念",
        "## 开发者碎碎念",
        "开发者碎碎念",
    ]:
        if marker in before_cursor:
            before_cursor = before_cursor.split(marker, 1)[1].strip()
            break

    before_cursor = before_cursor.replace("---", "").strip()
    if cursor_prompt.startswith("---"):
        cursor_prompt = cursor_prompt[3:].strip()
    return {"chit_chat": before_cursor, "cursor_prompt": cursor_prompt}


def _extract_quality_report(result: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("quality_report", "quality_result", "quality_validation", "quality_check"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _contains_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _local_quality_scan(prompt: str) -> Dict[str, Any]:
    sections = _split_two_part_output(prompt)
    industrial = sections["cursor_prompt"] or prompt

    perf = bool(
        re.search(r"\b(TTI|LCP|CLS|FID|INP|P95|P99|fps|QPS|ms|s)\b", industrial, re.I)
        and re.search(r"(<|<=|>=|>|≤|≥|==)\d+\s*(ms|s|fps|qps|%)", industrial, re.I)
    )
    edge = _contains_any(
        industrial,
        ["empty", "network", "timeout", "retry", "concurrent", "conflict", "slow query", "exception", "degradation", "idempotent"],
    )
    observability = _contains_any(
        industrial,
        ["trace_id", "trace id", "structured log", "log", "instrument", "metrics", "prometheus", "alert", "link tracing"],
    )
    tests = _contains_any(
        industrial,
        ["test assertion", "unit test", "integration test", "e2e", "end to end", "assertion", "validation", "pytest", "playwright"],
    )

    score = 50 + sum([perf, edge, observability, tests]) * 12
    if (
        ("[Context]" in industrial and "[Constraints]" in industrial)
        or ("<system_role>" in industrial and "<dynamic_engineering_contract>" in industrial)
    ):
        score += 2
    return {
        "score": min(score, 100),
        "perf_budget": perf,
        "edge_cases": edge,
        "observability": observability,
        "test_assertions": tests,
        "passed": all([perf, edge, observability, tests]),
        "source": "local_scan",
    }


def _build_quality_snapshot(result: Dict[str, Any]) -> Dict[str, Any]:
    prompt = result.get("generated_prompt", "") or ""
    report = _extract_quality_report(result)
    if report:
        snapshot = {
            "score": int(report.get("score", 0) or 0),
            "perf_budget": bool(report.get("perf_budget", False)),
            "edge_cases": bool(report.get("edge_cases", False)),
            "observability": bool(report.get("observability", False)),
            "test_assertions": bool(report.get("test_assertions", False)),
            "passed": bool(report.get("passed", False)),
            "source": "backend_quality_gate",
        }
    else:
        snapshot = _local_quality_scan(prompt)

    snapshot["retry_count"] = int(result.get("quality_retry_count", 0) or 0)
    snapshot["complexity"] = result.get("complexity", "simple")
    return snapshot


def _call_get_checkpoint_state() -> Optional[Dict[str, Any]]:
    params = inspect.signature(graph.get_checkpoint_state).parameters
    if params:
        if not st.session_state.thread_id:
            return None
        return graph.get_checkpoint_state(st.session_state.thread_id)
    return graph.get_checkpoint_state()


# ============================================================================
# Pipeline Orchestration
# ============================================================================

def _run_pipeline_e2e(user_input: str) -> Dict[str, Any]:
    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    _safe_clear_streaming_state(thread_id)

    done_event, holder = _start_pipeline_worker(user_input, thread_id)

    with st.status("v4.0 stage 1: DeepSeek generating dynamic directions...", expanded=False) as status:
        status.update(label="DeepSeek v4-pro distilling 2-3 attack vectors...", state="running", expanded=False)

        start_time = time.monotonic()
        while not done_event.is_set():
            if time.monotonic() - start_time > STREAM_TIMEOUT_SECONDS:
                holder["error"] = TimeoutError("Dynamic direction generation timeout - DeepSeek chain may be stuck.")
                done_event.set()
                break
            status.update(label="DeepSeek folding your request into selectable tech routes...", state="running", expanded=False)
            time.sleep(0.12)
        st.session_state.last_streamed_chit = ""

        if holder.get("error") is not None:
            raise holder["error"]

        result = holder.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Backend chain finished but returned no valid result.")

        retry_count = int(result.get("quality_retry_count", 0) or 0)
        if retry_count > 0:
            status.update(
                label=f"Quality gate triggered self-correction, retry #{retry_count} complete.",
                state="running",
                expanded=False,
            )

        if result.get("needs_direction_choice"):
            status.update(label="Dynamic directions ready - pick your attack vector.", state="complete", expanded=False)
        elif result.get("needs_clarification"):
            status.update(label="Need a quick vibe clarification.", state="complete", expanded=False)
        else:
            status.update(label="Prompt delivered. Quality radar settling.", state="complete", expanded=False)

    if not result.get("needs_direction_choice"):
        _safe_clear_streaming_state(thread_id)
    return result


def _resume_pipeline_e2e(selected_option: str) -> Dict[str, Any]:
    thread_id = st.session_state.thread_id or str(uuid.uuid4())
    user_input = st.session_state.user_input
    st.session_state.thread_id = thread_id
    _safe_clear_streaming_chit_only(thread_id)

    done_event, holder = _start_resume_worker(selected_option, thread_id, user_input)

    with st.status("Resuming from checkpoint, continuing v4.0 finalization chain...", expanded=False) as status:
        status.update(label="Claude streaming response based on your selection...", state="running", expanded=False)
        streamed = _stream_backend_chit(thread_id, done_event, holder, status)
        st.session_state.last_streamed_chit = streamed

        if holder.get("error") is not None:
            raise holder["error"]

        result = holder.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Resume chain finished but returned no valid result.")

        retry_count = int(result.get("quality_retry_count", 0) or 0)
        if retry_count > 0:
            status.update(
                label=f"Quality gate triggered self-correction, retry #{retry_count} complete.",
                state="running",
                expanded=False,
            )
        status.update(label="Prompt delivered. Quality radar settling.", state="complete", expanded=False)

    _safe_clear_streaming_state(thread_id)
    return result


def _resume_direction_e2e(direction_choice_id: str) -> Dict[str, Any]:
    thread_id = st.session_state.thread_id or str(uuid.uuid4())
    user_input = st.session_state.user_input
    st.session_state.thread_id = thread_id
    _safe_clear_streaming_chit_only(thread_id)

    done_event, holder = _start_direction_resume_worker(direction_choice_id, thread_id, user_input)

    with st.status("v4.0 stage 2: Claude assembling XML-scoped prompt...", expanded=False) as status:
        status.update(label="Claude absorbing your attack vector, preparing token stream...", state="running", expanded=False)
        streamed = _stream_backend_chit(thread_id, done_event, holder, status)
        st.session_state.last_streamed_chit = streamed

        if holder.get("error") is not None:
            raise holder["error"]

        result = holder.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Dynamic direction finalization chain finished but returned no valid result.")

        retry_count = int(result.get("quality_retry_count", 0) or 0)
        if retry_count > 0:
            status.update(
                label=f"Quality gate triggered self-correction, retry #{retry_count} complete.",
                state="running",
                expanded=False,
            )
        status.update(label="XML prompt delivered. Quality radar settling.", state="complete", expanded=False)

    _safe_clear_streaming_state(thread_id)
    return result


# ============================================================================
# Persistence
# ============================================================================

def _save_positive_sample(user_input: str, generated_prompt: str) -> None:
    with _POSITIVE_SAMPLE_LOCK:
        samples: list = []
        if os.path.exists(POSITIVE_SAMPLES_FILE):
            try:
                with open(POSITIVE_SAMPLES_FILE, "r", encoding="utf-8") as f:
                    samples = json.load(f)
            except (json.JSONDecodeError, IOError):
                samples = []

        samples.append(
            {
                "user_input": user_input,
                "generated_prompt": generated_prompt,
                "quality_snapshot": st.session_state.quality_snapshot,
                "timestamp": datetime.now().isoformat(),
            }
        )

        with open(POSITIVE_SAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)


def _build_vibe_descriptions(options: List[str]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    for option in options:
        if option in VIBE_TECH_MAPPING:
            descriptions[option] = f'I want "{option}": {VIBE_TECH_MAPPING[option]["intent"]}'
        else:
            descriptions[option] = f'I want "{option}" - whatever that feels like.'
    return descriptions


def _format_direction_option(direction: Dict[str, Any]) -> str:
    title = str(direction.get("title", "Unnamed direction")).strip()
    focus = str(direction.get("focus", "")).strip()
    if focus:
        return f"{title} ~ {focus[:52]}{'...' if len(focus) > 52 else ''}"
    return title

# ============================================================================
# Rendering - CodeNest Liquid Glass
# ============================================================================

def _render_sidebar() -> None:
    pass  # deliberately empty - no sidebar in CodeNest


def _render_header() -> None:
    st.markdown(
        f"""
<section class="codenest-hero">
  <div class="liquid-glass-card mf-panel cn-floating-card">
    <div class="status-orbit"></div>
    <strong>CodeNest Core</strong>
    <span>DeepSeek directions locked. Claude XML contract ready for streamed assembly.</span>
  </div>
  <div class="codenest-kicker">CodeNest Dynamic Prompt Runtime</div>
  <h1 class="codenest-title">VIBE AGENT CORE<span class="green-dot">.</span></h1>
  <div id="mf-typewriter" class="mf-typewriter"></div>
</section>
""",
        unsafe_allow_html=True,
    )


def _render_error() -> None:
    if not st.session_state.error:
        return
    st.error(st.session_state.error)
    if st.button("Dismiss"):
        st.session_state.error = None
        if st.session_state.current_stage == "error":
            st.session_state.current_stage = "idle"
        st.rerun()


def _render_input_area() -> None:
    disabled = st.session_state.current_stage in {"generating", "clarifying", "direction_select"}
    st.markdown(
        '<div class="liquid-glass-card mf-panel"><div class="cn-kicker" style="margin-bottom:0.4rem;">Request Intake</div>',
        unsafe_allow_html=True,
    )
    user_input = st.text_area(
        "Describe the system atmosphere",
        value=st.session_state.user_input,
        placeholder="Build me a monitoring dashboard with the oppressive rationality of MOSS from The Wandering Earth 2. Needs resilience, perf budgets, test contracts.",
        height=128,
        disabled=disabled,
        key="vibe_input_box",
    )

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        generate_clicked = st.button(
            "Launch v4.0",
            type="primary",
            use_container_width=True,
            disabled=disabled,
        )
    with col_reset:
        reset_clicked = st.button("Reset", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        _safe_clear_streaming_state(st.session_state.thread_id)
        _reset_all()
        st.toast("Cleared. Try a different vibe.")
        st.rerun()

    if not generate_clicked:
        return

    if not user_input.strip():
        st.toast("Say something - even vague works.")
        return

    st.session_state.user_input = user_input.strip()
    st.session_state.liked = False
    st.session_state.error = None
    st.session_state.current_stage = "generating"

    try:
        result = _run_pipeline_e2e(st.session_state.user_input)
    except Exception as exc:
        _safe_clear_streaming_state(st.session_state.thread_id)
        st.session_state.current_stage = "error"
        st.session_state.error = f"v4.0 dynamic reasoning furnace tipped: {exc}"
        st.rerun()

    _apply_graph_response(result)
    st.rerun()


def _render_clarification_form() -> None:
    if st.session_state.current_stage != "clarifying":
        return

    options = list(st.session_state.clarification_options or [])
    if not options:
        options = list(VIBE_TECH_MAPPING.keys())
        st.session_state.clarification_options = options

    descriptions = _build_vibe_descriptions(options)

    st.divider()
    st.markdown(
        """
<div class="liquid-glass-card mf-panel mf-panel-thin">
<div style="font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--mf-muted);margin-bottom:0.3rem;">Clarification Needed</div>
<p style="font-size:0.92rem;color:var(--mf-muted);margin:0;">This vibe is abstract - the industrial chain hit a fork. Which feels closest?</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.clarification_question:
        st.caption(f"System query: {st.session_state.clarification_question}")

    with st.form(key="clarify_form", clear_on_submit=False):
        selected_option = st.radio(
            "Pick a direction",
            options=options,
            format_func=lambda option: descriptions[option],
            key="clarify_radio_locked",
        )
        submitted = st.form_submit_button(
            "Confirm",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    st.session_state.current_stage = "generating"
    try:
        result = _resume_pipeline_e2e(selected_option)
    except Exception as exc:
        _safe_clear_streaming_state(st.session_state.thread_id)
        st.session_state.current_stage = "clarifying"
        st.session_state.error = f"Checkpoint resume tipped: {exc}"
        st.rerun()

    _apply_graph_response(result)
    st.rerun()


def _render_dynamic_direction_form() -> None:
    if st.session_state.current_stage != "direction_select":
        return

    directions = list(st.session_state.dynamic_directions or [])
    if not directions:
        st.warning("DeepSeek left no direction snapshots. Returning to input stage.")
        st.session_state.current_stage = "idle"
        return

    st.divider()
    st.markdown(
        """
<div class="liquid-glass-card mf-panel mf-panel-thin">
<div style="font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--mf-muted);margin-bottom:0.3rem;">Dynamic Attack Vectors</div>
<p style="font-size:0.92rem;color:var(--mf-muted);margin:0;">DeepSeek decomposed your request into these strategic directions. Pick the one to finalize.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    for direction in directions:
        design_tokens = direction.get("design_tokens", []) or []
        budgets = direction.get("performance_budget", []) or []
        anti_patterns = direction.get("anti_patterns", []) or []
        details = []
        if design_tokens:
            details.append(f"Tokens: {'; '.join(map(str, design_tokens[:2]))}")
        if budgets:
            details.append(f"Budget: {'; '.join(map(str, budgets[:2]))}")
        if anti_patterns:
            details.append(f"Anti: {'; '.join(map(str, anti_patterns[:2]))}")
        details_html = "".join(f"<li>{escape(item)}</li>" for item in details)
        st.markdown(
            f"""
<div class="liquid-glass-card mf-direction-card">
<strong>{escape(str(direction.get("title", "Unnamed")))}</strong>
{escape(str(direction.get("focus", "")))}
<ul>{details_html}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="mf-direction-actions">', unsafe_allow_html=True)
    button_columns = st.columns(max(1, min(3, len(directions))))
    selected_id = ""
    selected_direction: Dict[str, Any] = {}

    for idx, direction in enumerate(directions):
        direction_id = str(direction.get("id") or f"direction_{idx + 1}")
        label = _format_direction_option(direction)
        with button_columns[idx % len(button_columns)]:
            if st.button(
                label,
                key=f"direction_pick_{direction_id}",
                use_container_width=True,
                type="primary" if idx == 0 else "secondary",
            ):
                selected_id = direction_id
                selected_direction = direction

    st.markdown("</div>", unsafe_allow_html=True)

    if not selected_id:
        return

    st.session_state.direction_choice_id = selected_id
    st.session_state.selected_dynamic_direction = selected_direction
    st.session_state.current_stage = "generating"
    try:
        result = _resume_direction_e2e(selected_id)
    except Exception as exc:
        _safe_clear_streaming_chit_only(st.session_state.thread_id)
        st.session_state.current_stage = "direction_select"
        st.session_state.error = f"Dynamic direction chain tipped: {exc}"
        st.rerun()

    _apply_graph_response(result)
    st.rerun()


def _render_quality_panel(snapshot: Dict[str, Any]) -> None:
    if not snapshot:
        return

    score = int(snapshot.get("score", 0) or 0)
    retry_count = int(snapshot.get("retry_count", 0) or 0)

    st.markdown('<div style="margin-bottom:0.6rem;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--mf-muted);margin-bottom:0.4rem;">Quality Radar</div>', unsafe_allow_html=True)

    col_score, col_retry = st.columns(2)
    col_score.metric("Score", f"{score}/100")
    col_retry.metric("Retries", retry_count)

    badges = [
        ("Perf Budget", snapshot.get("perf_budget")),
        ("Edge Defense", snapshot.get("edge_cases")),
        ("Observability", snapshot.get("observability")),
        ("Test Assertions", snapshot.get("test_assertions")),
    ]
    badge_html = "".join(
        f'<span class="mf-chip {"ok" if ok else "warn"}">[{name} {"OK" if ok else "WAIT"}]</span>'
        for name, ok in badges
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_clipboard_copy_button(text: str) -> None:
    payload = json.dumps(text)
    components.html(
        f"""
<button class="cn-copy-pill" id="mf-copy-prompt" type="button" aria-label="Copy Cursor prompt">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="11" height="11" rx="2"></rect>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
  </svg>
  <span>Copy</span>
</button>
<script>
const button = document.getElementById('mf-copy-prompt');
const text = {payload};
button.addEventListener('click', async () => {{
  const label = button.querySelector('span');
  try {{
    await navigator.clipboard.writeText(text);
    label.textContent = 'Copied';
  }} catch (error) {{
    const area = document.createElement('textarea');
    area.value = text;
    area.style.position = 'fixed';
    area.style.left = '-9999px';
    document.body.appendChild(area);
    area.focus();
    area.select();
    document.execCommand('copy');
    area.remove();
    label.textContent = 'Copied';
  }}
  window.setTimeout(() => label.textContent = 'Copy', 1400);
}});
</script>
<style>
html, body {{
  margin: 0;
  background: transparent;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.cn-copy-pill {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  width: 100%;
  min-height: 36px;
  border-radius: 999px;
  padding: 0.3em 1.25em;
  font-size: 15px;
  font-weight: 600;
  background: rgba(3, 8, 7, 0.92);
  color: #d8ffed;
  border: 1px solid rgba(94, 210, 156, 0.42);
  box-shadow: inset 0 1px 1px rgba(255,255,255,0.10), 0 0 22px rgba(94,210,156,0.12);
  cursor: pointer;
  transition: all 200ms ease;
}}
.cn-copy-pill:hover {{
  background: #5ed29c;
  color: #06100c;
  border-color: rgba(94, 210, 156, 0.78);
  box-shadow: 0 0 26px rgba(94,210,156,0.24);
}}
.cn-copy-pill svg {{
  width: 14px;
  height: 14px;
  flex: 0 0 auto;
}}
</style>
""",
        height=42,
    )


def _render_result() -> None:
    prompt = st.session_state.generated_prompt
    if not prompt:
        return

    sections = _split_two_part_output(prompt)
    chit_chat = sections["chit_chat"] or st.session_state.last_streamed_chit
    cursor_prompt = sections["cursor_prompt"] or prompt
    response = st.session_state.saved_response or {}
    matched = st.session_state.matched_vibes or response.get("matched_vibes", [])
    inferred_style = response.get("inferred_style", "")
    selected_direction = st.session_state.selected_dynamic_direction or response.get("selected_dynamic_direction", {})

    st.divider()
    st.markdown(
        '<p style="font-size:0.82rem;color:var(--mf-muted);font-family:var(--mf-font-body);">Pipeline: background thread > token poll > quality radar > prompt lock.</p>',
        unsafe_allow_html=True,
    )

    if matched:
        tags_html = " ".join(f'<span class="mf-chip ok">{escape(str(v))}</span>' for v in matched)
        st.markdown(f'<p style="font-size:0.76rem;color:var(--mf-muted);margin-bottom:0.3rem;">Detected vibes:</p>{tags_html}', unsafe_allow_html=True)

    if inferred_style:
        st.markdown(
            f'<div class="liquid-glass-card mf-panel mf-panel-thin"><strong style="font-size:0.82rem;">Inferred style:</strong> {escape(str(inferred_style))}</div>',
            unsafe_allow_html=True,
        )

    if selected_direction:
        st.markdown(
            f'<div class="liquid-glass-card mf-panel mf-panel-thin"><strong style="font-size:0.82rem;">Attack vector:</strong> {escape(str(selected_direction.get("title", "")))}<br><span>{escape(str(selected_direction.get("focus", "")))}</span></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.25, 1])
    with left:
        if chit_chat:
            chit_html = "<br>".join(escape(line) for line in str(chit_chat).splitlines() if line.strip())
            st.markdown(
                f'<div class="liquid-glass-card mf-panel"><div class="cn-kicker" style="margin-bottom:0.4rem;">Developer Chit-chat</div><p style="font-size:0.88rem;line-height:1.72;">{chit_html}</p></div>',
                unsafe_allow_html=True,
            )
    with right:
        _render_quality_panel(st.session_state.quality_snapshot)

    st.markdown('<div class="liquid-glass-card mf-panel"><div class="cn-kicker" style="margin-bottom:0.4rem;">Cursor Prompt</div>', unsafe_allow_html=True)
    st.subheader("Cursor XML Prompt")
    st.caption("Copy this industrial prompt for Claude Code / Cursor. Locked in session_state - likes and reruns will not clear it.")
    st.code(cursor_prompt, language="markdown", line_numbers=False)

    col_copy, col_like, col_refresh, _ = st.columns([1.1, 1.2, 1.4, 3.3])

    with col_copy:
        _render_clipboard_copy_button(cursor_prompt)

    with col_like:
        if st.session_state.liked:
            st.button("Saved to sample library", disabled=True, use_container_width=True)
        elif st.button("This one hits", type="primary", use_container_width=True):
            _save_positive_sample(st.session_state.user_input, prompt)
            st.session_state.liked = True
            st.toast("Positive sample captured.")
            st.balloons()
            st.rerun()

    with col_refresh:
        if st.button("Check backend state", use_container_width=True):
            checkpoint = _call_get_checkpoint_state()
            if checkpoint:
                _apply_graph_response(checkpoint)
                st.toast("Backend state intact.")
            else:
                st.toast("No checkpoint found - page prompt is stable.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(BACKDROP_HTML, unsafe_allow_html=True)
    components.html(INTERACTION_SCRIPT_HTML, height=0, width=0)
    _render_sidebar()
    _render_header()
    _render_error()
    _render_input_area()
    _render_dynamic_direction_form()
    _render_clarification_form()
    _render_result()


if __name__ == "__main__":
    main()
