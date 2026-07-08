"""
Vibe-to-Prompt Agent - Streamlit Frontend v3.5 E2E Hook-up
==========================================================

This frontend consumes the real backend streaming hooks:
- get_streaming_chit_chat(thread_id)
- clear_streaming_state(thread_id)

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

import graph
from vibe_config import VIBE_TECH_MAPPING


st.set_page_config(
    page_title="Vibe-to-Prompt Agent v4.0",
    page_icon="✨",
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
    "今天又是用 Vibe 糊弄过去的一天，咖啡喝了吗？☕",
    "需求可以模糊，但 Prompt 必须硬核。💪",
    "先别焦虑，把感觉说出来，剩下的交给 3.5 流式工业链路。🧠",
    "这不是偷懒，这是把抽象需求产品化。📦",
    "把一句大白话变成一份可验收规格书，也算今日份降本增效。📈",
    "不要怕描述不专业，专业这件事我来补。🛠️",
    "先输入 Vibe，再假装一切都在掌控中。🚀",
    "今天的工位哲学：能让 Agent 写清楚的，就别自己憋。🌿",
]

COMPLEXITY_SIGNALS: List[str] = [
    "像", "就像", "仿佛", "电影", "动漫", "游戏", "小说", "末日", "压抑", "窒息",
    "梦境", "迷幻", "废土", "哥特", "黑客帝国", "代码雨", "全息", "霓虹",
    "银翼杀手", "攻壳", "MOSS", "流浪地球", "质感", "通感",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Instrument+Serif:ital@0;1&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
    --cn-bg: #070b0a;
    --cn-green: #5ed29c;
    --cn-cyan: #61f3e8;
    --cn-ink: #e8fff5;
    --cn-muted: rgba(223, 255, 241, 0.68);
    --cn-line: rgba(255, 255, 255, 0.10);
    --cn-glass: rgba(255, 255, 255, 0.01);
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--cn-bg) !important;
    color: var(--cn-ink) !important;
    font-family: "Inter", "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
}

.stApp::before {
    background:
        linear-gradient(90deg, rgba(7, 11, 10, 0.98) 0%, rgba(7, 11, 10, 0.64) 34%, rgba(7, 11, 10, 0.18) 100%),
        linear-gradient(0deg, rgba(7, 11, 10, 1) 0%, rgba(7, 11, 10, 0.72) 18%, rgba(7, 11, 10, 0.10) 58%),
        radial-gradient(ellipse at 50% -8%, rgba(97, 243, 232, 0.22), rgba(8, 77, 59, 0.18) 28%, transparent 62%);
}

.stApp::after {
    background:
        linear-gradient(90deg, transparent 24.92%, var(--cn-line) 25%, transparent 25.08%),
        linear-gradient(90deg, transparent 49.92%, var(--cn-line) 50%, transparent 50.08%),
        linear-gradient(90deg, transparent 74.92%, var(--cn-line) 75%, transparent 75.08%);
    opacity: 0.86;
}

.codenest-video {
    position: fixed;
    inset: 0;
    z-index: -2;
    width: 100vw;
    height: 100vh;
    object-fit: cover;
    opacity: 0.60;
    filter: saturate(0.9) contrast(1.04) brightness(0.56);
    pointer-events: none;
}

.codenest-aurora {
    position: fixed;
    top: -92px;
    left: 50%;
    transform: translateX(-50%);
    width: min(760px, 88vw);
    height: 220px;
    z-index: -1;
    pointer-events: none;
    filter: blur(25px);
    opacity: 0.82;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: rgba(5, 9, 8, 0.72) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
}

section.main > div,
.block-container {
    width: min(980px, calc(100vw - 2rem)) !important;
    max-width: min(980px, calc(100vw - 2rem)) !important;
    padding-top: 3.2rem !important;
    padding-bottom: 4rem !important;
    position: relative;
    z-index: 1;
}

.codenest-hero {
    margin: 0.4rem 0 1.55rem;
}

.stage-kicker,
.direction-panel strong,
.quality-title,
.tiny-note strong {
    font-family: "Plus Jakarta Sans", "Inter", sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase;
    font-weight: 800 !important;
    color: var(--cn-green) !important;
}

.codenest-title {
    margin: 0;
    max-width: 940px;
    font-family: "Inter", "Plus Jakarta Sans", sans-serif;
    font-size: clamp(40px, 8vw, 72px);
    line-height: 0.91;
    letter-spacing: -0.065em;
    font-weight: 900;
    color: #f4fff9;
    text-transform: uppercase;
    text-shadow: 0 0 34px rgba(94, 210, 156, 0.12);
}

.codenest-title .green-dot {
    color: var(--cn-green);
    text-shadow: 0 0 30px rgba(94, 210, 156, 0.55);
}

.caption {
    max-width: 760px;
    color: var(--cn-muted);
    font-size: 1rem;
    line-height: 1.76;
    margin-top: 0.9rem;
    margin-bottom: 1.25rem;
}

.liquid-glass-card,
.soft-panel,
.stream-shell,
.chit-chat-panel,
.quality-panel,
.clarify-panel,
.direction-panel,
.direction-card {
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.01) !important;
    background-blend-mode: luminosity;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    border: 0 !important;
    border-radius: 18px;
    box-shadow:
        inset 0 1px 1px rgba(255, 255, 255, 0.1),
        0 24px 80px rgba(0, 0, 0, 0.32);
}

.liquid-glass-card::before,
.soft-panel::before,
.stream-shell::before,
.chit-chat-panel::before,
.quality-panel::before,
.clarify-panel::before,
.direction-panel::before,
.direction-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1.4px;
    background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.06) 42%, rgba(94,210,156,0.36));
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.liquid-glass-card::after,
.stream-shell::after,
.quality-panel::after,
.direction-panel::after {
    content: "";
    position: absolute;
    inset: -1px;
    background: linear-gradient(110deg, transparent 0%, rgba(97, 243, 232, 0.10) 35%, transparent 62%);
    transform: translateX(-40%);
    animation: codenestSheen 7s ease-in-out infinite;
    pointer-events: none;
}

@keyframes codenestSheen {
    0%, 100% { transform: translateX(-56%); opacity: 0; }
    42% { opacity: 1; }
    62% { transform: translateX(58%); opacity: 0; }
}

@keyframes radarFadeIn {
    from { opacity: 0; transform: translateY(8px); filter: blur(2px); }
    to { opacity: 1; transform: translateY(0); filter: blur(0); }
}

.input-shell,
.result-shell {
    padding: 1.1rem 1.15rem 1.25rem;
    margin: 1rem 0 1.15rem;
}

.soft-panel,
.stream-shell,
.chit-chat-panel,
.quality-panel,
.clarify-panel,
.direction-panel {
    padding: 1rem 1.05rem;
    color: #dfffee;
    margin: 1rem 0;
}

.direction-card {
    padding: 0.95rem 1rem;
    margin: 0.72rem 0;
    color: rgba(232, 255, 245, 0.82);
}

.direction-card strong {
    display: inline-block;
    margin-bottom: 0.32rem;
    font-family: "Plus Jakarta Sans", "Inter", sans-serif;
    color: #f5fff9 !important;
    font-size: 0.96rem !important;
    letter-spacing: 0 !important;
    text-transform: none;
}

.direction-card ul {
    margin-top: 0.54rem;
    margin-bottom: 0.08rem;
    color: rgba(210, 255, 234, 0.70);
}

.chit-chat-panel h3,
.stream-shell h3 {
    margin: 0 0 0.65rem !important;
    color: #f4fff9 !important;
    font-family: "Plus Jakarta Sans", "Inter", sans-serif !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.chit-chat-panel p,
.moment-line,
.tiny-note {
    color: rgba(224, 255, 240, 0.72);
    font-size: 0.95rem;
    line-height: 1.72;
}

.quality-panel {
    animation: radarFadeIn 420ms ease-out both;
}

.quality-badge,
.vibe-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.26rem 0.68rem;
    margin: 0 0.36rem 0.36rem 0;
    font-family: "Plus Jakarta Sans", "Inter", sans-serif;
    font-size: 0.76rem;
    font-weight: 800;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.035);
    color: rgba(232, 255, 245, 0.82);
}

.quality-badge.ok,
.vibe-chip {
    color: #c8ffe2;
    border-color: rgba(94, 210, 156, 0.35);
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 0 22px rgba(94, 210, 156, 0.08);
}

.quality-badge.warn {
    color: #ffe7aa;
    border-color: rgba(245, 158, 11, 0.35);
}

.stTextArea label,
.stRadio label,
.stFormSubmitButton label {
    color: rgba(232, 255, 245, 0.82) !important;
    font-family: "Plus Jakarta Sans", "Inter", sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.stTextArea textarea {
    background: rgba(255, 255, 255, 0.018) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f4fff9 !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    line-height: 1.62 !important;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 18px 48px rgba(0,0,0,0.20);
}

.stTextArea div:has(> textarea),
.stTextArea div:has(textarea) {
    background: rgba(255, 255, 255, 0.018) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

.stTextArea textarea:focus {
    border-color: rgba(94, 210, 156, 0.72) !important;
    box-shadow: 0 0 0 1px rgba(94, 210, 156, 0.22), 0 0 28px rgba(94, 210, 156, 0.12) !important;
}

.stTextArea textarea::placeholder {
    color: rgba(232, 255, 245, 0.35) !important;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 999px !important;
    font-family: "Plus Jakarta Sans", "Inter", sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease !important;
    background: rgba(255, 255, 255, 0.025) !important;
    color: #effff7 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.09);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(94, 210, 156, 0.55) !important;
    box-shadow: 0 0 34px rgba(94, 210, 156, 0.15), inset 0 1px 1px rgba(255,255,255,0.12);
}

div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: linear-gradient(135deg, rgba(94, 210, 156, 0.95), rgba(97, 243, 232, 0.62)) !important;
    border-color: rgba(202, 255, 230, 0.56) !important;
    color: #06100c !important;
}

div[data-testid="stRadio"] > div {
    position: relative;
    background: rgba(255, 255, 255, 0.018) !important;
    border-radius: 16px;
    padding: 0.85rem 1rem;
    border: 1px solid rgba(255, 255, 255, 0.10);
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.08);
}

div[data-testid="stRadio"] p,
div[data-testid="stRadio"] span {
    color: rgba(232, 255, 245, 0.86) !important;
}

.stCodeBlock {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.08), 0 24px 70px rgba(0,0,0,0.30);
    overflow: hidden;
}

.stCodeBlock pre {
    background: rgba(3, 8, 7, 0.86) !important;
}

.stCodeBlock code {
    color: #d8ffed !important;
}

hr {
    border-color: rgba(255, 255, 255, 0.10) !important;
}

@media (max-width: 760px) {
    .stApp::after {
        display: none;
    }
    .block-container {
        padding-top: 2rem !important;
    }
    .input-shell,
    .result-shell {
        padding: 0.9rem;
    }
}
</style>
"""

BACKDROP_HTML = """
<video class="codenest-video" src="https://stream.mux.com/tLkHO1qZoaaQOUeVWo8hEBeGQfySP02EPS02BmnNFyXys.m3u8" autoplay muted loop playsinline preload="auto">
    <source src="https://stream.mux.com/tLkHO1qZoaaQOUeVWo8hEBeGQfySP02EPS02BmnNFyXys.m3u8" type="application/x-mpegURL">
</video>
<svg class="codenest-aurora" viewBox="0 0 760 220" aria-hidden="true">
    <defs>
        <radialGradient id="cnGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#61f3e8" stop-opacity="0.85"/>
            <stop offset="46%" stop-color="#0f6f51" stop-opacity="0.48"/>
            <stop offset="100%" stop-color="#070b0a" stop-opacity="0"/>
        </radialGradient>
    </defs>
    <ellipse cx="380" cy="92" rx="320" ry="78" fill="url(#cnGlow)"/>
</svg>
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
        raise RuntimeError("后端缺少 resume_after_direction，无法进入 v4.0 动态方向终审。")
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
            holder["error"] = TimeoutError("流式轮询超时，后台链路可能卡住。")
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
            status.update(label="碎碎念 Token 正在从后端流式抵达... 💬", state="running", expanded=False)
            yield delta
            continue

        if holder.get("error") is not None:
            break

        if done_event.is_set():
            # One final poll after completion to consume late cache writes.
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
                status.update(label="Claude 正在组装工业 Prompt，质量闸准备接棒... 🤖", state="running", expanded=False)
            else:
                status.update(label="正在等待后端首个碎碎念 Token... 💬", state="running", expanded=False)
            last_status_tick = now

        time.sleep(POLL_INTERVAL_SECONDS)


def _stream_backend_chit(
    thread_id: str,
    done_event: threading.Event,
    holder: Dict[str, Any],
    status: Any,
) -> str:
    st.markdown('<div class="stream-shell"><h3>💬 开发者碎碎念</h3>', unsafe_allow_html=True)
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
        and re.search(r"(<|<=|>=|>|≤|≥|\d+\s*(ms|s|fps|qps|%))", industrial, re.I)
    )
    edge = _contains_any(
        industrial,
        ["空数据", "网络", "超时", "重试", "并发", "冲突", "慢查询", "异常", "降级", "幂等"],
    )
    observability = _contains_any(
        industrial,
        ["trace_id", "trace id", "结构化日志", "日志", "埋点", "metrics", "Prometheus", "告警", "链路追踪"],
    )
    tests = _contains_any(
        industrial,
        ["测试断言", "单元测试", "集成测试", "E2E", "端到端", "断言", "验收", "pytest", "Playwright"],
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

    with st.status("v4.0 第一阶段启动：DeepSeek 正在生成动态方向...", expanded=False) as status:
        status.update(label="DeepSeek v4-pro 正在提炼 2-3 个主攻方向... 🧠", state="running", expanded=False)

        start_time = time.monotonic()
        while not done_event.is_set():
            if time.monotonic() - start_time > STREAM_TIMEOUT_SECONDS:
                holder["error"] = TimeoutError("动态方向生成超时，DeepSeek 链路可能卡住。")
                done_event.set()
                break
            status.update(label="DeepSeek 正在把需求压成可选择的技术路线...", state="running", expanded=False)
            time.sleep(0.12)
        st.session_state.last_streamed_chit = ""

        if holder.get("error") is not None:
            raise holder["error"]

        result = holder.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("后端链路结束但没有返回有效结果。")

        retry_count = int(result.get("quality_retry_count", 0) or 0)
        if retry_count > 0:
            status.update(
                label=f"质量闸触发 Self-Correction，已完成重试第 {retry_count} 次。",
                state="running",
                expanded=False,
            )

        if result.get("needs_direction_choice"):
            status.update(label="动态方向已提炼，等你选择主攻路线。", state="complete", expanded=False)
        elif result.get("needs_clarification"):
            status.update(label="需要你补一刀氛围感方向。", state="complete", expanded=False)
        else:
            status.update(label="🚀 Prompt 已交付，质量雷达正在淡入。", state="complete", expanded=False)

    if not result.get("needs_direction_choice"):
        _safe_clear_streaming_state(thread_id)
    return result


def _resume_pipeline_e2e(selected_option: str) -> Dict[str, Any]:
    thread_id = st.session_state.thread_id or str(uuid.uuid4())
    user_input = st.session_state.user_input
    st.session_state.thread_id = thread_id
    _safe_clear_streaming_chit_only(thread_id)

    done_event, holder = _start_resume_worker(selected_option, thread_id, user_input)

    with st.status("正在恢复 checkpoint，继续 v4.0 终审链路...", expanded=False) as status:
        status.update(label="Claude 正在基于你的选择流式回应... 💬", state="running", expanded=False)
        streamed = _stream_backend_chit(thread_id, done_event, holder, status)
        st.session_state.last_streamed_chit = streamed

        if holder.get("error") is not None:
            raise holder["error"]

        result = holder.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("恢复链路结束但没有返回有效结果。")

        retry_count = int(result.get("quality_retry_count", 0) or 0)
        if retry_count > 0:
            status.update(
                label=f"质量闸触发 Self-Correction，已完成重试第 {retry_count} 次。",
                state="running",
                expanded=False,
            )
        status.update(label="🚀 Prompt 已交付，质量雷达正在淡入。", state="complete", expanded=False)

    _safe_clear_streaming_state(thread_id)
    return result


def _resume_direction_e2e(direction_choice_id: str) -> Dict[str, Any]:
    thread_id = st.session_state.thread_id or str(uuid.uuid4())
    user_input = st.session_state.user_input
    st.session_state.thread_id = thread_id
    _safe_clear_streaming_chit_only(thread_id)

    done_event, holder = _start_direction_resume_worker(direction_choice_id, thread_id, user_input)

    with st.status("v4.0 第二阶段启动：Claude 正在按选中方向编排 XML Prompt...", expanded=False) as status:
        status.update(label="Claude 正在吸收你的主攻方向，准备流式碎碎念... 💬", state="running", expanded=False)
        streamed = _stream_backend_chit(thread_id, done_event, holder, status)
        st.session_state.last_streamed_chit = streamed

        if holder.get("error") is not None:
            raise holder["error"]

        result = holder.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("动态方向终审链路结束但没有返回有效结果。")

        retry_count = int(result.get("quality_retry_count", 0) or 0)
        if retry_count > 0:
            status.update(
                label=f"质量闸触发 Self-Correction，已完成重试第 {retry_count} 次。",
                state="running",
                expanded=False,
            )
        status.update(label="🚀 XML Prompt 已交付，质量雷达正在淡入。", state="complete", expanded=False)

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
            descriptions[option] = f"我想要「{option}」：{VIBE_TECH_MAPPING[option]['intent']}"
        else:
            descriptions[option] = f"我想要「{option}」这种感觉"
    return descriptions


def _format_direction_option(direction: Dict[str, Any]) -> str:
    title = str(direction.get("title", "未命名方向")).strip()
    focus = str(direction.get("focus", "")).strip()
    if focus:
        return f"{title}｜{focus[:52]}{'...' if len(focus) > 52 else ''}"
    return title


# ============================================================================
# Rendering
# ============================================================================

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 今日工位小纸条")
        st.info(st.session_state.office_quote)
        st.markdown("### v4.0 动态方向闭环")
        st.markdown(
            """
- 第一阶段：DeepSeek v4-pro 生成动态方向
- HITL：`st.session_state` 锁住方向快照
- 第二阶段：选择方向后恢复 checkpoint
- 收尾：Claude 编排 XML 严格作用域 Prompt
"""
        )


def _render_header() -> None:
    st.markdown(
        """
<section class="codenest-hero">
  <div class="stage-kicker">CodeNest Dynamic Prompt Runtime</div>
  <h1 class="codenest-title">VIBE AGENT CORE<span class="green-dot">.</span></h1>
  <p class="caption">你负责说“感觉”，DeepSeek 先动态提炼主攻方向；你拍板后，Claude 再把最终 XML Prompt 稳稳编出来。</p>
</section>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="soft-panel"><strong>今日摸鱼鼓励：</strong>{st.session_state.office_quote}</div>',
        unsafe_allow_html=True,
    )


def _render_error() -> None:
    if not st.session_state.error:
        return
    st.error(st.session_state.error)
    if st.button("我知道了，先把报错收起来"):
        st.session_state.error = None
        if st.session_state.current_stage == "error":
            st.session_state.current_stage = "idle"
        st.rerun()


def _render_input_area() -> None:
    disabled = st.session_state.current_stage in {"generating", "clarifying", "direction_select"}
    st.markdown(
        '<div class="liquid-glass-card input-shell"><div class="stage-kicker">Request Intake</div>',
        unsafe_allow_html=True,
    )
    user_input = st.text_area(
        "先把你脑子里的那个画面丢给我",
        value=st.session_state.user_input,
        placeholder=(
            "比如：给我搞一个像流浪地球2里 MOSS 那种压抑但绝对理性的监控大屏，"
            "要抗造、有性能预算、有测试断言..."
        ),
        height=136,
        disabled=disabled,
        key="vibe_input_box",
    )

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        generate_clicked = st.button(
            "启动 v4.0 动态思考炉",
            type="primary",
            use_container_width=True,
            disabled=disabled,
        )
    with col_reset:
        reset_clicked = st.button("重新来一把", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        _safe_clear_streaming_state(st.session_state.thread_id)
        _reset_all()
        st.toast("清空啦，换个 Vibe 继续整。🧹")
        st.rerun()

    if not generate_clicked:
        return

    if not user_input.strip():
        st.toast("先随便说两句也行，空白我是真的脑补不动。🥲")
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
        st.session_state.error = f"v4.0 动态思考炉运行时翻车了：{exc}"
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
<div class="clarify-panel">
<strong>哎呀，这个 Vibe 有点抽象，工业链路卡在分岔路口了。</strong><br>
你指的是哪种感觉？选一个最像的，我继续恢复 checkpoint 往下跑。
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.clarification_question:
        st.caption(f"系统原始问题：{st.session_state.clarification_question}")

    with st.form(key="clarify_form", clear_on_submit=False):
        selected_option = st.radio(
            "你更靠近哪种日常说法？",
            options=options,
            format_func=lambda option: descriptions[option],
            key="clarify_radio_locked",
        )
        submitted = st.form_submit_button(
            "确认我的氛围感",
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
        st.session_state.error = f"恢复 checkpoint 时翻车了：{exc}"
        st.rerun()

    _apply_graph_response(result)
    st.rerun()


def _render_dynamic_direction_form() -> None:
    if st.session_state.current_stage != "direction_select":
        return

    directions = list(st.session_state.dynamic_directions or [])
    if not directions:
        st.warning("DeepSeek 没有留下可选方向快照，我先把链路退回输入态。")
        st.session_state.current_stage = "idle"
        return

    st.divider()
    st.markdown(
        """
<div class="direction-panel">
<strong>DeepSeek 已经把需求拆成了几个动态主攻方向。</strong><br>
先选一个你最想押注的方向；我会把这个选择锁进 checkpoint，再交给 Claude 编排最终 XML 超级 Prompt。
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
            details.append(f"Design Tokens：{'; '.join(map(str, design_tokens[:2]))}")
        if budgets:
            details.append(f"性能预算：{'; '.join(map(str, budgets[:2]))}")
        if anti_patterns:
            details.append(f"反模式：{'; '.join(map(str, anti_patterns[:2]))}")
        details_html = "".join(f"<li>{escape(item)}</li>" for item in details)
        st.markdown(
            f"""
<div class="direction-card">
<strong>{escape(str(direction.get("title", "未命名方向")))}</strong><br>
{escape(str(direction.get("focus", "")))}
<ul>{details_html}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    option_ids = [str(d.get("id") or f"direction_{idx + 1}") for idx, d in enumerate(directions)]
    direction_by_id = {str(d.get("id") or f"direction_{idx + 1}"): d for idx, d in enumerate(directions)}

    with st.form(key="dynamic_direction_form", clear_on_submit=False):
        selected_id = st.radio(
            "这次最终 Prompt 主攻哪个方向？",
            options=option_ids,
            format_func=lambda option: _format_direction_option(direction_by_id.get(option, {})),
            key="dynamic_direction_radio_locked",
        )
        submitted = st.form_submit_button(
            "确认进阶生成",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    st.session_state.direction_choice_id = selected_id
    st.session_state.selected_dynamic_direction = direction_by_id.get(selected_id, {})
    st.session_state.current_stage = "generating"
    try:
        result = _resume_direction_e2e(selected_id)
    except Exception as exc:
        _safe_clear_streaming_chit_only(st.session_state.thread_id)
        st.session_state.current_stage = "direction_select"
        st.session_state.error = f"动态方向终审链路翻车了：{exc}"
        st.rerun()

    _apply_graph_response(result)
    st.rerun()


def _render_quality_panel(snapshot: Dict[str, Any]) -> None:
    if not snapshot:
        return

    score = int(snapshot.get("score", 0) or 0)
    retry_count = int(snapshot.get("retry_count", 0) or 0)

    st.markdown('<div class="quality-panel"><div class="quality-title">🛡️ 3.0 质量雷达</div>', unsafe_allow_html=True)
    col_score, col_retry, col_source = st.columns(3)
    col_score.metric("质量得分", f"{score}/100")
    col_retry.metric("重试次数", retry_count)
    col_source.metric("审查来源", "后端" if snapshot.get("source") == "backend_quality_gate" else "前端复核")

    badges = [
        ("性能预算", snapshot.get("perf_budget")),
        ("边界防御", snapshot.get("edge_cases")),
        ("可观测性", snapshot.get("observability")),
        ("测试断言", snapshot.get("test_assertions")),
    ]
    badge_html = "".join(
        f'<span class="quality-badge {"ok" if ok else "warn"}">[{name} {"✓" if ok else "!"}]</span>'
        for name, ok in badges
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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
        '<p class="moment-line">刚才链路大概经历了：后台线程跑图、前台轮询真实 Token、质量雷达淡入、最终 Prompt 锁定。</p>',
        unsafe_allow_html=True,
    )

    if matched:
        tags_html = " ".join(f'<span class="vibe-chip">{escape(str(v))}</span>' for v in matched)
        st.markdown(f'<p class="tiny-note">我抓到的味儿大概是：</p>{tags_html}', unsafe_allow_html=True)

    if inferred_style:
        st.markdown(
            f'<div class="soft-panel"><strong>我额外脑补了一点：</strong>{escape(str(inferred_style))}</div>',
            unsafe_allow_html=True,
        )

    if selected_direction:
        st.markdown(
            f'<div class="soft-panel"><strong>本次主攻方向：</strong>{escape(str(selected_direction.get("title", "")))}<br>{escape(str(selected_direction.get("focus", "")))}</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.25, 1])
    with left:
        if chit_chat:
            chit_html = "<br>".join(escape(line) for line in str(chit_chat).splitlines() if line.strip())
            st.markdown(
                f'<div class="chit-chat-panel"><h3>💬 开发者碎碎念</h3><p>{chit_html}</p></div>',
                unsafe_allow_html=True,
            )
    with right:
        _render_quality_panel(st.session_state.quality_snapshot)

    st.markdown('<div class="liquid-glass-card result-shell"><div class="stage-kicker">Claude XML Output</div>', unsafe_allow_html=True)
    st.subheader("🤖 复制去投喂 Cursor")
    st.caption("下面这段是建议复制给 Claude Code / Cursor 的工业 Prompt；内容已锁在 session_state 里，点赞、查看状态或 rerun 都不会刷掉。")
    st.code(cursor_prompt, language="markdown", line_numbers=False)

    col_copy, col_like, col_refresh, _ = st.columns([1.1, 1.2, 1.4, 3.3])

    with col_copy:
        if st.button("提示我复制", use_container_width=True):
            st.toast("代码块右上角可以一键复制，我在旁边给你举个灯。✨")

    with col_like:
        if st.session_state.liked:
            st.button("已收进样本库", disabled=True, use_container_width=True)
        elif st.button("这条很能打", type="primary", use_container_width=True):
            _save_positive_sample(st.session_state.user_input, prompt)
            st.session_state.liked = True
            st.toast("收到，这条好样本我已经偷偷记进小本本了。📒")
            st.balloons()
            st.rerun()

    with col_refresh:
        if st.button("看看后台状态", use_container_width=True):
            checkpoint = _call_get_checkpoint_state()
            if checkpoint:
                _apply_graph_response(checkpoint)
                st.toast("后台状态还在，没丢。🧾")
            else:
                st.toast("暂时没读到 checkpoint，但页面上的 Prompt 还稳稳挂着。")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(BACKDROP_HTML, unsafe_allow_html=True)
    _render_sidebar()
    _render_header()
    _render_error()
    _render_input_area()
    _render_dynamic_direction_form()
    _render_clarification_form()
    _render_result()


if __name__ == "__main__":
    main()
