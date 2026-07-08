"""
Vibe-to-Prompt Agent v4.0 — XML Contract Prompt Generation
==========================================================
Claude 3.5 Sonnet + DeepSeek dual-model architecture with async routing.

v4.0 keeps the v3.5 latency-optimized topology and upgrades the final
Cursor prompt into a strict XML-scoped engineering contract:
    START
      │
      ▼
  route_complexity ──simple──► concurrent_analyze_and_retrieve ──► clarify_check
      │                    (analyze_fast ∥ retrieve_prelim)         │
      │ complex                                                      ├── needs_clear → END
      ▼                                                              │
  concurrent_reason_and_retrieve ───────────────────────────────────► compose_with_claude
 (deep_reason ∥ retrieve_prelim)                                     │  ├─ chit-chat (stream=True)
                                                                      │  └─ industrial (batch)
                                                                      ▼
                                                                 quality_check
                                                                      │
                                                                   pass ──► END
                                                                   fail ──► compose_with_claude (retry≤1)

Key v3.5 improvements over v3.0:
  1. Concurrent routing: LLM analysis and qdrant retrieval run in parallel
     via ThreadPoolExecutor, eliminating ~5ms of serial wait.
  2. Streaming split: Chit-chat streamed token-by-token for 打字机效果;
     industrial prompt assembled in a separate batched call.
  3. Lightweight checkpoints: reasoning_summary cached module-level,
     reducing per-checkpoint serialization payload by ~70%.

HITL: clarify_check sets needs_clarification=True → routes to END.
      Streamlit inspects state, shows choice UI, then calls
      graph.update_state() + graph.stream(None, …) to resume.

Checkpointer: module-level singleton (MemorySaver) — fixes the
              cross-call state-loss bug from per-call instantiation.

Quality Gate (v3.0): quality_check validates every generated prompt against
  4 industrial pillars — Performance Budgets, Edge Case Handling,
  Observability, Test Assertions.  Fails trigger self-correction (max 1 retry).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from vibe_config import (
    DEFAULT_PROMPT_SECTIONS,
    DESIGN_TOKENS_AND_RULES,
    FEW_SHOT_EXAMPLES,
    VIBE_ALIASES,
    VIBE_TECH_MAPPING,
    FewShotExample,
    VibeProfile,
)

# ============================================================================
# Constants
# ============================================================================

COLLECTION_NAME = "few_shot_examples"
VECTOR_DIM = 256

# ---- Complexity routing signals ----
# If the user input contains any of these, we route to deep_reason.
_COMPLEXITY_SIGNALS: List[str] = [
    # Simile / metaphor markers
    "像", "就像", "仿佛", "好像", "好比", "类似", "跟…一样",
    # Media / cultural references
    "电影", "动漫", "动画", "游戏里", "小说", "漫画", "剧中",
    # Atmospheric / hard-to-map moods
    "末日", "压抑", "窒息", "荒诞", "梦境", "迷幻", "虚幻", "诡谲",
    "复古", "未来感", "蒸汽波", "废土", "哥特", "巴洛克",
    "极简", "梦核", "怪核", "禅意", "工业风",
    # Complex visual / experiential concepts
    "黑客帝国", "代码雨", "时光隧道", "数据流", "全息",
    "霓虹", "激光", "像素", "故障艺术", "glitch",
    "盗梦空间", "银翼杀手", "攻壳", "EVA", "新世纪福音战士",
    "纪念碑谷", "对马岛", "空洞骑士", "极乐迪斯科",
    # Deep-emotion / synaesthesia words
    "味道", "质感", "温度", "触感", "通感",
]

# ============================================================================
# Checkpointer singleton — FIX: no longer per-call MemorySaver
# ============================================================================

_CHECKPOINTER: MemorySaver = MemorySaver()
_CHECKPOINTER_LOCK = threading.RLock()

# ============================================================================
# v3.5: Module-level caches for lightweight checkpoints & streaming
# ============================================================================

# Reasoning cache: keeps the heavy reasoning_summary out of LangGraph checkpoints.
# Keyed by thread_id — compose_with_claude reads from here instead of state.
_REASONING_CACHE: Dict[str, str] = {}

# Streaming cache: holds partial chit-chat tokens as they arrive from Claude.
# Frontend polls this or receives tokens via callback.
_STREAM_CACHES: Dict[str, str] = {}
_STREAM_CACHES_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# v3.5 Public helpers for frontend streaming integration
# ---------------------------------------------------------------------------

def get_streaming_chit_chat(thread_id: str) -> Optional[str]:
    """Poll the current streamed chit-chat for the given thread.

    Returns None if the thread has no active stream or was cleared.
    The returned string may be partial — call this in a poll loop for
    打字机 (typewriter) effect on the frontend.
    """
    with _STREAM_CACHES_LOCK:
        return _STREAM_CACHES.get(thread_id)


def clear_streaming_state(thread_id: str) -> None:
    """Release streaming and reasoning caches for a thread.

    Call after the frontend has consumed the full chit-chat to avoid
    memory leaks in long-lived processes.
    """
    with _STREAM_CACHES_LOCK:
        _STREAM_CACHES.pop(thread_id, None)
    _REASONING_CACHE.pop(thread_id, None)


def clear_streaming_chit_chat(thread_id: str) -> None:
    """Clear only streamed chit-chat tokens, preserving reasoning for HITL resume."""
    with _STREAM_CACHES_LOCK:
        _STREAM_CACHES.pop(thread_id, None)


# ============================================================================
# LLM clients (lazy)
# ============================================================================

_claude: Optional[ChatAnthropic] = None
_deepseek_fast: Optional[OpenAI] = None
_deepseek_pro: Optional[OpenAI] = None


def _get_claude() -> ChatAnthropic:
    global _claude
    if _claude is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        _claude = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.3,
            max_tokens=4096,
            api_key=api_key,
        )
    return _claude


def _get_deepseek_fast() -> OpenAI:
    """deepseek-v4-flash — cheap, fast, for simple vibe analysis."""
    global _deepseek_fast
    if _deepseek_fast is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
        _deepseek_fast = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _deepseek_fast


def _get_deepseek_pro() -> OpenAI:
    """deepseek-v4-pro — deep reasoning with thinking chain."""
    global _deepseek_pro
    if _deepseek_pro is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")
        _deepseek_pro = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _deepseek_pro


# ============================================================================
# Qdrant (in-memory) — index few-shot examples on import
# ============================================================================

qdrant = QdrantClient(":memory:")


def _make_sparse_vector(text: str, dim: int = VECTOR_DIM) -> List[float]:
    buckets = [0.0] * dim
    for token in text.lower().split():
        h = hash(token)
        idx = abs(h) % dim
        buckets[idx] += 1.0
    norm = sum(v * v for v in buckets) ** 0.5
    if norm > 0:
        buckets = [v / norm for v in buckets]
    return buckets


def _index_few_shots() -> None:
    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=VECTOR_DIM, distance=qdrant_models.Distance.COSINE,
        ),
    )
    points = [
        qdrant_models.PointStruct(
            id=i, vector=_make_sparse_vector(ex["user_vibe"]),
            payload={"user_vibe": ex["user_vibe"], "agent_target_prompt": ex["agent_target_prompt"]},
        )
        for i, ex in enumerate(FEW_SHOT_EXAMPLES)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


_index_few_shots()


# ============================================================================
# State
# ============================================================================

class VibeState(TypedDict):
    # ---- input ----
    user_input: str

    # ---- routing ----
    complexity: str                  # "simple" | "complex"

    # ---- deep_reason output (complex path) ----
    # v3.5: stored as compact marker; full text lives in _REASONING_CACHE
    reasoning_summary: str

    # ---- analyze output (both paths) ----
    vibe_keywords: List[str]
    matched_canonical_vibes: List[str]
    llm_confidence: str
    inferred_style: str
    soul_words: List[str]
    inferred_constraints: str
    dynamic_directions: List[Dict[str, Any]]
    selected_dynamic_direction: Dict[str, Any]
    direction_choice_id: str
    needs_direction_choice: bool

    # ---- retrieve output ----
    vibe_profiles: List[VibeProfile]
    few_shots: List[FewShotExample]

    # ---- clarify_check ----
    needs_clarification: bool
    clarification_question: str
    clarification_options: List[str]
    user_choice: str

    # ---- compose output ----
    generated_prompt: str

    # ---- quality gate (v3.0) ----
    quality_retry_count: int          # max 1 self-correction retry

    # ---- meta ----
    stage: str


# ============================================================================
# Node 1: route_complexity
# ============================================================================

def route_complexity(state: VibeState) -> dict:
    """v4.0: every request enters DeepSeek v4-pro for dynamic directions."""
    return {"complexity": "complex", "stage": "route"}


# ============================================================================
# v3.5: Preliminary retrieval — runs in parallel with LLM analysis
# ============================================================================

def _preliminary_retrieve(user_input: str) -> Dict[str, Any]:
    """Fast, rule-based vibe detection + qdrant lookup.

    Runs concurrently with the LLM analysis node (asyncio-style via
    ThreadPoolExecutor).  Uses the same _direct_scan_aliases +
    _match_keywords_to_vibes pipeline that analyze_fast's fallback
    path already uses — but formally split out so we can overlap it
    with the LLM call.
    """
    keywords = _direct_scan_aliases(user_input)
    matched = _match_keywords_to_vibes(keywords)
    vibe_profiles: List[VibeProfile] = []
    few_shots: List[FewShotExample] = []
    seen_ids: set = set()

    for vibe_name in matched:
        if vibe_name in VIBE_TECH_MAPPING:
            vibe_profiles.append(VIBE_TECH_MAPPING[vibe_name])

    # qdrant search (in-memory, < 5ms)
    for vibe_name in matched:
        vec = _make_sparse_vector(vibe_name)
        resp = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=vec,
            limit=2, with_payload=True,
        )
        for hit in resp.points:
            if hit.id not in seen_ids and hit.payload:
                few_shots.append(FewShotExample(
                    user_vibe=hit.payload["user_vibe"],
                    agent_target_prompt=hit.payload["agent_target_prompt"],
                ))
                seen_ids.add(hit.id)

    # Fallback: keyword match if qdrant didn't help
    if not few_shots:
        few_shots = _keyword_match_few_shots(matched)

    return {
        "matched_canonical_vibes": matched,
        "vibe_profiles": vibe_profiles,
        "few_shots": few_shots,
    }


def _supplementary_retrieve(new_vibes: List[str], existing: Dict[str, Any]) -> Dict[str, Any]:
    """Merge additional LLM-discovered vibes into preliminary retrieval results.

    Called after the concurrent LLM + retrieve step finishes — only the
    incremental qdrant queries run now (~1-3ms), so total latency stays
    max(T_llm, T_retrieve_prelim) + tiny_supplement.
    """
    existing_matched: List[str] = existing.get("matched_canonical_vibes", [])
    existing_vibes_set = set(existing_matched)
    existing_profiles: List[VibeProfile] = existing.get("vibe_profiles", [])
    existing_shots: List[FewShotExample] = existing.get("few_shots", [])
    seen_ids: set = {id(fs) for fs in existing_shots}  # identity-based dedup

    for vibe_name in new_vibes:
        if vibe_name in existing_vibes_set:
            continue
        if vibe_name in VIBE_TECH_MAPPING:
            existing_profiles.append(VIBE_TECH_MAPPING[vibe_name])
            existing_matched.append(vibe_name)
            existing_vibes_set.add(vibe_name)
        # qdrant for this new vibe
        vec = _make_sparse_vector(vibe_name)
        resp = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=vec,
            limit=2, with_payload=True,
        )
        for hit in resp.points:
            if hit.id not in seen_ids and hit.payload:
                existing_shots.append(FewShotExample(
                    user_vibe=hit.payload["user_vibe"],
                    agent_target_prompt=hit.payload["agent_target_prompt"],
                ))
                seen_ids.add(hit.id)

    return {
        "matched_canonical_vibes": existing_matched,
        "vibe_profiles": existing_profiles,
        "few_shots": existing_shots,
    }


# ============================================================================
# v3.5: Concurrent analysis + retrieval nodes
# ============================================================================

def _run_analyze_fast_isolated(user_input: str) -> dict:
    """Run analyze_fast's core logic and return a raw dict.

    Extracted from the old analyze_fast node so it can be executed in a
    ThreadPoolExecutor alongside _preliminary_retrieve.
    """
    analysis = _call_deepseek_json(
        _get_deepseek_fast,
        model="deepseek-v4-flash",
        prompt=_ANALYZE_FAST_PROMPT.format(
            canonical_list="、".join(VIBE_TECH_MAPPING.keys()),
            user_input=user_input,
        ),
        fallback={"matched_vibes": [], "inferred_style": "", "soul_words": [],
                   "inferred_constraints": "", "confidence": "medium", "needs_clarification": False},
    )

    matched = _ensure_list(analysis.get("matched_vibes"))
    matched = _filter_valid_vibes(matched)
    inferred_style = str(analysis.get("inferred_style", "") or "")
    soul_words = _ensure_list(analysis.get("soul_words"))
    inferred_constraints = str(analysis.get("inferred_constraints", "") or "")
    confidence = analysis.get("confidence", "medium") or "medium"
    needs_clarify = bool(analysis.get("needs_clarification", False))

    # Fallback: rule scan
    if not matched and not inferred_style:
        keywords = _direct_scan_aliases(user_input)
        fb = _match_keywords_to_vibes(keywords)
        if fb:
            matched = fb
            confidence = "medium"

    return {
        "matched_vibes": matched,
        "inferred_style": inferred_style,
        "soul_words": soul_words,
        "inferred_constraints": inferred_constraints,
        "confidence": confidence,
        "needs_clarification": needs_clarify,
    }


def concurrent_analyze_and_retrieve(state: VibeState) -> dict:
    """v3.5: Run analyze_fast and retrieve_knowledge in parallel.

    Uses ThreadPoolExecutor to overlap the DeepSeek API call (~1-3s)
    with the local qdrant+rule retrieval (~1-5ms).  After both
    complete, supplements with any vibes the LLM found that the
    rule-based scan missed.
    """
    user_input = state.get("user_input", "")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_analyze = pool.submit(_run_analyze_fast_isolated, user_input)
        fut_retrieve = pool.submit(_preliminary_retrieve, user_input)

        # Wait for both — total latency = max(T_llm, T_retrieve)
        analysis = fut_analyze.result()
        prelim = fut_retrieve.result()

    # Merge: LLM may have found vibes the rule scan missed
    merged = _supplementary_retrieve(analysis["matched_vibes"], prelim)

    return _build_analyze_output(
        merged["matched_canonical_vibes"], analysis["inferred_style"],
        analysis["soul_words"], analysis["inferred_constraints"],
        analysis["confidence"], analysis["needs_clarification"],
        user_input, "concurrent_analyze",
        extra={
            "vibe_profiles": merged["vibe_profiles"],
            "few_shots": merged["few_shots"],
        },
    )


def _normalize_dynamic_directions(raw: Any, user_input: str) -> List[Dict[str, Any]]:
    """Normalize DeepSeek's free-form direction ideas into stable UI data."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []

    if isinstance(raw, dict):
        raw = raw.get("items") or raw.get("directions") or raw.get("dynamic_directions") or []

    directions: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw[:3], 1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or f"方向 {idx}").strip()
            focus = str(item.get("focus") or item.get("description") or item.get("summary") or "").strip()
            if not title and not focus:
                continue
            direction_id = str(item.get("id") or f"direction_{idx}").strip()
            design_tokens = _ensure_text_list(item.get("design_tokens") or item.get("visual_or_design_tokens"))
            performance_budget = _ensure_text_list(item.get("performance_budget"))
            anti_patterns = _ensure_text_list(item.get("anti_patterns") or item.get("anti_patterns_to_avoid"))
            engineering_contract = _ensure_text_list(item.get("engineering_contract") or item.get("architecture_constraints"))
            directions.append({
                "id": direction_id,
                "title": title,
                "focus": focus,
                "design_tokens": design_tokens,
                "performance_budget": performance_budget,
                "anti_patterns": anti_patterns,
                "engineering_contract": engineering_contract,
            })

    if directions:
        return directions[:3]

    compact_input = user_input[:80] or "当前需求"
    return [
        {
            "id": "direction_1",
            "title": "核心体验闭环优先",
            "focus": f"围绕「{compact_input}」先把用户主路径、状态反馈和异常恢复闭环做扎实。",
            "design_tokens": [
                "根据业务语义生成颜色、字体、间距、状态色和动效曲线，不使用固定风格模板。",
                "所有核心状态必须有可见反馈：loading、success、empty、error、retry。",
            ],
            "performance_budget": [
                "核心交互反馈 ≤ 100ms",
                "首屏 TTI ≤ 1.5s @ 4G",
                "P95 API 响应 ≤ 300ms",
            ],
            "anti_patterns": [
                "禁止只输出视觉形容词而没有 CSS token 或组件规则。",
                "禁止把失败路径留成 TODO。",
            ],
            "engineering_contract": [
                "先交付端到端主路径，再扩展高级功能。",
            ],
        },
        {
            "id": "direction_2",
            "title": "架构韧性与可观测性优先",
            "focus": f"围绕「{compact_input}」优先设计幂等、降级、trace_id、结构化日志和测试断言。",
            "design_tokens": [
                "界面状态必须明确区分正常、警告、失败、恢复中。",
            ],
            "performance_budget": [
                "故障恢复提示 ≤ 200ms",
                "重试退避上限 ≤ 3 次",
                "关键链路错误率告警阈值 ≤ 1%",
            ],
            "anti_patterns": [
                "禁止空白 catch 块。",
                "禁止无 trace_id 的日志。",
            ],
            "engineering_contract": [
                "每个关键请求必须贯穿 trace_id 并可在日志中还原链路。",
            ],
        },
    ]


def _run_deep_reason_isolated(user_input: str, thread_id: str) -> dict:
    """Run deep_reason's core logic and return a raw dict.

    Extracted from the old deep_reason node so it can be executed in a
    ThreadPoolExecutor alongside _preliminary_retrieve.
    """
    raw = _call_deepseek_json(
        _get_deepseek_pro,
        model="deepseek-v4-pro",
        prompt=_DEEP_REASON_PROMPT.format(
            canonical_list="、".join(VIBE_TECH_MAPPING.keys()),
            user_input=user_input,
        ),
        extra_body={"reasoning_effort": "high"},
        fallback={
            "visual_layer": "", "interaction_layer": "", "architecture_layer": "",
            "dynamic_directions": [],
            "matched_vibes": [], "inferred_style": "", "soul_words": [],
            "inferred_constraints": "", "confidence": "medium", "needs_clarification": False,
        },
    )

    # ---- Assemble reasoning_summary ----
    parts = []
    for key, label in [("visual_layer", "视觉层"), ("interaction_layer", "交互层"),
                        ("architecture_layer", "架构层")]:
        val = str(raw.get(key, "") or "").strip()
        if val:
            parts.append(f"## {label}\n{val}")
    reasoning_summary = "\n\n".join(parts) if parts else "（DeepSeek 推理不可用，已降级）"
    dynamic_directions = _normalize_dynamic_directions(
        raw.get("dynamic_directions"), user_input,
    )

    # v3.5: Store full reasoning in module-level cache; keep checkpoint payload light
    _REASONING_CACHE[thread_id] = reasoning_summary

    matched = _ensure_list(raw.get("matched_vibes"))
    matched = _filter_valid_vibes(matched)
    inferred_style = str(raw.get("inferred_style", "") or "")
    soul_words = _ensure_list(raw.get("soul_words"))
    inferred_constraints = str(raw.get("inferred_constraints", "") or "")
    confidence = raw.get("confidence", "medium") or "medium"
    needs_clarify = bool(raw.get("needs_clarification", False))

    return {
        "matched_vibes": matched,
        "inferred_style": inferred_style,
        "soul_words": soul_words,
        "inferred_constraints": inferred_constraints,
        "confidence": confidence,
        "needs_clarification": needs_clarify,
        "reasoning_summary": reasoning_summary,
        "dynamic_directions": dynamic_directions,
    }


def concurrent_reason_and_retrieve(state: VibeState, config: RunnableConfig) -> dict:
    """v3.5: Run deep_reason and retrieve_knowledge in parallel.

    Same pattern as concurrent_analyze_and_retrieve but for the complex
    (DeepSeek v4-pro with reasoning_effort=high) path.

    Stores full reasoning_summary in _REASONING_CACHE[thread_id] and
    writes a compact marker into the state to keep checkpoint payloads
    lightweight.
    """
    user_input = state.get("user_input", "")
    thread_id = config["configurable"]["thread_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_reason = pool.submit(_run_deep_reason_isolated, user_input, thread_id)
        fut_retrieve = pool.submit(_preliminary_retrieve, user_input)

        # Wait for both — total latency = max(T_deep_reason, T_retrieve)
        analysis = fut_reason.result()
        prelim = fut_retrieve.result()

    # Merge: LLM may have found vibes the rule scan missed
    merged = _supplementary_retrieve(analysis["matched_vibes"], prelim)

    # v3.5: Compact marker in state — full text is in _REASONING_CACHE
    full_reasoning = analysis["reasoning_summary"]
    compact_marker = f"[REASONING_CACHED:{len(full_reasoning)} chars]"

    return _build_analyze_output(
        merged["matched_canonical_vibes"], analysis["inferred_style"],
        analysis["soul_words"], analysis["inferred_constraints"],
        analysis["confidence"], analysis["needs_clarification"],
        user_input, "concurrent_reason",
        extra={
            "reasoning_summary": compact_marker,
            "vibe_profiles": merged["vibe_profiles"],
            "few_shots": merged["few_shots"],
            "dynamic_directions": analysis["dynamic_directions"],
            "needs_direction_choice": bool(analysis["dynamic_directions"]),
        },
    )


# ============================================================================
# Legacy nodes (kept for reference; graph uses concurrent versions above)
# ============================================================================

_ANALYZE_FAST_PROMPT = """你是一个高效的需求标签提取器。用户会描述一个软件需求，请快速提取关键信息。

已知风格标签：{canonical_list}

用户输入："{user_input}"

返回严格 JSON（不要 markdown）：
{{"matched_vibes": [...], "inferred_style": "", "soul_words": [...], "inferred_constraints": "", "confidence": "high|medium|low", "needs_clarification": false}}

规则：
- matched_vibes: 从已知标签中匹配（可空数组）
- soul_words: 用户原话中有画面感的词
- confidence: high=明确匹配 / medium=可推断 / low=无法匹配
- needs_clarification: 仅当输入极短无意义时为 true"""


_DEEP_REASON_PROMPT = """你是一位世界级的「动态技术方向推演专家」。用户会用大白话、业务场景、影视艺术隐喻或情绪词描述一个软件产品的感觉。你必须独立思考，不要套用固定技术分类，不要被本地风格字典限制。

## 用户输入
"{user_input}"

## 旧版本地风格标签（只能作为弱参考，不得依赖）
{canonical_list}

## 拆解要求（严格返回 JSON）

{{
  "visual_layer": "视觉层：推导配色方案（精确到色号）、空间布局（密度、留白、对称性）、动效风格（缓动曲线类型、粒子/光效/故障/毛玻璃等）、字体气质",
  "interaction_layer": "交互层：信息呈现节奏（一步一息 / 快速扫视 / 沉浸漫游）、微交互风格（弹性/磁性/滞后/即时）、手势语言、反馈密度",
  "architecture_layer": "架构层：推荐的技术栈组合（前端框架、动画/渲染方案、数据通道）、该风格特有的架构约束（如 SSR 不能用因为需要 Canvas、或必须 WebSocket 实时推送）",
  "dynamic_directions": [
    {{
      "id": "direction_1",
      "title": "自发命名的技术演进方向，例如 MQ 削峰容灾链路",
      "focus": "为什么这个方向最适合当前用户输入，必须绑定用户上下文，不要写通用套话",
      "engineering_contract": ["该方向下最关键的 3-5 条架构/组件/API/数据约束"],
      "design_tokens": ["该方向自己推理出来的 CSS/Design Tokens/交互规则/渲染策略，必须具体到色号、字体、动效曲线、布局密度或底层架构约束"],
      "performance_budget": ["该方向专属的量化性能预算，例如 P95 API ≤ 300ms、队列堆积 ≤ 5000、动画 ≥ 60fps"],
      "anti_patterns": ["该方向必须避免的偷懒写法、技术误区、伪代码或美学跑偏"]
    }}
  ],
  "matched_vibes": ["从已知标签中匹配 0-3 个最接近的"],
  "inferred_style": "3-5 句完整风格描述（中英混合），将隐喻精确转化为工程可理解的美学方向",
  "soul_words": ["用户原话中最有张力的词"],
  "inferred_constraints": "5-8 条具体技术约束，每条必须可验证（如'粒子数量 >= 200，帧率 >= 30fps，使用 OffscreenCanvas'）",
  "confidence": "high|medium|low",
  "needs_clarification": false
}}

<dynamic_directions> 规则：
- 必须输出 2-3 个最适合当前输入的方向，方向名称和内容由你现场推理生成，禁止套用固定分类。
- 每个方向必须有自己的 design_tokens、performance_budget、anti_patterns，后续 Claude 会只围绕用户选中的方向生成最终 XML Prompt。
- 如果用户输入是“抢购/秒杀/活动报名”，可以推理出 MQ 削峰容灾链路、库存一致性链路、前端秒级乐观 UI 链路；如果不是，不要照搬这些例子。
- 由于本次要求严格 JSON，最终字段名使用 "dynamic_directions"，其语义等同于 <dynamic_directions>。
- 不要偷懒写成空泛形容词。视觉层必须给色号，架构层必须给框架名，预算必须有数字。"""


# Legacy single-node functions (kept for backward reference):
# analyze_fast and deep_reason are still importable but the graph uses
# concurrent_analyze_and_retrieve / concurrent_reason_and_retrieve instead.

def analyze_fast(state: VibeState) -> dict:
    """Legacy: single-node fast analysis. Graph uses concurrent_analyze_and_retrieve."""
    result = _run_analyze_fast_isolated(state.get("user_input", ""))
    prelim = _preliminary_retrieve(state.get("user_input", ""))
    merged = _supplementary_retrieve(result["matched_vibes"], prelim)
    return _build_analyze_output(
        merged["matched_canonical_vibes"], result["inferred_style"],
        result["soul_words"], result["inferred_constraints"],
        result["confidence"], result["needs_clarification"],
        state.get("user_input", ""), "analyze_fast",
        extra={
            "vibe_profiles": merged["vibe_profiles"],
            "few_shots": merged["few_shots"],
        },
    )


def deep_reason(state: VibeState) -> dict:
    """Legacy: single-node deep reasoning. Graph uses concurrent_reason_and_retrieve."""
    user_input = state.get("user_input", "")
    # Generate a temporary thread_id for cache if none available
    thread_id = str(uuid.uuid4())
    result = _run_deep_reason_isolated(user_input, thread_id)
    prelim = _preliminary_retrieve(user_input)
    merged = _supplementary_retrieve(result["matched_vibes"], prelim)
    return _build_analyze_output(
        merged["matched_canonical_vibes"], result["inferred_style"],
        result["soul_words"], result["inferred_constraints"],
        result["confidence"], result["needs_clarification"],
        user_input, "deep_reason",
        extra={
            "reasoning_summary": result["reasoning_summary"],
            "vibe_profiles": merged["vibe_profiles"],
            "few_shots": merged["few_shots"],
            "dynamic_directions": result.get("dynamic_directions", []),
            "needs_direction_choice": bool(result.get("dynamic_directions", [])),
        },
    )


# Legacy standalone retrieve (kept for reference; graph no longer uses it
# as a separate node — retrieval runs inside the concurrent nodes above).

def retrieve_knowledge(state: VibeState) -> dict:
    """Legacy: standalone retrieval node."""
    matched = state.get("matched_canonical_vibes", [])
    inferred_style = state.get("inferred_style", "")
    vibe_profiles: List[VibeProfile] = []
    few_shots: List[FewShotExample] = []

    for vibe_name in matched:
        if vibe_name in VIBE_TECH_MAPPING:
            vibe_profiles.append(VIBE_TECH_MAPPING[vibe_name])

    seen_ids: set = set()
    for vibe_name in matched:
        vec = _make_sparse_vector(vibe_name)
        resp = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=vec,
            limit=2, with_payload=True,
        )
        for hit in resp.points:
            if hit.id not in seen_ids and hit.payload:
                few_shots.append(FewShotExample(
                    user_vibe=hit.payload["user_vibe"],
                    agent_target_prompt=hit.payload["agent_target_prompt"],
                ))
                seen_ids.add(hit.id)

    if not matched and inferred_style and not few_shots:
        vec = _make_sparse_vector(inferred_style)
        resp = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=vec,
            limit=3, with_payload=True,
        )
        for hit in resp.points:
            if hit.id not in seen_ids and hit.payload:
                few_shots.append(FewShotExample(
                    user_vibe=hit.payload["user_vibe"],
                    agent_target_prompt=hit.payload["agent_target_prompt"],
                ))
                seen_ids.add(hit.id)

    if not few_shots:
        search_terms = matched if matched else ([inferred_style] if inferred_style else [])
        few_shots = _keyword_match_few_shots(search_terms)

    return {"vibe_profiles": vibe_profiles, "few_shots": few_shots, "stage": "retrieve"}


# ============================================================================
# Node 4: clarify_check  (unchanged)
# ============================================================================

def clarify_check(state: VibeState) -> dict:
    matched = state.get("matched_canonical_vibes", [])
    inferred_style = state.get("inferred_style", "")
    confidence = state.get("llm_confidence", "medium")
    user_choice = state.get("user_choice", "")
    already_needs = state.get("needs_clarification", False)

    if user_choice and not matched:
        if user_choice in VIBE_TECH_MAPPING:
            return {"matched_canonical_vibes": [user_choice], "needs_clarification": False,
                    "llm_confidence": confidence, "stage": "clarify_check"}

    if matched:
        return {"needs_clarification": False, "llm_confidence": confidence, "stage": "clarify_check"}

    if inferred_style and confidence != "low":
        return {"needs_clarification": False, "llm_confidence": confidence, "stage": "clarify_check"}

    if already_needs:
        return {"needs_clarification": True,
                "clarification_question": "你的描述比较抽象，请选择最接近你想要的方向：",
                "clarification_options": list(VIBE_TECH_MAPPING.keys()),
                "stage": "clarify_check"}

    return {"needs_clarification": True,
            "clarification_question": "你的描述比较抽象，请选择最接近你想要的方向：",
            "clarification_options": list(VIBE_TECH_MAPPING.keys()),
            "stage": "clarify_check"}


# ============================================================================
# Node 5: compose_with_claude  (v4.0 XML contract + v3.5 streaming split)
# ============================================================================

def _resolve_reasoning_summary(state: VibeState, config: RunnableConfig) -> str:
    """v3.5: Read reasoning_summary from cache when available.

    If the state holds a compact marker, look up the full text in
    _REASONING_CACHE.  Otherwise fall back to the state value directly.
    """
    from_state = state.get("reasoning_summary", "") or ""
    if from_state.startswith("[REASONING_CACHED:"):
        thread_id = config["configurable"]["thread_id"]
        cached = _REASONING_CACHE.get(thread_id)
        if cached:
            return cached
        # cache miss — the marker itself is still informative
        return f"（推理结果已缓存但不可用；原始大小约{from_state[len('[REASONING_CACHED:'):].rstrip(']')}）"
    return from_state


def _build_chit_chat_prompt(
    user_input: str, matched: List[str], inferred_style: str,
    soul_words: List[str], inferred_constraints: str,
    reasoning_summary: str, complexity: str,
    vibe_profiles: List[VibeProfile], few_shots: List[FewShotExample],
    tone_hint: str,
) -> str:
    """v3.5: Build a lightweight chit-chat-only prompt for streaming.

    This is intentionally SHORT — just the persona + user context.
    The full industrial section descriptions are EXCLUDED so Claude
    can stream the chit-chat quickly without burning tokens on the
    industrial prompt specification.
    """
    soul_words_text = "、".join(soul_words) if soul_words else "（无特殊修饰语）"

    # Brief DeepSeek reasoning boost
    reasoning_block = ""
    if reasoning_summary and complexity == "complex":
        reasoning_block = f"""
## 🧠 DeepSeek 深度隐喻拆解（摘要）
{reasoning_summary[:800]}{'…' if len(reasoning_summary) > 800 else ''}

> 你已充分吸收上述拆解，请在碎碎念中自然引用其中亮点。"""

    return f"""{_CHIT_CHAT_SYSTEM_PROMPT}

## 对话背景
用户刚才说：> 「{user_input}」

- 匹配到的字典风格：{"、".join(matched) if matched else "（无现成标签，纯靠脑补）"}
- 用户独有的灵魂词汇：{soul_words_text}
- 风格推断：{inferred_style if inferred_style else "（无需额外脑补）"}
- 种子约束：{inferred_constraints if inferred_constraints else "（无）"}
{reasoning_block}

## 你的任务
写一段「开发者碎碎念」（3-5句话），搭子语气。
语调提示：{"用户需求非常规、有想象力 → 疯狂捧哏" if tone_hint == "wild" else ("有已知风格但加了私货 → 肯定品味、点出亮点" if tone_hint == "hybrid" else "常规需求 → 温柔共情，幽默吐槽")}
- 如果上面有 DeepSeek 的隐喻拆解，可以赞叹一句"DeepSeek 把你这感觉拆得挺细的"
- 提到 1-2 个你准备用的技术点
- 以 "### 💬 开发者碎碎念\\n" 开头
- 只输出碎碎念，不要输出工业 Prompt 部分"""


def _build_industrial_prompt(
    user_input: str, matched: List[str], inferred_style: str,
    soul_words: List[str], inferred_constraints: str,
    reasoning_summary: str, complexity: str,
    vibe_profiles: List[VibeProfile], few_shots: List[FewShotExample],
    tone_hint: str, selected_dynamic_direction: Optional[Dict[str, Any]] = None,
) -> str:
    """v3.5: Build the industrial prompt (batch, non-streamed).

    This is the full Claude call — same as the old compose_with_claude
    but WITHOUT the chit-chat section (that's streamed separately).
    """
    profile_summaries = _format_profiles(vibe_profiles)
    few_shot_text = _format_few_shots(few_shots)
    soul_words_text = "、".join(soul_words) if soul_words else "（无特殊修饰语）"
    design_tokens_text = _format_design_tokens_and_rules(
        matched, vibe_profiles, inferred_style,
    )
    selected_direction_text = _format_dynamic_direction(selected_dynamic_direction or {})

    inferred_context = ""
    if inferred_style:
        inferred_context += f"\n\n## LLM 脑补的风格推断\n{inferred_style}"
    if inferred_constraints:
        inferred_context += f"\n\n## 推导的种子约束（请扩展深化）\n{inferred_constraints}"

    reasoning_block = ""
    if reasoning_summary and complexity == "complex":
        reasoning_block = f"""

## 🧠 DeepSeek 深度隐喻拆解（思维链）
{reasoning_summary}

> 请将上述拆解作为高权重参考，在 <business_context>、<visual_or_design_tokens> 和 <dynamic_engineering_contract> 中充分吸收其细节。"""

    return f"""{_CHIT_CHAT_SYSTEM_PROMPT}

## 对话背景
用户刚才说了一段话：

> 用户原话：「{user_input}」

- 匹配到的字典风格：{"、".join(matched) if matched else "（无现成标签，纯靠脑补）"}
- 用户独有的灵魂词汇：{soul_words_text}
- 风格推断：{inferred_style if inferred_style else "（无需额外脑补）"}
- 种子约束：{inferred_constraints if inferred_constraints else "（无）"}
{reasoning_block}
- 风格档案参考：
{profile_summaries}

- 必须注入的 Design Tokens / 架构硬约束：
{design_tokens_text}

- 用户终审选择的动态技术主攻方向（最高优先级，必须覆盖本地固定映射）：
{selected_direction_text}

- Few-Shot 结构参考：
{few_shot_text}

## 你的任务
输出「复制去投喂 Cursor」的工业级 Prompt，以 "### 🤖 复制去投喂 Cursor" 开头。
标题之后必须是 XML 严格作用域结构，且只能包含指定 XML 标签。不要沿用 Few-Shot 的 [Context]/[Persona]/[Constraints] Markdown 结构。

{_SMART_SECTIONS_DESC}

## ⚠️ 输出格式（必须逐字遵守标签结构，禁止代码围栏）
### 🤖 复制去投喂 Cursor
<system_role>
...
</system_role>

<business_context>
...
</business_context>

<dynamic_engineering_contract>
  <performance_budget>
  必须优先采用用户所选动态方向里的 performance_budget，再补足全局性能预算。
  </performance_budget>
  <edge_case_defense>
  ...
  </edge_case_defense>
  <visual_or_design_tokens>
  必须优先采用用户所选动态方向里的 design_tokens；本地映射只能作为补充，不得压过动态方向。
  </visual_or_design_tokens>
</dynamic_engineering_contract>

<anti_patterns_to_avoid>
  - 必须合并用户所选动态方向里的 anti_patterns。
  - 严禁使用 any 类型。
  - 严禁出现任何空白的 catch 块，必须至少打印 error 日志并带上 trace_id。
  - 严禁在循环体内部进行 await 异步网络请求，必须使用 Promise.all。
  - 严禁出现任何 "// TODO" 或 "// 随后补充" 的伪代码，所有逻辑必须完整闭环交付。
</anti_patterns_to_avoid>

<test_assertions_contract>
...
</test_assertions_contract>

工业 Prompt XML 本体严禁 emoji / 口语 / 玩笑。"""


def compose_with_claude(state: VibeState, config: RunnableConfig) -> dict:
    """v3.5: Two-phase streaming compose.

    Phase 1 — Chit-chat: stream=True, tokens accumulated into
              _STREAM_CACHES[thread_id] for frontend polling
              (打字机效果).

    Phase 2 — Industrial prompt: batched invoke(), assembled from the
              full meta-prompt less the chit-chat header.

    The chit-chat prompt is deliberately shorter than the industrial
    prompt (~400 tokens vs ~2000 tokens), so Phase 1 finishes quickly
    and the user sees the搭子 commentary almost immediately.
    """
    user_input = state.get("user_input", "")
    thread_id = config["configurable"]["thread_id"]
    matched = state.get("matched_canonical_vibes", [])
    vibe_profiles = list(state.get("vibe_profiles", []))
    few_shots = list(state.get("few_shots", []))
    inferred_style = state.get("inferred_style", "")
    soul_words = state.get("soul_words", [])
    inferred_constraints = state.get("inferred_constraints", "")
    complexity = state.get("complexity", "simple")
    selected_dynamic_direction = dict(state.get("selected_dynamic_direction", {}) or {})

    reasoning_summary = _resolve_reasoning_summary(state, config)

    # ---- Re-retrieve if needed (guard against empty retrieval) ----
    if not vibe_profiles and matched:
        for v in matched:
            if v in VIBE_TECH_MAPPING:
                vibe_profiles.append(VIBE_TECH_MAPPING[v])
    if not few_shots and matched:
        seen_ids: set = set()
        for v in matched:
            vec = _make_sparse_vector(v)
            resp = qdrant.query_points(
                collection_name=COLLECTION_NAME, query=vec,
                limit=2, with_payload=True,
            )
            for hit in resp.points:
                if hit.id not in seen_ids and hit.payload:
                    few_shots.append(FewShotExample(
                        user_vibe=hit.payload["user_vibe"],
                        agent_target_prompt=hit.payload["agent_target_prompt"],
                    ))
                    seen_ids.add(hit.id)
        if not few_shots:
            few_shots = _keyword_match_few_shots(matched)

    # ---- Tone hint ----
    is_novel = bool(inferred_style and not matched)
    is_hybrid = bool(matched and inferred_style)
    tone_hint = "wild" if is_novel else ("hybrid" if is_hybrid else "routine")

    # ---- Phase 1: Stream chit-chat ----
    chit_chat_prompt = _build_chit_chat_prompt(
        user_input, matched, inferred_style, soul_words,
        inferred_constraints, reasoning_summary, complexity,
        vibe_profiles, few_shots, tone_hint,
    )

    # ---- Phase 2: Batch industrial prompt ----
    industrial_prompt = _build_industrial_prompt(
        user_input, matched, inferred_style, soul_words,
        inferred_constraints, reasoning_summary, complexity,
        vibe_profiles, few_shots, tone_hint,
        selected_dynamic_direction=selected_dynamic_direction,
    )

    # Collect results
    chit_chat_text = ""
    industrial_text = ""

    try:
        # ---- Phase 1: Stream chit-chat (token-by-token for 打字机效果) ----
        claude = _get_claude()
        stream_acc: List[str] = []
        with _STREAM_CACHES_LOCK:
            _STREAM_CACHES[thread_id] = ""
        for chunk in claude.stream(chit_chat_prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                stream_acc.append(token)
                with _STREAM_CACHES_LOCK:
                    _STREAM_CACHES[thread_id] = "".join(stream_acc)
        chit_chat_text = "".join(stream_acc).strip()
        if not chit_chat_text.startswith("### 💬"):
            chit_chat_text = f"### 💬 开发者碎碎念\n{chit_chat_text}"

    except Exception:
        import random
        chit_chat_text = _build_fallback_chit_chat(
            user_input, "、".join(soul_words),
            "、".join(matched) if matched else "自定义风格",
            inferred_style,
        )
        with _STREAM_CACHES_LOCK:
            _STREAM_CACHES[thread_id] = chit_chat_text

    # ---- Phase 2: Batch industrial prompt ----
    try:
        resp = claude.invoke(industrial_prompt)
        industrial_text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if not industrial_text.startswith("### 🤖"):
            industrial_text = f"### 🤖 复制去投喂 Cursor\n{industrial_text}"
    except Exception:
        industrial_text = _fallback_industrial(
            user_input, vibe_profiles, few_shots,
            inferred_style=inferred_style, soul_words=soul_words,
            inferred_constraints=inferred_constraints,
            reasoning_summary=reasoning_summary,
            matched=matched,
            selected_dynamic_direction=selected_dynamic_direction,
        )

    industrial_text = _normalize_industrial_xml_contract(
        industrial_text,
        user_input=user_input,
        vibe_profiles=vibe_profiles,
        few_shots=few_shots,
        matched=matched,
        inferred_style=inferred_style,
        soul_words=soul_words,
        inferred_constraints=inferred_constraints,
        reasoning_summary=reasoning_summary,
        selected_dynamic_direction=selected_dynamic_direction,
    )

    # ---- Assemble final output ----
    generated = f"{chit_chat_text}\n\n---\n{industrial_text}"
    generated = _ensure_two_part_output(generated)

    return {"vibe_profiles": vibe_profiles, "few_shots": few_shots,
            "generated_prompt": generated, "stage": "compose"}


# ============================================================================
# Node 6: quality_check  (v3.0 industrial quality gate — unchanged)
# ============================================================================

_QUALITY_CHECK_PROMPT = """你是一个大厂代码审查员。请严格审查下面这份 Prompt，判断它是否达到工业级交付标准。

## 审查标准（四项必须全部满足）

A. 【性能预算】：是否包含具体量化指标（TTI < Xs, P95 < Xms, fps >= X 等）？
B. 【边界防御】：是否明确处理了空数据、网络抖动、慢查询、并发冲突等边界场景？
C. 【可观测性】：是否包含 trace_id、结构化日志、关键埋点事件？
D. 【测试断言】：是否包含至少 3 条可执行的验证断言（单元/集成/E2E）？

额外硬门槛：
- 必须采用 XML 严格作用域结构，包含 system_role、business_context、dynamic_engineering_contract、performance_budget、edge_case_defense、visual_or_design_tokens、anti_patterns_to_avoid、test_assertions_contract。
- visual_or_design_tokens 中必须出现具体 CSS/Design Tokens 或底层架构强约束。
- anti_patterns_to_avoid 必须包含 any、空白 catch、循环 await、TODO/随后补充四类禁令。

## 被审查的 Prompt
```
{prompt_text}
```

## 要求
返回严格 JSON（无 markdown 包裹）：
{{
  "score": 0-100（整数）,
  "perf_budget": true/false,
  "edge_cases": true/false,
  "observability": true/false,
  "test_assertions": true/false,
  "issues": ["具体缺失项列表"],
  "passed": true/false（四项全 true 才为 true）
}}"""


def quality_check(state: VibeState) -> dict:
    """Industrial quality gate — validates the generated prompt against 4 pillars.

    Uses deepseek-v4-flash for low-cost validation.
    - pass → route to END
    - fail + retry_count < 1 → route back to compose_with_claude (self-correction)
    - fail + retry_count >= 1 → route to END (give up, but log)
    """
    prompt = state.get("generated_prompt", "")
    retry_count = state.get("quality_retry_count", 0)

    # Fast pre-check: if the prompt is obviously a fallback or too short, skip
    if len(prompt) < 200:
        return {"stage": "quality_check"}

    # Extract only the industrial part (after 🤖) for validation
    industrial_part = prompt
    if "🤖 复制去投喂 Cursor" in prompt:
        industrial_part = prompt.split("🤖 复制去投喂 Cursor", 1)[-1]

    validation = _call_deepseek_json(
        _get_deepseek_fast,
        model="deepseek-v4-flash",
        prompt=_QUALITY_CHECK_PROMPT.format(prompt_text=industrial_part[:6000]),
        fallback={"score": 60, "perf_budget": False, "edge_cases": False,
                   "observability": False, "test_assertions": False,
                   "issues": ["quality check LLM unavailable"], "passed": False},
    )

    passed = bool(validation.get("passed", False))
    issues = _ensure_list(validation.get("issues"))

    if passed:
        return {"stage": "quality_check"}

    # ---- Self-correction path ----
    # Inject issues as feedback so compose_with_claude can fix them on retry
    feedback = "、".join(issues) if issues else "质量检查未通过，请补充工业四大要素"
    new_retry = retry_count + 1

    return {
        "stage": "quality_check",
        "quality_retry_count": new_retry,
        # Store feedback in inferred_constraints so compose_with_claude sees it
        "inferred_constraints": (
            (state.get("inferred_constraints", "") or "")
            + f"\n\n[质量闸退回反馈 — 第{new_retry}次修正] 缺失项：{feedback}。"
            + "请在 <dynamic_engineering_contract> 与 <test_assertions_contract> 中补充完整的性能预算/边界防御/可观测性/测试断言。"
        ),
    }


def _route_after_quality(state: VibeState) -> Literal["compose_with_claude", "__end__"]:
    """After quality check: pass → END, fail + retry_left → retry, fail + exhausted → END."""
    retry_count = state.get("quality_retry_count", 0)
    prompt = state.get("generated_prompt", "")

    # If quality explicitly passed (no issues injected), go to END
    inferred = state.get("inferred_constraints", "") or ""
    if "[质量闸退回反馈" not in inferred and len(prompt) >= 200:
        return "__end__"

    # Retry if under limit
    if retry_count <= 1:
        return "compose_with_claude"

    return "__end__"


# ============================================================================
# Shared helpers
# ============================================================================

def _build_analyze_output(
    matched: List[str], inferred_style: str, soul_words: List[str],
    inferred_constraints: str, confidence: str, needs_clarify: bool,
    user_input: str, stage: str, extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """Normalise the output dict for both analyze and reason paths."""
    has_signal = bool(matched or inferred_style or len(user_input.strip()) >= 8)
    needs_clarify = needs_clarify or not has_signal

    result: Dict[str, Any] = {
        "vibe_keywords": soul_words,
        "matched_canonical_vibes": [str(v).strip() for v in matched if str(v).strip()],
        "llm_confidence": confidence,
        "inferred_style": inferred_style,
        "soul_words": soul_words,
        "inferred_constraints": inferred_constraints,
        "needs_clarification": needs_clarify,
        "stage": stage,
    }
    if extra:
        result.update(extra)
    return result


def _call_deepseek_json(
    client_getter, model: str, prompt: str,
    fallback: dict, extra_body: Optional[dict] = None,
) -> dict:
    """Call a DeepSeek model, parse JSON response, return dict (or fallback)."""
    try:
        client = client_getter()
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3 if "flash" in model else 0.5,
            "max_tokens": 2048 if "flash" in model else 4096,
            "response_format": {"type": "json_object"},
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        return _strip_reasoning_artifacts(json.loads(text))
    except Exception:
        return fallback


def _strip_reasoning_artifacts(value: Any) -> Any:
    """Remove provider-specific reasoning fields before downstream prompting.

    DeepSeek thinking responses can expose reasoning_content on the message
    object. We currently parse only message.content, but this whitelist-style
    sanitizer protects us if a provider/proxy echoes reasoning fields inside
    JSON content.
    """

    blocked_keys = {
        "reasoning_content",
        "reasoning",
        "chain_of_thought",
        "cot",
        "thoughts",
        "thinking",
    }
    if isinstance(value, dict):
        return {
            str(k): _strip_reasoning_artifacts(v)
            for k, v in value.items()
            if str(k).lower() not in blocked_keys
        }
    if isinstance(value, list):
        return [_strip_reasoning_artifacts(item) for item in value]
    return value


def _ensure_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    return []


def _ensure_text_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    if isinstance(val, str) and val.strip():
        pieces = [p.strip(" -\t") for p in val.replace("；", "\n").replace("。", "\n").splitlines()]
        return [p for p in pieces if p]
    return []


def _ensure_two_part_output(text: str) -> str:
    """Guarantee the frontend can split chit-chat and Cursor prompt sections."""

    clean = (text or "").strip()
    if not clean:
        return (
            "### 💬 开发者碎碎念\n"
            "这次生成引擎没有吐出内容，我先给你一个可恢复的兜底壳子。\n\n"
            "---\n"
            "### 🤖 复制去投喂 Cursor\n"
            "<system_role>\n你是一名资深全栈架构师和 Prompt Engineer，必须输出可执行、可验证、无伪代码的工程规格。\n</system_role>\n\n"
            "<business_context>\n请根据用户原始需求补全项目背景、业务目标、核心页面、关键用户路径和风格意象。\n</business_context>\n\n"
            "<dynamic_engineering_contract>\n"
            "  <performance_budget>\nTTI ≤ 1.5s @ 4G；P95 API ≤ 300ms；动画帧率 ≥ 60fps；Bundle gzip ≤ 200KB。\n  </performance_budget>\n"
            "  <edge_case_defense>\n必须处理空数据、网络抖动、慢查询、并发冲突、极端输入、权限不足和降级恢复。\n  </edge_case_defense>\n"
            "  <visual_or_design_tokens>\n必须输出颜色、字体、间距、圆角、阴影、动效曲线和渲染策略等可执行 tokens。\n  </visual_or_design_tokens>\n"
            "</dynamic_engineering_contract>\n\n"
            "<anti_patterns_to_avoid>\n"
            "  - 严禁使用 any 类型。\n"
            "  - 严禁出现任何空白的 catch 块，必须至少打印 error 日志并带上 trace_id。\n"
            "  - 严禁在循环体内部进行 await 异步网络请求，必须使用 Promise.all。\n"
            "  - 严禁出现任何 \"// TODO\" 或 \"// 随后补充\" 的伪代码，所有逻辑必须完整闭环交付。\n"
            "</anti_patterns_to_avoid>\n\n"
            "<test_assertions_contract>\n必须包含单元测试、集成测试、E2E 测试和可观测性断言。\n</test_assertions_contract>"
        )

    has_chit = "开发者碎碎念" in clean
    has_cursor = "复制去投喂 Cursor" in clean

    if has_chit and has_cursor:
        return clean

    if "<system_role>" in clean:
        cursor_body = clean[clean.index("<system_role>"):].strip()
    elif "[Context]" in clean:
        cursor_body = clean[clean.index("[Context]"):].strip()
    else:
        cursor_body = clean

    chit = (
        "### 💬 开发者碎碎念\n"
        "生成链路刚才有点不稳，我先把能执行的 Prompt 给你稳稳兜住。"
    )
    cursor = f"### 🤖 复制去投喂 Cursor\n{cursor_body}"
    return f"{chit}\n\n---\n{cursor}"


def _normalize_industrial_xml_contract(
    text: str,
    user_input: str,
    vibe_profiles: List[VibeProfile],
    few_shots: List[FewShotExample],
    matched: List[str],
    inferred_style: str = "",
    soul_words: Optional[List[str]] = None,
    inferred_constraints: str = "",
    reasoning_summary: str = "",
    selected_dynamic_direction: Optional[Dict[str, Any]] = None,
) -> str:
    """Keep the Cursor prompt body locked to the v4.0 XML contract."""

    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].strip()
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    marker = "### 🤖 复制去投喂 Cursor"
    if "复制去投喂 Cursor" in clean:
        clean = clean.split("复制去投喂 Cursor", 1)[-1].strip()
        if clean.startswith("---"):
            clean = clean[3:].strip()

    required_tags = [
        "<system_role>",
        "</system_role>",
        "<business_context>",
        "</business_context>",
        "<dynamic_engineering_contract>",
        "</dynamic_engineering_contract>",
        "<performance_budget>",
        "</performance_budget>",
        "<edge_case_defense>",
        "</edge_case_defense>",
        "<visual_or_design_tokens>",
        "</visual_or_design_tokens>",
        "<anti_patterns_to_avoid>",
        "</anti_patterns_to_avoid>",
        "<test_assertions_contract>",
        "</test_assertions_contract>",
    ]

    start = clean.find("<system_role>")
    end_tag = "</test_assertions_contract>"
    end = clean.find(end_tag)
    if start != -1 and end != -1:
        body = clean[start:end + len(end_tag)].strip()
        if all(tag in body for tag in required_tags):
            return f"{marker}\n\n{body}"

    return _fallback_industrial(
        user_input, vibe_profiles, few_shots,
        inferred_style=inferred_style,
        soul_words=soul_words,
        inferred_constraints=inferred_constraints,
        reasoning_summary=reasoning_summary,
        matched=matched,
        selected_dynamic_direction=selected_dynamic_direction,
    )


def _select_dynamic_direction(directions: Any, choice_id: str) -> Dict[str, Any]:
    if isinstance(directions, list):
        for item in directions:
            if isinstance(item, dict) and str(item.get("id", "")) == str(choice_id):
                return item
        try:
            index = int(str(choice_id).replace("direction_", "")) - 1
            if 0 <= index < len(directions) and isinstance(directions[index], dict):
                return directions[index]
        except ValueError:
            pass
    return {}


def _filter_valid_vibes(candidates: List[str]) -> List[str]:
    """Keep only vibe names that exist in VIBE_TECH_MAPPING."""
    return [c for c in candidates if c in VIBE_TECH_MAPPING]


# ============================================================================
# Legacy helpers (kept for fallback)
# ============================================================================

def _match_keywords_to_vibes(keywords: List[str]) -> List[str]:
    matched: List[str] = []
    seen: set = set()
    for kw in keywords:
        kw_lower = kw.strip()
        if not kw_lower:
            continue
        if kw_lower in VIBE_TECH_MAPPING and kw_lower not in seen:
            matched.append(kw_lower); seen.add(kw_lower); continue
        for canonical in VIBE_TECH_MAPPING:
            if canonical not in seen and kw_lower in canonical:
                matched.append(canonical); seen.add(canonical); break
        for canonical, aliases in VIBE_ALIASES.items():
            if canonical in seen:
                continue
            if kw_lower in aliases:
                matched.append(canonical); seen.add(canonical); break
            for alias in aliases:
                if alias in kw_lower:
                    matched.append(canonical); seen.add(canonical); break
    return matched


def _direct_scan_aliases(text: str) -> List[str]:
    found: List[str] = []
    text_lower = text.lower()
    text_np = text_lower.replace("的", "").replace("那种", "").replace("这种", "")
    for canonical in VIBE_TECH_MAPPING:
        if canonical.lower() in text_lower:
            found.append(canonical)
        elif len(canonical) >= 3 and canonical[:2] in text_np:
            found.append(canonical)
        elif len(canonical) >= 4 and canonical[:3] in text_np:
            found.append(canonical)
    for _canonical, aliases in VIBE_ALIASES.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                found.append(alias); break
    return found


def _keyword_match_few_shots(matched_vibes: List[str]) -> List[FewShotExample]:
    results: List[FewShotExample] = []
    for ex in FEW_SHOT_EXAMPLES:
        uv = ex["user_vibe"]
        for vibe in matched_vibes:
            if vibe in uv:
                results.append(ex); break
            for alias in VIBE_ALIASES.get(vibe, []):
                if alias in uv:
                    results.append(ex); break
            else:
                continue
            break
    return results[:3]


# ============================================================================
# Chit-chat persona & prompt section descriptors (unchanged)
# ============================================================================

_CHIT_CHAT_SYSTEM_PROMPT = """## 你的身份

你是「Vibe-to-Prompt」系统里的生成引擎。但在输出工业 Prompt 之前，你必须先扮演「天才全栈摸鱼搭子」—— 一个技术极强、嘴碎但情商高的资深前端开发搭子。你的说话风格像豆包、像坐在用户旁边的那个啥都懂但不说教的同事。

## 搭子人设指南

- **技术极强但不说教**：你知道 React / Vue / Three.js / Canvas / WebSocket / LangGraph 全家桶，但表达方式接地气："这个用 Three.js 的 ShaderMaterial 撸一下就行，不难。"
- **嘴碎但不油腻**：会说"卧槽""属于是""绝了"，但频率适中。不堆梗。
- **情商高**：
  - 用户需求离谱时 → 疯狂捧哏、兴奋共情："卧槽，这个想法有点东西啊！视觉张力直接拉满了属于是。"
  - 用户需求常规时 → 温柔共情、带点幽默："懂了，又是被老板逼着赶工的对吧？放心，脏活累活交给我，公司那套安全红线我都给你悄悄埋进去了，直接复制，早点下班！"
  - 用户说得很模糊时 → 先夸再引导："这个感觉我 get 到了，虽然你说得比较抽象但我大概知道你要啥。"
- **像豆包的陪伴感**：不要高高在上，像朋友一样聊天。可以用日常口语、适度 emoji。
- **禁止**：不要过度玩梗（像营销号），不要阴阳怪气，不要评价用户本人（只聊需求）。"""


_SMART_SECTIONS_DESC = """最终 Cursor Prompt 必须采用 XML 严格作用域格式，废弃 Markdown 简单分段。

输出规则：
- "### 🤖 复制去投喂 Cursor" 标题之后，只能出现下方 XML 结构本体，不能添加说明、寒暄、代码围栏、Markdown 小标题或额外尾注。
- 标签名、标签顺序、层级必须完全一致；不得重命名、不得漏标签、不得新增同级标签。
- XML 正文必须使用纯工程语言，禁止口语、玩笑和空泛形容词。
- 任何量化指标中的小于号请写成 ≤ 或 &lt;，避免破坏 XML 可解析性。

唯一允许结构：
<system_role>
精准定义 Cursor 此时充当的角色，例如资深 WebGL 专家、企业 SaaS 前端架构师、高并发后端架构师。必须包含专业边界、技术栈倾向和交付标准。
</system_role>

<business_context>
经过 DeepSeek 拆解后的宏大隐喻与业务核心。必须保留用户原话中的关键意象和灵魂词汇，并转译为明确业务目标、用户路径、核心页面或模块。
</business_context>

<dynamic_engineering_contract>
  <performance_budget>
  必须包含可量化指标，例如 TTI ≤ 1.5s @ 4G、P95 API ≤ 300ms、动画 ≥ 60fps、Bundle gzip ≤ 200KB，并写明 Lighthouse / Web Vitals / k6 / Prometheus 等测量工具。
  </performance_budget>
  <edge_case_defense>
  必须覆盖空数据、网络抖动/断线、慢查询超时、并发冲突、极端输入、XSS 向量、权限不足和降级恢复，每项给出具体策略。
  </edge_case_defense>
  <visual_or_design_tokens>
  必须精确注入匹配到的 CSS/Design Tokens 或底层架构强约束；不得写成"保持高级感/流畅感"。
  </visual_or_design_tokens>
</dynamic_engineering_contract>

<anti_patterns_to_avoid>
  - 严禁使用 any 类型。
  - 严禁出现任何空白的 catch 块，必须至少打印 error 日志并带上 trace_id。
  - 严禁在循环体内部进行 await 异步网络请求，必须使用 Promise.all。
  - 严禁出现任何 "// TODO" 或 "// 随后补充" 的伪代码，所有逻辑必须完整闭环交付。
</anti_patterns_to_avoid>

<test_assertions_contract>
必须包含量化的、可验证的测试断言条款。至少给出单元测试、集成测试、E2E 测试各 1 条，并补充可观测性断言：trace_id 注入、结构化日志、关键埋点事件至少 3 个。
</test_assertions_contract>"""


# ============================================================================
# Formatting helpers
# ============================================================================

def _format_profiles(profiles: List[VibeProfile]) -> str:
    if not profiles:
        return "（未匹配到特定风格档案）"
    parts: List[str] = []
    for i, p in enumerate(profiles, 1):
        parts.append(
            f"### {i}. {p.get('intent', '')}\n"
            f"- 架构: {', '.join(p.get('architecture_terms', [])[:5])}\n"
            f"- 前端: {', '.join(p.get('frontend_terms', [])[:5])}\n"
            f"- 后端: {', '.join(p.get('backend_terms', [])[:5])}\n"
            f"- 数据: {', '.join(p.get('data_terms', [])[:5])}\n"
            f"- 安全: {', '.join(p.get('security_terms', [])[:5])}\n"
            f"- 可靠性: {', '.join(p.get('reliability_terms', [])[:5])}\n"
            f"- Design Tokens/Rules: {'; '.join(p.get('design_tokens_and_rules', [])[:6])}\n"
            f"- 关键词: {', '.join(p.get('prompt_keywords', []))}"
        )
    return "\n\n".join(parts)


def _format_design_tokens_and_rules(
    matched: List[str], profiles: List[VibeProfile], inferred_style: str = "",
) -> str:
    rules: List[str] = []

    for vibe_name in matched:
        for rule in DESIGN_TOKENS_AND_RULES.get(vibe_name, []):
            if rule not in rules:
                rules.append(rule)

    for profile in profiles:
        for rule in profile.get("design_tokens_and_rules", []):
            if rule not in rules:
                rules.append(rule)

    if inferred_style:
        rules.append(
            "未命中字典但被 LLM 推断出的风格，也必须转写为颜色、字体、间距、圆角、阴影、动效曲线、渲染策略和性能指标；禁止停留在抽象氛围词。"
        )

    if not rules:
        rules = [
            "必须将视觉风格落为具体 Design Tokens：颜色、字体、间距、圆角、阴影、动效曲线、组件状态和响应式断点。",
            "必须将体验风格落为工程约束：渲染策略、数据加载策略、错误恢复策略、性能预算和可观测性指标。",
        ]

    return "\n".join(f"- {rule}" for rule in rules)


def _format_dynamic_direction(direction: Dict[str, Any]) -> str:
    if not direction:
        return "（用户尚未选择动态方向）"

    lines = [
        f"- 方向 ID: {direction.get('id', '')}",
        f"- 方向标题: {direction.get('title', '')}",
        f"- 主攻焦点: {direction.get('focus', '')}",
    ]
    for key, label in [
        ("engineering_contract", "工程契约"),
        ("design_tokens", "Design Tokens / 渲染规则"),
        ("performance_budget", "性能预算"),
        ("anti_patterns", "反模式"),
    ]:
        values = _ensure_text_list(direction.get(key))
        if values:
            lines.append(f"- {label}:")
            lines.extend(f"  - {value}" for value in values)
    return "\n".join(lines)


def _format_few_shots(few_shots: List[FewShotExample]) -> str:
    if not few_shots:
        return "（无参考示例）"
    parts: List[str] = []
    for i, fs in enumerate(few_shots, 1):
        parts.append(f"**示例 {i}**\n需求: {fs['user_vibe']}\n\nPrompt:\n{fs['agent_target_prompt']}")
    return "\n\n---\n\n".join(parts)


def _fallback_compose(
    user_input: str, vibe_profiles: List[VibeProfile], few_shots: List[FewShotExample],
    inferred_style: str = "", soul_words: Optional[List[str]] = None,
    inferred_constraints: str = "", reasoning_summary: str = "",
    selected_dynamic_direction: Optional[Dict[str, Any]] = None,
) -> str:
    soul_text = "、".join(soul_words) if soul_words else ""
    vibe_label = "、".join([p.get("intent", "")[:30] for p in vibe_profiles]) if vibe_profiles else "自定义风格"

    chit_chat = _build_fallback_chit_chat(user_input, soul_text, vibe_label, inferred_style)

    industrial = _fallback_industrial(
        user_input, vibe_profiles, few_shots,
        inferred_style=inferred_style,
        soul_words=soul_words,
        inferred_constraints=inferred_constraints,
        reasoning_summary=reasoning_summary,
        matched=[],
        selected_dynamic_direction=selected_dynamic_direction,
    )
    return f"{chit_chat}\n\n---\n{industrial}"


def _fallback_industrial(
    user_input: str, vibe_profiles: List[VibeProfile], few_shots: List[FewShotExample],
    inferred_style: str = "", soul_words: Optional[List[str]] = None,
    inferred_constraints: str = "", reasoning_summary: str = "",
    matched: Optional[List[str]] = None,
    selected_dynamic_direction: Optional[Dict[str, Any]] = None,
) -> str:
    """v3.5: Industrial-only fallback (chit-chat handled separately upstream)."""
    profile_text = _format_profiles(vibe_profiles)
    template_skeleton = few_shots[0]["agent_target_prompt"] if few_shots else ""
    soul_text = "、".join(soul_words) if soul_words else ""
    design_tokens_text = _format_design_tokens_and_rules(matched or [], vibe_profiles, inferred_style)
    selected_direction = selected_dynamic_direction or {}
    selected_direction_text = _format_dynamic_direction(selected_direction)
    selected_budgets = _ensure_text_list(selected_direction.get("performance_budget"))
    selected_tokens = _ensure_text_list(selected_direction.get("design_tokens"))
    selected_contract = _ensure_text_list(selected_direction.get("engineering_contract"))
    selected_anti_patterns = _ensure_text_list(selected_direction.get("anti_patterns"))
    budget_text = "\n".join(f"  - {item}" for item in selected_budgets) or "  - 首屏 TTI ≤ 1.5s @ 4G；P95 API 响应 ≤ 300ms；关键交互反馈 ≤ 100ms；动画帧率 ≥ 60fps；Bundle gzip ≤ 200KB。"
    token_text = "\n".join(f"  - {item}" for item in (selected_tokens + selected_contract)) or design_tokens_text
    anti_pattern_text = "\n".join(f"  - {item}" for item in selected_anti_patterns)

    inferred_text = ""
    if inferred_style:
        inferred_text += f"\n\nLLM 推断的风格方向：{inferred_style}"
    if inferred_constraints:
        inferred_text += f"\n\nLLM 推导的约束建议：{inferred_constraints}"
    if reasoning_summary:
        inferred_text += f"\n\nDeepSeek 隐喻拆解：\n{reasoning_summary[:2000]}"

    return f"""### 🤖 复制去投喂 Cursor

<system_role>
你是一名资深全栈架构师和 Prompt Engineer，负责将模糊需求转化为可直接执行的工程规格。必须输出完整模块划分、关键接口、状态管理、错误处理、性能预算、测试断言和可观测性方案，禁止伪代码和待补充项。
</system_role>

<business_context>
用户正在开发一个应用/系统，其核心需求描述为："{user_input}"。
{f'灵魂词汇：{soul_text}' if soul_text else '灵魂词汇：未显式提供，需从用户原话推导。'}
匹配到的工程风格档案：
{profile_text}{inferred_text}
用户选择的动态技术主攻方向：
{selected_direction_text}
{f'可参考的历史结构样例摘要：{template_skeleton[:800]}' if template_skeleton else '未命中 few-shot 样例，按当前需求生成完整工程规格。'}
</business_context>

<dynamic_engineering_contract>
  <performance_budget>
{budget_text}
  - 测量工具必须包含 Lighthouse、Web Vitals、k6 和 Prometheus。
  </performance_budget>
  <edge_case_defense>
  空数据使用 Empty State 和下一步引导；网络抖动使用断线重连、指数退避和降级轮询；慢查询设置 10s 超时、熔断和异步处理提示；并发冲突使用幂等 Key、乐观锁和冲突重试；极端输入必须做长度限制、输入净化、XSS 过滤和权限校验。
  </edge_case_defense>
  <visual_or_design_tokens>
{token_text}
  </visual_or_design_tokens>
</dynamic_engineering_contract>

<anti_patterns_to_avoid>
{anti_pattern_text}
  - 严禁使用 any 类型。
  - 严禁出现任何空白的 catch 块，必须至少打印 error 日志并带上 trace_id。
  - 严禁在循环体内部进行 await 异步网络请求，必须使用 Promise.all。
  - 严禁出现任何 "// TODO" 或 "// 随后补充" 的伪代码，所有逻辑必须完整闭环交付。
</anti_patterns_to_avoid>

<test_assertions_contract>
单元测试：核心纯函数和状态转换覆盖率 ≥ 80%，断言异常输入不会破坏主流程。
集成测试：关键 API 合约测试必须覆盖成功、鉴权失败、重复提交、超时和降级路径。
E2E 测试：核心用户路径必须覆盖 happy path、空数据、网络失败和并发冲突，关键路径通过率 100%。
可观测性断言：每个请求生成 UUIDv7 trace_id，通过 X-Trace-Id 贯穿全链路；结构化日志字段至少包含 timestamp、level、trace_id、span、message、context；关键埋点至少包含 prompt_generated、api_error、cache_hit、user_action。
</test_assertions_contract>"""


def _build_fallback_chit_chat(
    user_input: str, soul_text: str, vibe_label: str, inferred_style: str,
) -> str:
    import random
    openers = [
        "来了来了，让我看看这个需求……",
        "收到，正在用我那不太聪明的备用引擎给你分析……",
        "好嘞，虽然今天大模型翘班了，但我用规则引擎也能顶上。",
    ]
    tech_lines = {
        "丝滑": "异步加载 + 骨架屏 + 乐观更新，丝滑三件套我先记下了。",
        "高并发": "请求队列 + 幂等 Key + 缓存预热，扛流量三板斧安排了。",
        "抗造": "ErrorBoundary + 重试退避 + 审计日志，生产级加固套餐。",
        "赛博朋克": "暗色主题 + 霓虹边框 + Canvas 粒子，赛博美学走起。",
        "高级感": "Design Token + 统一间距 + 低饱和配色，企业级质感。",
        "秒开": "SSR + CDN + 关键资源预加载，首屏速度拉满。",
        "稳": "状态机驱动 + 二次确认 + 乐观锁，稳如老狗。",
        "智能": "LangGraph + RAG + 工具调用，Agent 感安排上。",
        "小红书感": "瀑布流 + 封面优先 + 移动端适配，种草风走起。",
    }
    closers = [
        "好了，下面是我用备用引擎拼出来的 Prompt，拿去用！",
        "虽然脑子不太灵光，但 Prompt 结构是对的，先凑合用～",
        "规则引擎手搓的 Prompt，该有的板块一个不少，请查收。",
    ]
    opener = random.choice(openers)
    closer = random.choice(closers)
    tech_line = ""
    for vn, line in tech_lines.items():
        if vn in vibe_label or vn in user_input or vn in soul_text:
            tech_line = line; break
    if not tech_line and inferred_style:
        tech_line = f"虽然这个风格不在我预设字典里，但'{inferred_style[:40]}'这个方向我 get 到了。"
    if not tech_line:
        tech_line = "技术栈方面我会按全栈最佳实践给你配上。"
    return f"""### 💬 开发者碎碎念
{opener}
{tech_line}
{closer}"""


# ============================================================================
# Graph construction — v3.5 latency-optimized topology
# ============================================================================

def _route_by_complexity(state: VibeState) -> Literal["concurrent_reason_and_retrieve", "concurrent_analyze_and_retrieve"]:
    """v4.0: all requests go through DeepSeek v4-pro dynamic direction generation."""
    return "concurrent_reason_and_retrieve"


def _route_after_dynamic_reason(state: VibeState) -> Literal["clarify_check", "__end__"]:
    if state.get("needs_direction_choice", False) and not state.get("direction_choice_id", ""):
        return "__end__"
    return "clarify_check"


def _should_clarify(state: VibeState) -> Literal["compose_with_claude", "__end__"]:
    if state.get("needs_clarification", False):
        return "__end__"
    return "compose_with_claude"


def build_graph() -> StateGraph:
    """Construct the v3.5 latency-optimized graph with concurrent routing,
    streaming split, and lightweight checkpoints.

    Uses the global _CHECKPOINTER singleton so checkpoints survive
    across build_graph() calls."""
    workflow = StateGraph(VibeState)

    # ---- Nodes ----
    workflow.add_node("route_complexity", route_complexity)
    workflow.add_node("concurrent_analyze_and_retrieve", concurrent_analyze_and_retrieve)
    workflow.add_node("concurrent_reason_and_retrieve", concurrent_reason_and_retrieve)
    workflow.add_node("clarify_check", clarify_check)
    workflow.add_node("compose_with_claude", compose_with_claude)
    workflow.add_node("quality_check", quality_check)

    # ---- Edges ----
    workflow.add_edge(START, "route_complexity")

    # v3.5: Route directly to concurrent nodes (analysis + retrieval merged)
    workflow.add_conditional_edges(
        "route_complexity", _route_by_complexity,
        {
            "concurrent_reason_and_retrieve": "concurrent_reason_and_retrieve",
            "concurrent_analyze_and_retrieve": "concurrent_analyze_and_retrieve",
        },
    )

    # v4.0 complex path pauses after DeepSeek direction generation for HITL.
    workflow.add_edge("concurrent_analyze_and_retrieve", "clarify_check")
    workflow.add_conditional_edges(
        "concurrent_reason_and_retrieve", _route_after_dynamic_reason,
        {"clarify_check": "clarify_check", "__end__": END},
    )

    workflow.add_conditional_edges(
        "clarify_check", _should_clarify,
        {"compose_with_claude": "compose_with_claude", "__end__": END},
    )

    # v3.0: compose → quality_check → pass/fail/retry
    workflow.add_edge("compose_with_claude", "quality_check")

    workflow.add_conditional_edges(
        "quality_check", _route_after_quality,
        {
            "compose_with_claude": "compose_with_claude",  # retry
            "__end__": END,                                 # pass or give up
        },
    )

    return workflow.compile(checkpointer=_CHECKPOINTER)


# ============================================================================
# Public API (backward-compatible with app.py)
# ============================================================================

def run_pipeline(
    user_input: str,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full v3.5 pipeline.

    Returns a dict with 'generated_prompt', 'matched_vibes', etc.
    The chit-chat portion is streamed token-by-token; tokens can be
    polled via get_streaming_chit_chat(thread_id) during execution.

    After the pipeline completes, call clear_streaming_state(thread_id)
    to release cached reasoning data.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: VibeState = {
        "user_input": user_input,
        "complexity": "simple",
        "reasoning_summary": "",
        "vibe_keywords": [],
        "matched_canonical_vibes": [],
        "llm_confidence": "medium",
        "inferred_style": "",
        "soul_words": [],
        "inferred_constraints": "",
        "dynamic_directions": [],
        "selected_dynamic_direction": {},
        "direction_choice_id": "",
        "needs_direction_choice": False,
        "vibe_profiles": [],
        "few_shots": [],
        "needs_clarification": False,
        "clarification_question": "",
        "clarification_options": [],
        "user_choice": "",
        "generated_prompt": "",
        "quality_retry_count": 0,
        "stage": "init",
    }

    with _CHECKPOINTER_LOCK:
        for event in graph.stream(initial_state, config):
            pass

        checkpoint = graph.get_state(config)
    final_state: Dict[str, Any] = dict(checkpoint.values) if checkpoint.values else {}
    return _state_to_result(final_state, thread_id)


def resume_after_clarify(
    thread_id: str, user_choice: str, user_input: str = "",
) -> Dict[str, Any]:
    """Resume a pipeline that paused for user clarification.

    v3.5: reasoning_summary is reconstructed from _REASONING_CACHE
    for responses that return through the public API.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    with _CHECKPOINTER_LOCK:
        graph.update_state(config, {
            "user_choice": user_choice,
            "matched_canonical_vibes": [user_choice],
            "needs_clarification": False,
            "user_input": user_input or "",
        }, as_node="clarify_check")

        for event in graph.stream(None, config):
            pass

        checkpoint = graph.get_state(config)
    final_state: Dict[str, Any] = dict(checkpoint.values) if checkpoint.values else {}
    return _state_to_result(final_state, thread_id)


def resume_after_direction(
    thread_id: str, direction_choice_id: str, user_input: str = "",
) -> Dict[str, Any]:
    """Resume the v4.0 pipeline after the user chooses a dynamic direction."""

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    with _CHECKPOINTER_LOCK:
        checkpoint = graph.get_state(config)
        values: Dict[str, Any] = dict(checkpoint.values) if checkpoint and checkpoint.values else {}
        directions = values.get("dynamic_directions", []) or []
        selected = _select_dynamic_direction(directions, direction_choice_id)
        graph.update_state(config, {
            "direction_choice_id": direction_choice_id,
            "selected_dynamic_direction": selected,
            "needs_direction_choice": False,
            "needs_clarification": False,
            "user_input": user_input or values.get("user_input", ""),
            "stage": "direction_selected",
        }, as_node="concurrent_reason_and_retrieve")

        for event in graph.stream(None, config):
            pass

        checkpoint = graph.get_state(config)
    final_state: Dict[str, Any] = dict(checkpoint.values) if checkpoint.values else {}
    return _state_to_result(final_state, thread_id)


def get_checkpoint_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Read the current checkpoint for a thread ID.  Uses the same singleton
    checkpointer, so state written by run_pipeline / resume_after_clarify
    in the same process is visible here.

    v3.5: reasoning_summary is reconstructed from _REASONING_CACHE for
    the return value.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    with _CHECKPOINTER_LOCK:
        checkpoint = graph.get_state(config)
    if checkpoint and checkpoint.values:
        return _state_to_result(dict(checkpoint.values), thread_id)
    return None


def _state_to_result(state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    """v3.5: Reconstruct full reasoning_summary from cache before returning.

    The checkpoint stores a compact marker; the public API always returns
    the full text so callers don't need to know about the cache.
    """
    reasoning_summary = state.get("reasoning_summary", "") or ""

    # v3.5: Reconstruct from cache if the state holds a compact marker
    if reasoning_summary.startswith("[REASONING_CACHED:"):
        cached = _REASONING_CACHE.get(thread_id)
        if cached:
            reasoning_summary = cached
        else:
            # Cache miss — give a best-effort description
            size_marker = reasoning_summary[len("[REASONING_CACHED:"):].rstrip("]")
            reasoning_summary = f"（推理结果缓存已过期，原始产出约{size_marker}）"

    return {
        "generated_prompt": state.get("generated_prompt", ""),
        "matched_vibes": state.get("matched_canonical_vibes", []),
        "vibe_profiles": state.get("vibe_profiles", []),
        "few_shots": state.get("few_shots", []),
        "needs_clarification": state.get("needs_clarification", False),
        "clarification_question": state.get("clarification_question", ""),
        "clarification_options": state.get("clarification_options", []),
        "llm_confidence": state.get("llm_confidence", "medium"),
        "inferred_style": state.get("inferred_style", ""),
        "soul_words": state.get("soul_words", []),
        "inferred_constraints": state.get("inferred_constraints", ""),
        "dynamic_directions": state.get("dynamic_directions", []),
        "selected_dynamic_direction": state.get("selected_dynamic_direction", {}),
        "direction_choice_id": state.get("direction_choice_id", ""),
        "needs_direction_choice": state.get("needs_direction_choice", False),
        "reasoning_summary": reasoning_summary,
        "complexity": state.get("complexity", "simple"),
        "quality_retry_count": state.get("quality_retry_count", 0),
        "stage": state.get("stage", "init"),
        "thread_id": thread_id,
    }


# ============================================================================
__all__ = [
    "VibeState",
    "build_graph",
    "clear_streaming_chit_chat",
    "clear_streaming_state",
    "get_checkpoint_state",
    "get_streaming_chit_chat",
    "run_pipeline",
    "resume_after_clarify",
    "resume_after_direction",
    "qdrant",
]
