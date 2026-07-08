"""
Vibe-to-Prompt configuration.

This module contains:
1. A vibe mapping dictionary that translates fuzzy product/design vibes into
   concrete architecture, frontend, backend, security, and reliability terms.
2. A few-shot database for converting vague user language into structured
   industrial-grade prompts.

The data is intentionally dependency-free so it can be imported by Streamlit,
LangGraph nodes, retrieval tools, or tests without side effects.
"""

from __future__ import annotations

from typing import Dict, List, NotRequired, TypedDict


class VibeProfile(TypedDict):
    """Structured mapping from a vibe word to engineering implementation hints."""

    intent: str
    architecture_terms: List[str]
    frontend_terms: List[str]
    backend_terms: List[str]
    data_terms: List[str]
    security_terms: List[str]
    reliability_terms: List[str]
    design_tokens_and_rules: NotRequired[List[str]]
    prompt_keywords: List[str]


class FewShotExample(TypedDict):
    """Few-shot example used by the prompt generation agent."""

    user_vibe: str
    agent_target_prompt: str


VIBE_TECH_MAPPING: Dict[str, VibeProfile] = {
    "丝滑": {
        "intent": "强调低等待、低卡顿、即时反馈和顺畅交互。",
        "architecture_terms": [
            "SSE 流式响应",
            "WebSocket 实时通道",
            "异步任务队列",
            "渐进式渲染",
            "边生成边展示",
        ],
        "frontend_terms": [
            "Skeleton Loading",
            "Optimistic UI",
            "防抖节流",
            "虚拟列表",
            "懒加载",
            "请求取消 AbortController",
            "状态局部更新",
        ],
        "backend_terms": [
            "异步 I/O",
            "连接池",
            "分页查询",
            "流式接口",
            "后台任务 worker",
        ],
        "data_terms": [
            "Redis 热点缓存",
            "查询索引优化",
            "增量数据加载",
        ],
        "security_terms": [
            "接口限流",
            "请求签名校验",
        ],
        "reliability_terms": [
            "超时控制",
            "重试退避",
            "Loading/Error/Empty 三态",
            "前端 ErrorBoundary",
        ],
        "design_tokens_and_rules": [
            "必须开启 Web Worker 离线处理数据流，禁止在主线程做高成本解析、排序、聚合。",
            "核心 UI 渲染必须引入虚拟列表（Virtual List），长列表 DOM 节点数保持在可视区域 + overscan 范围内。",
            "所有高频交互必须实现 0ms 乐观更新（Optimistic UI），失败时提供可恢复回滚与明确错误提示。",
            "动画与过渡仅使用 transform/opacity，目标 60fps；输入、滚动、拖拽链路不得被同步请求阻塞。",
        ],
        "prompt_keywords": [
            "low-latency",
            "streaming-first",
            "instant feedback",
            "non-blocking UX",
        ],
    },
    "高并发": {
        "intent": "强调吞吐、水平扩展、削峰填谷和热点数据治理。",
        "architecture_terms": [
            "水平扩展",
            "负载均衡",
            "无状态服务",
            "读写分离",
            "CQRS",
            "事件驱动架构",
        ],
        "frontend_terms": [
            "请求合并",
            "防抖节流",
            "客户端缓存",
            "分页与游标加载",
        ],
        "backend_terms": [
            "消息队列",
            "异步消费",
            "连接池",
            "限流熔断",
            "批处理写入",
            "幂等接口",
        ],
        "data_terms": [
            "Redis 缓存",
            "缓存预热",
            "热点 Key 拆分",
            "数据库索引",
            "分库分表",
            "最终一致性",
        ],
        "security_terms": [
            "IP/用户维度限流",
            "JWT 鉴权",
            "RBAC 权限控制",
        ],
        "reliability_terms": [
            "熔断降级",
            "排队机制",
            "重试退避",
            "死信队列",
            "容量压测",
            "Prometheus 指标监控",
        ],
        "prompt_keywords": [
            "high-throughput",
            "horizontally scalable",
            "backpressure-aware",
            "idempotent",
        ],
    },
    "抗造": {
        "intent": "强调容错、可恢复、边界处理和长期稳定运行。",
        "architecture_terms": [
            "分层架构",
            "故障隔离",
            "防腐层",
            "可观测性优先",
            "灰度发布",
        ],
        "frontend_terms": [
            "ErrorBoundary",
            "表单校验",
            "离线提示",
            "失败重试按钮",
            "降级 UI",
        ],
        "backend_terms": [
            "幂等设计",
            "事务边界",
            "重试退避",
            "超时控制",
            "熔断器",
            "补偿任务",
        ],
        "data_terms": [
            "数据备份",
            "迁移脚本回滚",
            "唯一约束",
            "审计日志",
            "软删除",
        ],
        "security_terms": [
            "JWT",
            "RBAC",
            "输入净化",
            "CSRF 防护",
            "敏感字段脱敏",
        ],
        "reliability_terms": [
            "结构化日志",
            "链路追踪",
            "健康检查",
            "告警规则",
            "单元测试",
            "集成测试",
        ],
        "prompt_keywords": [
            "fault-tolerant",
            "production-ready",
            "recoverable",
            "observable",
        ],
    },
    "赛博朋克": {
        "intent": "强调未来感、霓虹视觉、高对比界面和数据流动感。",
        "architecture_terms": [
            "实时数据面板",
            "事件流",
            "WebSocket",
            "流式日志展示",
        ],
        "frontend_terms": [
            "暗色主题",
            "霓虹描边",
            "高对比色",
            "网格背景",
            "玻璃拟态",
            "微动效",
            "HUD 式布局",
        ],
        "backend_terms": [
            "实时推送接口",
            "异步事件处理",
            "日志聚合",
        ],
        "data_terms": [
            "时序数据",
            "实时指标聚合",
            "Redis Pub/Sub",
        ],
        "security_terms": [
            "JWT 鉴权",
            "访问令牌刷新",
            "操作审计",
        ],
        "reliability_terms": [
            "断线重连",
            "心跳检测",
            "前端 ErrorBoundary",
            "实时通道降级轮询",
        ],
        "prompt_keywords": [
            "cyberpunk",
            "neon HUD",
            "dark futuristic UI",
            "real-time telemetry",
        ],
    },
    "高级感": {
        "intent": "强调克制、专业、可信、精致但不过度装饰。",
        "architecture_terms": [
            "设计系统",
            "组件化架构",
            "Design Token",
            "可访问性标准",
        ],
        "frontend_terms": [
            "统一间距体系",
            "清晰信息层级",
            "低饱和配色",
            "细腻 hover/focus 状态",
            "响应式布局",
            "可访问性 ARIA",
        ],
        "backend_terms": [
            "清晰 API 契约",
            "OpenAPI Schema",
            "统一错误码",
        ],
        "data_terms": [
            "字段语义标准化",
            "展示数据格式化",
            "空状态数据策略",
        ],
        "security_terms": [
            "最小权限原则",
            "敏感信息隐藏",
        ],
        "reliability_terms": [
            "一致的错误提示",
            "端到端测试",
            "视觉回归测试",
        ],
        "prompt_keywords": [
            "premium",
            "restrained",
            "polished enterprise UI",
            "design-system-driven",
        ],
    },
    "苹果风": {
        "intent": "强调苹果式极简、毛玻璃层次、克制动效和系统字体质感。",
        "architecture_terms": [
            "Design Token 驱动",
            "组件化架构",
            "响应式断点",
            "可访问性优先",
        ],
        "frontend_terms": [
            "font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif",
            "backdrop-filter: blur(20px)",
            "border: 1px solid rgba(255,255,255,0.1)",
            "cubic-bezier(0.25, 1, 0.5, 1)",
            "8pt spacing grid",
            "reduced-motion 适配",
        ],
        "backend_terms": [
            "清晰 API 契约",
            "BFF 聚合",
            "统一错误响应",
        ],
        "data_terms": [
            "展示字段格式化",
            "空状态数据策略",
            "客户端缓存",
        ],
        "security_terms": [
            "最小权限原则",
            "敏感信息默认隐藏",
        ],
        "reliability_terms": [
            "视觉回归测试",
            "可访问性扫描",
            "Core Web Vitals 预算",
        ],
        "design_tokens_and_rules": [
            "全局字体必须使用 font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif。",
            "半透明层必须使用 backdrop-filter: blur(20px)；玻璃层边框必须为 border: 1px solid rgba(255,255,255,0.1)。",
            "动效曲线统一使用 cubic-bezier(0.25, 1, 0.5, 1)，时长控制在 160ms-320ms。",
            "界面以极简信息层级、8pt 间距系统、柔和阴影和可访问 focus ring 为硬性视觉规则。",
        ],
        "prompt_keywords": [
            "apple-inspired",
            "minimal",
            "glassmorphism",
            "system typography",
        ],
    },
    "古典黑铁美学": {
        "intent": "强调古典黑铁、机械秩序、强结构网格和冷硬工业质感。",
        "architecture_terms": [
            "模块化机械面板",
            "状态机驱动交互",
            "可审计操作流",
            "强约束设计系统",
        ],
        "frontend_terms": [
            "颜色基调 #1A1A1A / #2C2D30",
            "border-radius: 0",
            "border: 2px solid #000",
            "粗线条网格",
            "等宽字体标签",
            "拒绝任何渐变色",
        ],
        "backend_terms": [
            "显式状态流转",
            "审计日志",
            "统一错误码",
        ],
        "data_terms": [
            "事件溯源",
            "不可变操作记录",
            "状态枚举约束",
        ],
        "security_terms": [
            "RBAC 权限墙",
            "危险操作二次确认",
            "审计追踪",
        ],
        "reliability_terms": [
            "故障隔离",
            "回滚策略",
            "契约测试",
        ],
        "design_tokens_and_rules": [
            "颜色基调必须锁定为 #1A1A1A、#2C2D30、#000000 与有限高亮色；拒绝任何渐变色。",
            "所有面板与按钮必须采用强刚性直角 border-radius: 0，禁止胶囊按钮和柔软圆角卡片。",
            "结构线必须使用粗线条网格 border: 2px solid #000，表现铸铁、铭牌、机械分隔感。",
            "交互动效必须短促、机械、可预测；禁止果冻弹性、漂浮光斑和轻飘装饰。",
        ],
        "prompt_keywords": [
            "black iron",
            "mechanical UI",
            "brutalist grid",
            "no gradients",
        ],
    },
    "秒开": {
        "intent": "强调首屏性能、缓存命中和关键路径压缩。",
        "architecture_terms": [
            "SSR/SSG",
            "CDN",
            "边缘缓存",
            "关键资源预加载",
        ],
        "frontend_terms": [
            "代码分割",
            "图片懒加载",
            "资源预取",
            "首屏 Skeleton",
            "减少阻塞脚本",
        ],
        "backend_terms": [
            "接口聚合",
            "BFF",
            "响应压缩",
            "缓存头 Cache-Control",
        ],
        "data_terms": [
            "Redis 缓存",
            "本地缓存",
            "预计算聚合结果",
        ],
        "security_terms": [
            "缓存隔离",
            "鉴权结果短期缓存",
        ],
        "reliability_terms": [
            "性能预算",
            "Core Web Vitals",
            "慢查询监控",
        ],
        "prompt_keywords": [
            "fast first paint",
            "cache-first",
            "performance budget",
            "instant load",
        ],
    },
    "稳": {
        "intent": "强调业务一致性、可预测行为和稳定交付。",
        "architecture_terms": [
            "领域分层",
            "显式状态机",
            "事务一致性",
            "版本化 API",
        ],
        "frontend_terms": [
            "受控表单",
            "状态机驱动 UI",
            "明确的确认弹窗",
            "不可逆操作二次确认",
        ],
        "backend_terms": [
            "幂等 Key",
            "事务脚本",
            "乐观锁",
            "统一异常处理",
        ],
        "data_terms": [
            "唯一索引",
            "外键约束",
            "审计日志",
            "数据校验规则",
        ],
        "security_terms": [
            "RBAC",
            "操作权限校验",
            "审计追踪",
        ],
        "reliability_terms": [
            "单元测试",
            "契约测试",
            "回归测试",
            "发布回滚方案",
        ],
        "prompt_keywords": [
            "stable",
            "predictable",
            "transaction-safe",
            "well-tested",
        ],
    },
    "智能": {
        "intent": "强调自动理解、上下文记忆、工具调用和可解释结果。",
        "architecture_terms": [
            "LangGraph 状态图",
            "Agentic Workflow",
            "RAG",
            "工具调用",
            "Human-in-the-loop",
        ],
        "frontend_terms": [
            "多轮对话 UI",
            "流式输出",
            "引用来源展示",
            "可编辑中间结果",
        ],
        "backend_terms": [
            "Prompt 模板管理",
            "工具路由",
            "模型降级策略",
            "上下文裁剪",
            "结构化输出解析",
        ],
        "data_terms": [
            "向量数据库",
            "Embedding",
            "语义检索",
            "会话记忆",
            "Few-Shot 示例库",
        ],
        "security_terms": [
            "Prompt Injection 防护",
            "工具权限白名单",
            "敏感信息过滤",
        ],
        "reliability_terms": [
            "输出 JSON Schema 校验",
            "失败重试",
            "模型调用日志",
            "质量评估集",
        ],
        "prompt_keywords": [
            "agentic",
            "context-aware",
            "tool-augmented",
            "schema-constrained",
        ],
    },
    "小红书感": {
        "intent": "强调轻量、精致、生活方式化表达和强可读性。",
        "architecture_terms": [
            "内容卡片流",
            "标签系统",
            "推荐排序",
            "草稿自动保存",
        ],
        "frontend_terms": [
            "瀑布流布局",
            "封面图优先",
            "标签胶囊",
            "轻量动效",
            "移动端优先",
        ],
        "backend_terms": [
            "内容审核队列",
            "图片压缩",
            "异步发布",
        ],
        "data_terms": [
            "标签索引",
            "用户行为埋点",
            "内容热度分",
        ],
        "security_terms": [
            "内容安全过滤",
            "用户鉴权",
            "防刷机制",
        ],
        "reliability_terms": [
            "草稿恢复",
            "上传失败重试",
            "图片 CDN 回源降级",
        ],
        "prompt_keywords": [
            "lifestyle content",
            "mobile-first",
            "visual-first",
            "lightweight editorial",
        ],
    },
}


_DEFAULT_DESIGN_TOKENS_AND_RULES: List[str] = [
    "必须将风格词落到可执行 Design Tokens：颜色、字体、间距、圆角、阴影、动效曲线和组件状态都要有明确值。",
    "禁止只写'高级/好看/现代/流畅'等抽象形容词；每条视觉要求必须绑定 CSS token、组件规则或可测指标。",
]


DESIGN_TOKENS_AND_RULES: Dict[str, List[str]] = {
    vibe: list(_DEFAULT_DESIGN_TOKENS_AND_RULES)
    for vibe in VIBE_TECH_MAPPING
}

DESIGN_TOKENS_AND_RULES.update({
    "丝滑": [
        "必须开启 Web Worker 离线处理数据流，禁止在主线程做高成本解析、排序、聚合。",
        "核心 UI 渲染必须引入虚拟列表（Virtual List），长列表 DOM 节点数保持在可视区域 + overscan 范围内。",
        "所有高频交互必须实现 0ms 乐观更新（Optimistic UI），失败时提供可恢复回滚与明确错误提示。",
        "动画与过渡仅使用 transform/opacity，目标 60fps；输入、滚动、拖拽链路不得被同步请求阻塞。",
    ],
    "高并发": [
        "提交链路必须使用幂等 Key、排队反馈和限流提示；高峰写入走消息队列削峰。",
        "前端不得无上限并发请求；必须实现请求合并、AbortController 取消和指数退避重试。",
    ],
    "抗造": [
        "所有异常路径必须有 Loading/Error/Empty/Retry 四态；禁止空白 catch 块。",
        "必须注入 trace_id、结构化日志、告警规则和回滚/补偿策略。",
    ],
    "赛博朋克": [
        "主背景使用 #05070A 或 #090B10，霓虹高亮限定 cyan/magenta/acid-green 三色以内。",
        "HUD 网格和霓虹描边必须服务于数据可读性；禁止大面积低对比发光文字。",
    ],
    "高级感": [
        "必须建立颜色、字体、间距、圆角、阴影、状态 token；主色低饱和，信息密度服务于高频操作。",
        "禁止营销式大 Hero、夸张渐变和装饰性卡片堆叠；后台/工具类界面优先扫描效率。",
    ],
    "苹果风": [
        "全局字体必须使用 font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif。",
        "半透明层必须使用 backdrop-filter: blur(20px)；玻璃层边框必须为 border: 1px solid rgba(255,255,255,0.1)。",
        "动效曲线统一使用 cubic-bezier(0.25, 1, 0.5, 1)，时长控制在 160ms-320ms。",
        "界面以极简信息层级、8pt 间距系统、柔和阴影和可访问 focus ring 为硬性视觉规则。",
    ],
    "古典黑铁美学": [
        "颜色基调必须锁定为 #1A1A1A、#2C2D30、#000000 与有限高亮色；拒绝任何渐变色。",
        "所有面板与按钮必须采用强刚性直角 border-radius: 0，禁止胶囊按钮和柔软圆角卡片。",
        "结构线必须使用粗线条网格 border: 2px solid #000，表现铸铁、铭牌、机械分隔感。",
        "交互动效必须短促、机械、可预测；禁止果冻弹性、漂浮光斑和轻飘装饰。",
    ],
    "秒开": [
        "首屏关键路径必须明确 SSR/SSG、CDN、preload/prefetch 和 bundle split 策略。",
        "图片、图表和非首屏模块必须懒加载；首屏 TTI、LCP、bundle gzip 大小必须量化。",
    ],
    "稳": [
        "关键业务状态必须用显式状态机建模；危险操作必须二次确认并写入审计日志。",
        "并发冲突必须使用乐观锁、幂等 Key 或事务边界处理。",
    ],
    "智能": [
        "LLM 输出必须使用 JSON Schema 或等价结构化校验；工具调用必须有白名单和权限边界。",
        "必须处理 Prompt Injection、上下文裁剪、模型降级和可解释引用来源。",
    ],
    "小红书感": [
        "移动端优先，封面图比例、卡片间距、标签胶囊和阅读节奏必须 token 化。",
        "图片上传链路必须有压缩、重试、进度反馈和 CDN 回源降级。",
    ],
})

for _vibe_name, _rules in DESIGN_TOKENS_AND_RULES.items():
    profile = VIBE_TECH_MAPPING.get(_vibe_name)
    if not profile:
        continue
    existing_rules = list(profile.get("design_tokens_and_rules", []))
    profile["design_tokens_and_rules"] = existing_rules + [
        rule for rule in _rules if rule not in existing_rules
    ]


FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    {
        "user_vibe": "我想做一个 AI 聊天工具，感觉要像打字机一样丝滑，用户不要干等，最好边想边出结果。",
        "agent_target_prompt": """[Context]
你正在开发一个基于 LangGraph + Streamlit 的 AI 聊天应用。用户输入问题后，系统需要调用大模型、展示中间状态，并将最终答案保存到会话历史。目标是降低等待感，并让用户明确知道 Agent 当前执行到哪一步。

[Persona]
你是一名资深 AI 应用架构师和全栈工程师，熟悉 LangGraph 状态机、Streamlit 交互模型、SSE/流式输出、异步任务编排和生产级错误处理。

[Vibe/Style]
整体体验必须“丝滑”：采用 streaming-first 的交互方式，边生成边展示；界面提供 Skeleton Loading、步骤状态、增量输出和可取消请求；避免整页刷新和长时间空白等待。

[Constraints]
- 使用 LangGraph 组织 Agent 节点，至少包含 intent_parse、tool_route、llm_generate、finalize 四个阶段。
- LLM 输出必须支持流式展示；如 Streamlit 原生能力不足，需要用占位容器增量刷新。
- 前端必须处理 Loading/Error/Empty 三态，并提供失败重试入口。
- 请求层加入超时控制、重试退避和结构化日志。
- 会话历史必须可追踪，每轮消息包含 role、content、timestamp、trace_id。
- 不要只给概念说明，必须输出可落地的模块划分、关键函数和 Prompt 模板。""",
    },
    {
        "user_vibe": "我要做一个活动报名系统，可能一下子很多人冲进来，别崩，要高并发，还得防止重复报名。",
        "agent_target_prompt": """[Context]
你正在设计一个活动报名系统，核心流程包括活动列表、报名提交、名额扣减、支付或确认、报名结果查询。系统需要应对短时间高峰流量，并保证同一用户不会重复报名，同一名额不会被超卖。

[Persona]
你是一名高并发后端架构师，熟悉 Redis、消息队列、数据库事务、幂等接口、限流熔断、缓存预热和压测方案。

[Vibe/Style]
整体方案必须体现“高并发”和“抗造”：优先削峰填谷，热点数据走缓存，关键写入保证幂等和一致性；用户侧反馈要明确，不能因为排队或限流而表现为无响应。

[Constraints]
- 活动详情和剩余名额使用 Redis 缓存，并设计缓存预热与过期策略。
- 报名提交接口必须使用幂等 Key，防止重复点击、网络重试和重复提交。
- 名额扣减必须说明 Redis 原子扣减、数据库最终落库和失败补偿策略。
- 高峰写入通过消息队列异步消费，必须包含死信队列和重试退避。
- 需要提供限流、熔断、排队提示和降级页面方案。
- 必须给出数据表核心字段、关键 API、状态流转和压测指标。""",
    },
    {
        "user_vibe": "帮我做一个管理后台，要看起来很高级，但不是花里胡哨，老板看了觉得专业，员工每天用也顺手。",
        "agent_target_prompt": """[Context]
你正在构建一个企业级管理后台，包含仪表盘、客户列表、订单处理、权限管理和操作审计。系统面向内部员工和管理层，要求长期高频使用，信息密度适中，操作路径清晰。

[Persona]
你是一名企业 SaaS 产品设计师兼前端架构师，熟悉设计系统、组件化 UI、RBAC 权限模型、表格交互、可访问性和可维护的前端工程结构。

[Vibe/Style]
整体体验要有“高级感”和“稳”：视觉克制、层级清楚、间距统一、颜色低饱和；交互强调可预测、可恢复和少打扰。避免营销风大 Hero、夸张渐变和装饰性卡片堆叠。

[Constraints]
- 使用设计系统思维定义颜色、字体、间距、圆角、阴影和状态 Token。
- 后台首页必须优先呈现关键指标、待办事项、异常提醒和最近操作。
- 表格必须支持搜索、筛选、排序、分页、批量操作和空状态。
- 权限控制采用 JWT + RBAC，前后端都要校验权限边界。
- 危险操作必须二次确认，并写入审计日志。
- 输出必须包含页面信息架构、组件清单、状态设计、权限设计和错误处理策略。""",
    },
    {
        "user_vibe": "我想做个赛博朋克风的数据大屏，实时刷数据，有那种未来城市控制台的感觉。",
        "agent_target_prompt": """[Context]
你正在实现一个实时数据监控大屏，用于展示业务指标、系统健康度、告警事件、地域分布和趋势变化。大屏运行在固定展示终端上，需要长时间稳定刷新。

[Persona]
你是一名实时可视化工程师，熟悉 WebSocket、时序数据、前端动效、Canvas/SVG 图表、监控指标聚合和大屏性能优化。

[Vibe/Style]
视觉风格必须是“赛博朋克”：暗色底、高对比、霓虹描边、HUD 式布局、网格背景和轻量数据流动效。动效服务于实时感，不能影响指标可读性。

[Constraints]
- 实时数据通过 WebSocket 推送，必须包含心跳检测、断线重连和轮询降级。
- 指标卡片需要区分正常、预警、严重三种状态，并提供清晰颜色规范。
- 图表刷新必须控制频率，避免高频重绘造成卡顿。
- 数据层需要说明时序数据结构、聚合窗口和异常值处理。
- 前端必须包含 ErrorBoundary、空数据占位和连接状态提示。
- 输出必须包含布局方案、组件结构、数据协议示例和性能优化清单。""",
    },
    {
        "user_vibe": "我要做一个能把用户大白话变成专业 Prompt 的工具，最好聪明一点，能判断用户到底想要什么风格。",
        "agent_target_prompt": """[Context]
你正在开发一个 Vibe-to-Prompt Agent。用户会输入模糊的大白话，例如“丝滑一点”“要抗造”“像小红书那种感觉”。系统需要识别氛围词、映射到工程术语，并生成可直接给 Claude Code 使用的结构化 Prompt。

[Persona]
你是一名 Prompt Engineer、LangGraph Agent 设计师和工程语料专家，擅长将模糊需求转化为可执行的软件规格、架构约束和实现提示。

[Vibe/Style]
Agent 必须表现得“智能”和“稳”：能识别显性氛围词与隐含意图，能补全缺失上下文，但不会过度脑补；输出风格专业、具体、可执行。

[Constraints]
- 使用 LangGraph 构建流程，至少包含 vibe_extract、term_mapping、prompt_compose、quality_check 四个节点。
- 氛围词必须通过 VIBE_TECH_MAPPING 映射到架构、前端、后端、数据、安全和可靠性术语。
- Few-Shot 示例必须参与生成过程，用于稳定输出格式和语气。
- 最终 Prompt 必须严格包含 [Context]、[Persona]、[Vibe/Style]、[Constraints] 四个板块。
- 对不明确需求要给出合理默认值，并在 Constraints 中标记假设。
- 输出必须可直接复制给 Claude Code 执行，避免空泛形容词。""",
    },
]


VIBE_ALIASES: Dict[str, List[str]] = {
    "丝滑": ["流畅", "顺滑", "不卡", "不卡顿", "无等待", "像打字机", "实时输出"],
    "高并发": ["很多人同时用", "扛流量", "秒杀", "峰值流量", "别崩"],
    "抗造": ["稳定", "健壮", "耐用", "容错", "生产级", "别炸"],
    "赛博朋克": ["霓虹", "未来感", "科技感", "HUD", "暗黑科技"],
    "高级感": ["专业", "克制", "精致", "企业级", "老板喜欢"],
    "苹果风": ["苹果风", "苹果式", "Apple 风", "iOS 感", "macOS 感", "极简", "极简主义", "毛玻璃"],
    "古典黑铁美学": ["古典黑铁", "黑铁", "机械感", "机械美学", "铸铁", "工业机械", "黑铁美学"],
    "秒开": ["快", "首屏快", "加载快", "即开即用"],
    "稳": ["可靠", "可预测", "不出错", "业务安全"],
    "智能": ["聪明", "自动判断", "能理解", "会补全", "Agent 感"],
    "小红书感": ["种草", "生活方式", "笔记感", "封面好看", "轻盈"],
}


DEFAULT_PROMPT_SECTIONS: List[str] = [
    "<system_role>",
    "<business_context>",
    "<dynamic_engineering_contract>",
    "<anti_patterns_to_avoid>",
    "<test_assertions_contract>",
]


def get_vibe_profile(vibe: str) -> VibeProfile:
    """
    Return a vibe profile by canonical vibe name or alias.

    Raises:
        KeyError: If the vibe cannot be matched.
    """

    normalized_vibe = vibe.strip()
    if normalized_vibe in VIBE_TECH_MAPPING:
        return VIBE_TECH_MAPPING[normalized_vibe]

    for canonical_vibe, aliases in VIBE_ALIASES.items():
        if normalized_vibe in aliases:
            return VIBE_TECH_MAPPING[canonical_vibe]

    raise KeyError(f"Unknown vibe: {vibe}")


__all__ = [
    "DEFAULT_PROMPT_SECTIONS",
    "DESIGN_TOKENS_AND_RULES",
    "FEW_SHOT_EXAMPLES",
    "VIBE_ALIASES",
    "VIBE_TECH_MAPPING",
    "FewShotExample",
    "VibeProfile",
    "get_vibe_profile",
]
