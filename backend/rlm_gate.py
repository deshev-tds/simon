import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config import (
    AGENT_TRIGGER_KEYWORDS,
    RLM_ENABLED,
    RLM_MAX_DEBT_RATIO,
    RLM_MIN_DEBT_FOR_CHECK,
    RLM_MIN_FTS_HITS,
    RLM_MIN_QUERY_LEN,
    RLM_RECENT_WINDOW_TURNS,
    RLM_VECTOR_WEAK_DIST,
)


_RE_RECALL = re.compile(
    r"\b(last time|earlier|previous|remember|discussed|said|told|преди|помниш|каза)\b",
    re.IGNORECASE,
)
_RE_COMPLEX = re.compile(
    r"\b(connection|relation|difference|compare|summary|timeline|trace|сравни|връзка)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "if", "in", "into", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "you", "your", "we", "i", "me", "my", "our",
    "this", "that", "these", "those",
    "и", "в", "на", "за", "с", "по", "от", "какво", "как", "ти", "ние",
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"\w+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _likely_in_recent_window(user_query: str, recent_history: List[Dict[str, str]]) -> bool:
    if not user_query or not recent_history:
        return False
    query_tokens = set(_tokenize(user_query))
    if not query_tokens:
        return False
    overlap_needed = 1 if len(query_tokens) <= 1 else min(3, len(query_tokens))
    for msg in recent_history[-RLM_RECENT_WINDOW_TURNS:]:
        content = (msg.get("content") or "")
        msg_tokens = set(_tokenize(content))
        if len(query_tokens.intersection(msg_tokens)) >= overlap_needed:
            return True
    return False


@dataclass
class GateContext:
    user_query: str
    session_tokens: int
    window_tokens: int
    vector_dists: List[float]
    fts_hit_count: int
    recent_history: List[Dict[str, str]]
    vector_enabled: bool = True


@dataclass
class GateDecision:
    trigger: bool
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class RLMGatekeeper:
    @staticmethod
    def evaluate(ctx: GateContext) -> GateDecision:
        if not RLM_ENABLED:
            return GateDecision(False, "disabled", {})

        query = (ctx.user_query or "").strip()
        query_len = len(query)
        query_lower = query.lower()

        if any(k in query_lower for k in AGENT_TRIGGER_KEYWORDS):
            return GateDecision(True, "explicit_intent", {"query_len": query_len})

        if _likely_in_recent_window(query, ctx.recent_history):
            return GateDecision(False, "likely_in_recent_window", {"query_len": query_len})

        is_recall = bool(_RE_RECALL.search(query))
        is_complex = bool(_RE_COMPLEX.search(query))

        if query_len < RLM_MIN_QUERY_LEN and not is_recall and not is_complex:
            return GateDecision(False, "query_too_short", {"query_len": query_len})

        debt_ratio = ctx.session_tokens / max(1, ctx.window_tokens)

        top_dist: Optional[float] = None
        vector_weak = False
        if ctx.vector_enabled:
            if ctx.vector_dists:
                top_dist = min(ctx.vector_dists)
                vector_weak = top_dist > RLM_VECTOR_WEAK_DIST
            else:
                vector_weak = True

        fts_weak = ctx.fts_hit_count < RLM_MIN_FTS_HITS
        is_weak_retrieval = vector_weak or fts_weak

        metrics = {
            "debt": round(debt_ratio, 2),
            "top_dist": round(top_dist, 3) if top_dist is not None else None,
            "fts_hits": ctx.fts_hit_count,
            "query_len": query_len,
            "is_recall": is_recall,
            "is_complex": is_complex,
            "weak_retrieval": is_weak_retrieval,
        }

        if debt_ratio >= RLM_MAX_DEBT_RATIO and (is_recall or is_complex):
            return GateDecision(True, "high_debt_override", metrics)

        if (is_recall or is_complex) and (debt_ratio >= RLM_MIN_DEBT_FOR_CHECK or is_weak_retrieval):
            return GateDecision(True, "complex_intent_with_context_gap", metrics)

        if (
            debt_ratio >= RLM_MIN_DEBT_FOR_CHECK
            and is_weak_retrieval
            and ctx.fts_hit_count == 0
            and query_len > max(RLM_MIN_QUERY_LEN * 2, 40)
        ):
            return GateDecision(True, "retrieval_failure_fallback", metrics)

        return GateDecision(False, "base_rag_sufficient", metrics)
