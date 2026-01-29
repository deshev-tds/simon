import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rlm_gate import GateContext, RLMGatekeeper


def test_rlm_gate_explicit_intent_triggers():
    ctx = GateContext(
        user_query="Please do a deep dive on Project X",
        session_tokens=100,
        window_tokens=100,
        vector_dists=[],
        fts_hit_count=0,
        recent_history=[],
        vector_enabled=False,
    )
    decision = RLMGatekeeper.evaluate(ctx)
    assert decision.trigger is True
    assert decision.reason == "explicit_intent"


def test_rlm_gate_recent_window_short_circuit():
    ctx = GateContext(
        user_query="What was the budget figure?",
        session_tokens=500,
        window_tokens=200,
        vector_dists=[],
        fts_hit_count=0,
        recent_history=[
            {"role": "user", "content": "We discussed the budget figure for Project X yesterday."},
            {"role": "assistant", "content": "The budget figure was 2.4M USD."},
        ],
        vector_enabled=False,
    )
    decision = RLMGatekeeper.evaluate(ctx)
    assert decision.trigger is False
    assert decision.reason == "likely_in_recent_window"


def test_rlm_gate_high_debt_and_weak_retrieval():
    ctx = GateContext(
        user_query="What did we say earlier about the audit timeline?",
        session_tokens=2500,
        window_tokens=800,
        vector_dists=[0.9],
        fts_hit_count=0,
        recent_history=[],
        vector_enabled=True,
    )
    decision = RLMGatekeeper.evaluate(ctx)
    assert decision.trigger is True
    assert decision.reason in {"high_debt_override", "complex_intent_with_context_gap"}
