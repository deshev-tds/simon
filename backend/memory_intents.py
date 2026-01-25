import re

# Recall intent (deep archive).
ARCHIVE_EXPLICIT_PREFIXES = (
    "archive:",
    "memory:",
    "/archive",
    "/memory",
)

# Recall intent triggers (English + Bulgarian). Tune these lists as needed.
MEMORY_RECALL_PATTERNS_EN = [
    r"\b(chatgpt|gpt)\b.*\b(history|archive|conversation|conversations)\b",
    r"\b(past|previous|earlier|last)\b.*\b(chat|conversation|discussion|talk)\b",
    r"\b(do you remember|remember when)\b",
    r"\b(what did (we|i|you) (say|discuss|talk(?:ed)? about))\b",
    r"\b(you said|you told me|i told you)\b",
    r"\b(have we talked about|did we talk about|have we discussed|did we discuss)\b",
    r"\b(we talked about|we discussed)\b.*\b(before|earlier|last time|previously)\b",
]
MEMORY_RECALL_PATTERNS_BG = [
    r"\bпомниш ли\b",
    r"\bспомняш ли си\b",
    r"\bпомниш ли какво\b",
    r"\bговорихме ли\b",
    r"\bобсъждали ли сме\b",
    r"\bобсъждахме ли\b",
    r"\bказвал ли съм\b",
    r"\bказвал ли съм ти\b",
    r"\bказвал ли си\b",
    r"\bпреди сме говорили\b",
    r"\bдали сме говорили\b",
]

# Save intent triggers (English + Bulgarian). Tune these lists as needed.
MEMORY_SAVE_PATTERNS_EN = [
    r"\bremember this\b",
    r"\bremember that\b",
    r"\bsave this\b",
    r"\bstore this\b",
    r"\bkeep this in mind\b",
    r"\bmake a note\b",
    r"\bnote this down\b",
    r"\badd (this|that) to memory\b",
]
MEMORY_SAVE_PATTERNS_BG = [
    r"\bзапомни това\b",
    r"\bзапомни го\b",
    r"\bзапомни си\b",
    r"\bзапомни следното\b",
    r"\bпомни това\b",
    r"\bпомни го\b",
    r"\bзапиши това\b",
    r"\bзапиши го\b",
    r"\bзапиши си\b",
]

_RECALL_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in (MEMORY_RECALL_PATTERNS_EN + MEMORY_RECALL_PATTERNS_BG)
]
_SAVE_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in (MEMORY_SAVE_PATTERNS_EN + MEMORY_SAVE_PATTERNS_BG)
]


def detect_archive_recall(user_text: str):
    if not user_text:
        return False, False, ""
    raw_text = user_text.strip()
    lowered = raw_text.lower()
    for prefix in ARCHIVE_EXPLICIT_PREFIXES:
        if lowered.startswith(prefix):
            trimmed = raw_text[len(prefix):].strip()
            return True, True, trimmed or raw_text
    for pattern in _RECALL_PATTERNS:
        if pattern.search(lowered):
            return True, False, raw_text
    return False, False, raw_text


def detect_memory_save(user_text: str) -> bool:
    if not user_text:
        return False
    lowered = user_text.strip().lower()
    for pattern in _SAVE_PATTERNS:
        if pattern.search(lowered):
            return True
    return False


__all__ = [
    "ARCHIVE_EXPLICIT_PREFIXES",
    "MEMORY_RECALL_PATTERNS_EN",
    "MEMORY_RECALL_PATTERNS_BG",
    "MEMORY_SAVE_PATTERNS_EN",
    "MEMORY_SAVE_PATTERNS_BG",
    "detect_archive_recall",
    "detect_memory_save",
]
