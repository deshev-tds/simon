#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

SUBJECTS_EN = [
    "The agent", "The system", "The server", "The pipeline", "The module",
    "The service", "The assistant", "The process", "The client", "The router",
]
VERBS_EN = [
    "processes", "stores", "routes", "validates", "checks", "syncs",
    "indexes", "summarizes", "filters", "collects",
]
OBJECTS_EN = [
    "events", "logs", "requests", "context", "metadata", "signals",
    "documents", "sessions", "payloads", "metrics",
]
ADVERBS_EN = [
    "quickly", "carefully", "silently", "periodically", "reliably",
    "safely", "efficiently", "consistently", "thoroughly", "automatically",
]
TIMES_EN = [
    "Today", "This morning", "Tonight", "Yesterday", "During testing",
    "In production", "During deployment",
]
PLACES_EN = [
    "in the primary cluster", "on the local machine", "in the cache layer",
    "in the staging environment", "in the data pipeline", "in the core service",
]

SUBJECTS_BG = [
    "Агентът", "Системата", "Сървърът", "Пайплайнът", "Модулът",
    "Процесът", "Асистентът", "Клиентът", "Рутерът", "Услугата",
]
VERBS_BG = [
    "обработва", "запазва", "маршрутизира", "валидира", "проверява",
    "синхронизира", "индексира", "филтрира", "събира", "анализира",
]
OBJECTS_BG = [
    "събитията", "логовете", "заявките", "контекста", "метаданните",
    "сигналите", "документите", "сесиите", "пейлоудите", "метриките",
]
ADVERBS_BG = [
    "бързо", "внимателно", "тихо", "периодично", "надеждно",
    "безопасно", "ефективно", "последователно", "автоматично", "старателно",
]
TIMES_BG = [
    "Днес", "Сутринта", "Вечерта", "Вчера", "По време на тестове",
    "В продукция", "По време на деплой",
]
PLACES_BG = [
    "в основния клъстер", "на локалната машина", "в кеш слоя",
    "в staging средата", "в data pipeline-а", "в основната услуга",
]

TEMPLATES_EN = [
    "{subject} {verb} {object} {adverb}.",
    "{time}, {subject} {verb} {object} {place}.",
    "Report {idx}: {subject} {verb} {object} {adverb}.",
    "Note {idx}: {subject} {verb} {object} {place}.",
]
TEMPLATES_BG = [
    "{subject} {verb} {object} {adverb}.",
    "{time} {subject} {verb} {object} {place}.",
    "Доклад {idx}: {subject} {verb} {object} {adverb}.",
    "Бележка {idx}: {subject} {verb} {object} {place}.",
]


def _build_sentence(rng: random.Random, lang: str, idx: int) -> str:
    if lang == "bg":
        return rng.choice(TEMPLATES_BG).format(
            subject=rng.choice(SUBJECTS_BG),
            verb=rng.choice(VERBS_BG),
            object=rng.choice(OBJECTS_BG),
            adverb=rng.choice(ADVERBS_BG),
            time=rng.choice(TIMES_BG),
            place=rng.choice(PLACES_BG),
            idx=idx,
        )
    return rng.choice(TEMPLATES_EN).format(
        subject=rng.choice(SUBJECTS_EN),
        verb=rng.choice(VERBS_EN),
        object=rng.choice(OBJECTS_EN),
        adverb=rng.choice(ADVERBS_EN),
        time=rng.choice(TIMES_EN),
        place=rng.choice(PLACES_EN),
        idx=idx,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate semantically meaningful filler sentences for chat UI demos."
    )
    parser.add_argument("--count", type=int, default=50, help="Number of sentences to generate.")
    parser.add_argument(
        "--lang",
        type=str,
        default="both",
        choices=["en", "bg", "both"],
        help="Language of generated sentences.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for each line.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for each line.")
    parser.add_argument("--out", type=str, default="", help="Write output to a file instead of stdout.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    langs = ["en", "bg"] if args.lang == "both" else [args.lang]

    lines = []
    for i in range(1, args.count + 1):
        lang = rng.choice(langs)
        sentence = _build_sentence(rng, lang, i)
        lines.append(f"{args.prefix}{sentence}{args.suffix}")

    output = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
