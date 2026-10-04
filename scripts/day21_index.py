"""
Day 21 — Document Indexing Pipeline
====================================
Builds two FAISS indexes from sewing instruction PDFs,
compares chunking strategies, and runs demo queries.

Usage:
    uv run python scripts/day21_index.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

DOCS_DIR = Path("/Users/kate_yal/sewing_docs")
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


def build_indexes() -> dict:
    from bot_ai_patterns.indexer import IndexPipeline

    pipeline = IndexPipeline(
        docs_dir=DOCS_DIR,
        index_dir=INDEX_DIR,
        chunk_size=600,
        overlap=120,
    )
    return pipeline.run()


def compare_strategies(stores: dict) -> None:
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА")
    print("=" * 60)

    fixed_s = stores["fixed"].stats()
    struct_s = stores["structure"].stats()

    metrics = [
        ("total_chunks",   "Всего чанков"),
        ("total_chars",    "Всего символов"),
        ("avg_chars",      "Средний размер"),
        ("min_chars",      "Минимальный"),
        ("max_chars",      "Максимальный"),
        ("unique_sources", "Файлов-источников"),
    ]
    header = f"{'Метрика':<25} {'fixed':>12} {'structure':>12}"
    print(header)
    print("-" * len(header))
    for key, label in metrics:
        print(f"{label:<25} {str(fixed_s.get(key, '—')):>12} {str(struct_s.get(key, '—')):>12}")

    print()
    print("ВЫВОД:")
    print(
        "  Fixed-size: предсказуемый размер, равномерное покрытие,\n"
        "  но может разрезать шаг инструкции посередине.\n\n"
        "  Structure:  чанки = логические блоки (раздел/абзац),\n"
        "  лучше для поиска по конкретному шагу пошива."
    )


def demo_search(stores: dict) -> None:
    from bot_ai_patterns.indexer import Embedder

    embedder = Embedder()
    queries = [
        "обработка боковых швов",
        "втачивание рукава",
        "обработка пояса брюк",
        "разметка и раскрой",
        "обработка низа изделия",
    ]

    print("\n" + "=" * 60)
    print("ДЕМО-ПОИСК (top-3 на каждый запрос)")
    print("=" * 60)

    for query in queries:
        qvec = embedder.embed_query(query)
        print(f"\n🔍 «{query}»")
        for strategy_name, store in stores.items():
            results = store.search(qvec, top_k=3)
            print(f"  [{strategy_name}]")
            for r in results:
                snippet = r["text"][:150].replace("\n", " ").strip()
                print(f"    score={r['score']:.3f} | {r['title']} | section: {r['section'] or '—'}")
                print(f"    {textwrap.shorten(snippet, 110)}")


def load_indexes() -> dict:
    from bot_ai_patterns.indexer import IndexStore

    stores = {}
    for name in ("fixed", "structure"):
        store = IndexStore(INDEX_DIR)
        store.load(name)
        stores[name] = store
    return stores


if __name__ == "__main__":
    fixed_faiss = INDEX_DIR / "fixed.faiss"
    if fixed_faiss.exists():
        print("Индекс уже существует, загружаем…")
        stores = load_indexes()
    else:
        print("Строим индекс с нуля…")
        stores = build_indexes()

    compare_strategies(stores)
    demo_search(stores)
    print("\nГотово! Индексы сохранены в data/index/")
