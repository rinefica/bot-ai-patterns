"""
День 28: RAG-система с локальной LLM + сравнение с облаком.

Режимы:
  L — Local (qwen2.5:7b через Ollama)
  C — Cloud (DeepSeek)
  B — Both (сравнение)

Запуск:
  uv run python scripts/day28_local_rag.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_ai_patterns.local_llm import LocalLLMClient
from bot_ai_patterns.rag.local_agent import LocalRagAgent
from bot_ai_patterns.rag.agent import RagAgent, RagResponse

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
RAG_MODE = "rag"  # plain / rag / rerank / full


def make_cloud_agent() -> RagAgent | None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return None
    try:
        return RagAgent(INDEX_DIR)
    except Exception:
        return None


def make_local_agent() -> LocalRagAgent | None:
    if not LocalLLMClient.is_available():
        print("  Ollama недоступен. Запустите: ollama serve")
        return None
    return LocalRagAgent(INDEX_DIR)


def run(agent: RagAgent | LocalRagAgent, question: str, label: str) -> RagResponse:
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"Вопрос: {question}")
    print("-" * 60)
    start = time.perf_counter()
    resp = agent.ask_rag(question)
    elapsed = time.perf_counter() - start
    print(f"Ответ:\n{resp.answer}")
    print(f"\nИсточников: {len(resp.sources)} | Токены: {resp.prompt_tokens}+{resp.completion_tokens} | Время: {elapsed:.1f}s")
    return resp


def compare(local: LocalRagAgent, cloud: RagAgent, question: str) -> None:
    print(f"\n{'#'*60}")
    print(f"СРАВНЕНИЕ: {question}")
    print(f"{'#'*60}")

    start = time.perf_counter()
    resp_local = local.ask_rag(question)
    t_local = time.perf_counter() - start

    start = time.perf_counter()
    resp_cloud = cloud.ask_rag(question)
    t_cloud = time.perf_counter() - start

    print(f"\n{'='*60}")
    print("[LOCAL — qwen2.5:7b]")
    print("-" * 60)
    print(resp_local.answer)
    print(f"\nВремя: {t_local:.1f}s | Токены: {resp_local.prompt_tokens}+{resp_local.completion_tokens}")

    print(f"\n{'='*60}")
    print("[CLOUD — DeepSeek]")
    print("-" * 60)
    print(resp_cloud.answer)
    print(f"\nВремя: {t_cloud:.1f}s | Токены: {resp_cloud.prompt_tokens}+{resp_cloud.completion_tokens}")

    print(f"\n{'='*60}")
    print("ИТОГ")
    print("-" * 60)
    faster = "local" if t_local < t_cloud else "cloud"
    ratio = max(t_local, t_cloud) / min(t_local, t_cloud)
    print(f"  Скорость:     {faster} быстрее в {ratio:.1f}x")
    print(f"  Local:        {t_local:.1f}s")
    print(f"  Cloud:        {t_cloud:.1f}s")
    print(f"  Источников:   local={len(resp_local.sources)} cloud={len(resp_cloud.sources)}")


def choose_mode(has_cloud: bool) -> str:
    opts = "[L]ocal"
    if has_cloud:
        opts += " / [C]loud / [B]oth"
    print(f"\nРежим: {opts}  (по умолчанию: L)")
    choice = input("Режим > ").strip().lower() or "l"
    if choice in ("l", "local"):
        return "local"
    if choice in ("c", "cloud") and has_cloud:
        return "cloud"
    if choice in ("b", "both") and has_cloud:
        return "both"
    return "local"


def main() -> None:
    print("День 28: RAG + локальная LLM")
    print(f"Индекс: {INDEX_DIR}")

    print("\nЗагрузка агентов...")
    local = make_local_agent()
    cloud = make_cloud_agent()

    if local is None:
        print("Локальный агент недоступен. Завершение.")
        return

    print(f"  Local:  qwen2.5:7b (Ollama)")
    print(f"  Cloud:  {'DeepSeek' if cloud else 'недоступен (нет DEEPSEEK_API_KEY)'}")

    mode = choose_mode(has_cloud=cloud is not None)

    print("\nЗадайте вопрос по выкройкам (или 'выход'):\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not question:
            continue
        if question.lower() in {"выход", "exit", "quit", "q"}:
            print("Выход.")
            break

        if mode == "local":
            run(local, question, "LOCAL — qwen2.5:7b")
        elif mode == "cloud" and cloud:
            run(cloud, question, "CLOUD — DeepSeek")
        elif mode == "both" and cloud:
            compare(local, cloud, question)
        else:
            run(local, question, "LOCAL — qwen2.5:7b")

        print()


if __name__ == "__main__":
    main()
