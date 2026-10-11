"""
День 29: Запуск полного бенчмарка оптимизации локальной LLM.

Запуск:
  uv run python scripts/day29_benchmark.py          # все шаги
  uv run python scripts/day29_benchmark.py --params # только параметры
  uv run python scripts/day29_benchmark.py --prompts # только промпты
  uv run python scripts/day29_benchmark.py --quant  # только квантование
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
from bot_ai_patterns.day29_optimizer import (
    load_catalog,
    bench_params,
    bench_quant,
    bench_prompts,
    print_summary,
    OLLAMA_URL,
    TEST_QUESTIONS,
    PARAM_CONFIGS,
    PROMPT_OPTIMIZED,
    build_user_prompt,
    run_single,
)


def check_server() -> bool:
    try:
        with httpx.Client(timeout=5) as c:
            return c.get(f"{OLLAMA_URL}/api/tags").status_code == 200
    except httpx.ConnectError:
        return False


def main() -> None:
    args = sys.argv[1:]
    run_all = not args
    run_p = run_all or "--params" in args
    run_q = run_all or "--quant" in args
    run_pr = run_all or "--prompts" in args

    print("День 29: Оптимизация локальной LLM")
    print(f"API: {OLLAMA_URL} | Модель: qwen2.5:7b")

    if not check_server():
        print("Ollama недоступен. Запустите: ollama serve")
        sys.exit(1)

    print("\nЗагрузка каталога выкроек...")
    catalog = load_catalog()
    n = catalog.count("Файл:")
    print(f"  {n} выкроек в каталоге")

    # ── Шаг 1: Baseline (один вопрос, дефолт) ─────────────────────────────
    print("\n[ШАГ 1] Baseline — дефолтные параметры")
    for q in TEST_QUESTIONS:
        print(f"  Вопрос: «{q}»", end=" ", flush=True)
        r = run_single(
            config_name="baseline",
            model="qwen2.5:7b",
            system="Ты помощник по шитью и выкройкам. Отвечай на русском.",
            question=q,
            catalog=catalog,
            options=PARAM_CONFIGS["baseline"],
            verbose=True,
        )

    param_results = []
    quant_results = []
    prompt_results = []

    # ── Шаг 2: Параметры ──────────────────────────────────────────────────
    if run_p:
        param_results = bench_params(catalog, verbose=True)

    # ── Шаг 3: Квантование ────────────────────────────────────────────────
    if run_q:
        quant_results = bench_quant(catalog, verbose=True)

    # ── Шаг 4: Промпты ────────────────────────────────────────────────────
    if run_pr:
        prompt_results = bench_prompts(catalog, verbose=True)

    # ── Шаг 5: Итог ───────────────────────────────────────────────────────
    if param_results or quant_results or prompt_results:
        print_summary(param_results, quant_results, prompt_results)

    # ── Демо: лучшая конфигурация ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ДЕМО: лучшая конфигурация (low_temp + optimized prompt)")
    print("=" * 70)
    for q in TEST_QUESTIONS:
        print(f"\nВопрос: {q}")
        r = run_single(
            config_name="best",
            model="qwen2.5:7b",
            system=PROMPT_OPTIMIZED,
            question=q,
            catalog=catalog,
            options=PARAM_CONFIGS["low_temp"],
            verbose=False,
        )
        print(f"Ответ ({r.elapsed:.1f}s | {r.tokens_per_sec:.1f} tok/s):")
        print(r.answer)


if __name__ == "__main__":
    main()
