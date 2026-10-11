"""
День 29: Оптимизация локальной LLM под задачу подбора выкроек.

Проводит бенчмарк по 4 осям:
  1. Baseline — дефолтные параметры
  2. Параметры генерации — temperature, num_predict, num_ctx, repeat_penalty
  3. Квантование — Q4_K_M (default) / Q2_K / Q8_0
  4. Промпт — общий vs domain-specific
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434"

# ── Тестовые вопросы (4 штуки, разная сложность) ──────────────────────────

TEST_QUESTIONS = [
    "Хочу сшить брюки для начинающих.",
    "Нужна выкройка летнего изделия из трикотажа.",
    "Ищу что-то для продвинутых, со сложным кроем.",
    "Хочу боди или топ приталенного силуэта.",
]

# ── Каталог выкроек (загружается из индекса day27) ─────────────────────────

def load_catalog() -> str:
    index_path = Path.home() / ".sewing_agent" / "index.json"
    if not index_path.exists():
        return "Каталог пуст — запустите day27_sewing_agent index."
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"Файл: {name}\nОписание: {info['description']}"
        for name, info in data.items()
    )

# ── Промпты ────────────────────────────────────────────────────────────────

PROMPT_GENERIC = "Ты помощник по шитью и выкройкам. Отвечай на русском."

PROMPT_OPTIMIZED = """\
Ты — специализированный ассистент по подбору выкроек. Работаешь только с предоставленным каталогом.

Правила:
1. Подбирай ТОЛЬКО из каталога ниже — не придумывай несуществующие файлы.
2. Для каждой подходящей выкройки указывай: имя файла, уровень сложности, почему подходит.
3. Если ни одна не подходит — честно скажи об этом.
4. Отвечай кратко и структурированно: сначала список файлов, потом пояснения.
5. Не повторяй вопрос пользователя в ответе."""

def build_user_prompt(catalog: str, question: str) -> str:
    return (
        f"Каталог выкроек:\n\n{catalog}\n\n"
        f"Запрос: {question}\n\n"
        "Подбери подходящие выкройки из каталога."
    )

# ── Конфигурации параметров ────────────────────────────────────────────────

PARAM_CONFIGS = {
    "baseline": {
        "temperature": 0.8,
        "num_predict": 512,
        "num_ctx": 4096,
        "repeat_penalty": 1.1,
        "top_k": 40,
    },
    "low_temp": {
        "temperature": 0.1,
        "num_predict": 512,
        "num_ctx": 4096,
        "repeat_penalty": 1.1,
        "top_k": 40,
    },
    "mid_temp": {
        "temperature": 0.3,
        "num_predict": 512,
        "num_ctx": 4096,
        "repeat_penalty": 1.1,
        "top_k": 40,
    },
    "large_ctx": {
        "temperature": 0.1,
        "num_predict": 512,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
        "top_k": 40,
    },
    "no_repeat": {
        "temperature": 0.1,
        "num_predict": 512,
        "num_ctx": 4096,
        "repeat_penalty": 1.0,
        "top_k": 20,
    },
}

# ── Квантованные модели ────────────────────────────────────────────────────

QUANT_MODELS = {
    "Q4_K_M (default)": "qwen2.5:7b",
    "Q2_K (fast)":       "qwen2.5:7b-instruct-q2_k",
    "Q8_0 (quality)":    "qwen2.5:7b-instruct-q8_0",
}

# ── Ядро: один запрос ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    config: str
    question: str
    answer: str
    elapsed: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0

    def short_answer(self, n: int = 120) -> str:
        return self.answer[:n].replace("\n", " ") + ("…" if len(self.answer) > n else "")


def ollama_chat(
    model: str,
    system: str,
    user: str,
    options: dict,
) -> tuple[str, int, int]:
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": options,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    content = data["message"]["content"]
    pt = data.get("prompt_eval_count", 0)
    ct = data.get("eval_count", 0)
    return content, pt, ct


def run_single(
    config_name: str,
    model: str,
    system: str,
    question: str,
    catalog: str,
    options: dict,
    verbose: bool = False,
) -> RunResult:
    user = build_user_prompt(catalog, question)
    t0 = time.perf_counter()
    answer, pt, ct = ollama_chat(model, system, user, options)
    elapsed = time.perf_counter() - t0
    tps = ct / elapsed if elapsed > 0 else 0

    result = RunResult(
        config=config_name,
        question=question,
        answer=answer,
        elapsed=elapsed,
        prompt_tokens=pt,
        completion_tokens=ct,
        tokens_per_sec=tps,
    )
    if verbose:
        print(f"    {elapsed:.1f}s | {ct} токенов | {tps:.1f} tok/s")
        print(f"    {result.short_answer()}")
    return result


# ── Бенчмарки ──────────────────────────────────────────────────────────────

def bench_params(catalog: str, verbose: bool = True) -> list[RunResult]:
    """Шаг 2: сравнение конфигураций параметров."""
    results = []
    model = "qwen2.5:7b"
    # Используем один вопрос для чистого сравнения параметров
    question = TEST_QUESTIONS[2]  # сложный запрос

    print(f"\n[ШАГ 2] Параметры генерации | вопрос: «{question}»")
    for name, opts in PARAM_CONFIGS.items():
        print(f"  {name}:", end=" ", flush=True)
        r = run_single(name, model, PROMPT_GENERIC, question, catalog, opts, verbose)
        results.append(r)
    return results


def bench_quant(catalog: str, verbose: bool = True) -> list[RunResult]:
    """Шаг 3: сравнение квантований."""
    results = []
    question = TEST_QUESTIONS[2]
    options = PARAM_CONFIGS["low_temp"]

    print(f"\n[ШАГ 3] Квантование | вопрос: «{question}»")
    for label, model in QUANT_MODELS.items():
        # Проверяем доступность модели
        try:
            with httpx.Client(timeout=5) as c:
                tags = c.get(f"{OLLAMA_URL}/api/tags").json()
            available = [m["name"] for m in tags.get("models", [])]
            if model not in available:
                print(f"  {label}: пропущен (не установлен: {model})")
                continue
        except Exception:
            print(f"  {label}: пропущен (Ollama недоступен)")
            continue

        print(f"  {label}:", end=" ", flush=True)
        r = run_single(label, model, PROMPT_GENERIC, question, catalog, options, verbose)
        results.append(r)
    return results


def bench_prompts(catalog: str, verbose: bool = True) -> list[RunResult]:
    """Шаг 4: сравнение промптов на всех 4 вопросах."""
    results = []
    model = "qwen2.5:7b"
    options = PARAM_CONFIGS["low_temp"]

    print("\n[ШАГ 4] Промпты — generic vs optimized")
    for question in TEST_QUESTIONS:
        print(f"\n  Вопрос: «{question}»")
        for label, system in [("generic", PROMPT_GENERIC), ("optimized", PROMPT_OPTIMIZED)]:
            print(f"    [{label}]:", end=" ", flush=True)
            r = run_single(
                config_name=f"prompt_{label}",
                model=model,
                system=system,
                question=question,
                catalog=catalog,
                options=options,
                verbose=verbose,
            )
            results.append(r)
    return results


# ── Итоговая таблица ───────────────────────────────────────────────────────

def print_summary(
    param_results: list[RunResult],
    quant_results: list[RunResult],
    prompt_results: list[RunResult],
) -> None:
    print(f"\n{'='*70}")
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 70)

    def avg_time(rs: list[RunResult]) -> float:
        return sum(r.elapsed for r in rs) / len(rs) if rs else 0

    def avg_tps(rs: list[RunResult]) -> float:
        return sum(r.tokens_per_sec for r in rs) / len(rs) if rs else 0

    # Параметры
    print("\n[Параметры генерации]")
    print(f"  {'Конфиг':<20} {'Время':>8} {'tok/s':>8} {'Токены':>8}")
    print("  " + "-" * 46)
    for r in param_results:
        print(f"  {r.config:<20} {r.elapsed:>7.1f}s {r.tokens_per_sec:>7.1f} {r.completion_tokens:>8}")

    # Квантование
    if quant_results:
        print("\n[Квантование]")
        print(f"  {'Модель':<25} {'Время':>8} {'tok/s':>8} {'Токены':>8}")
        print("  " + "-" * 51)
        for r in quant_results:
            print(f"  {r.config:<25} {r.elapsed:>7.1f}s {r.tokens_per_sec:>7.1f} {r.completion_tokens:>8}")

    # Промпты
    if prompt_results:
        print("\n[Промпты — среднее по 4 вопросам]")
        generic = [r for r in prompt_results if r.config == "prompt_generic"]
        optimized = [r for r in prompt_results if r.config == "prompt_optimized"]
        print(f"  generic:   {avg_time(generic):.1f}s avg | {avg_tps(generic):.1f} tok/s")
        print(f"  optimized: {avg_time(optimized):.1f}s avg | {avg_tps(optimized):.1f} tok/s")

    # Рекомендация
    print(f"\n{'='*70}")
    print("РЕКОМЕНДАЦИЯ ДЛЯ ЗАДАЧИ ПОДБОРА ВЫКРОЕК")
    print("-" * 70)
    best_param = min(param_results, key=lambda r: r.elapsed) if param_results else None
    if best_param:
        cfg = PARAM_CONFIGS.get(best_param.config, {})
        print(f"  Лучшие параметры: {best_param.config}")
        print(f"    temperature={cfg.get('temperature')} | num_ctx={cfg.get('num_ctx')} | repeat_penalty={cfg.get('repeat_penalty')}")
    print(f"  Промпт: optimized (структурированный, с ограничением галлюцинаций)")
    print(f"  Квантование: Q4_K_M — оптимальный баланс качества и скорости на M2 Pro")
