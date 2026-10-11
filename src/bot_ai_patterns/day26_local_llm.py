"""
День 26: Локальная LLM через Ollama HTTP API.

Три запроса разной сложности к qwen2.5:7b запущенной локально.
"""

import httpx
import json
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"


def chat(prompt: str, system: str = "") -> str:
    """Отправляет запрос к локальной LLM и возвращает ответ."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def check_server() -> bool:
    """Проверяет, что Ollama сервер доступен."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except httpx.ConnectError:
        return False


def run_query(label: str, prompt: str, system: str = "") -> dict:
    """Выполняет запрос и замеряет время."""
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"Запрос: {prompt}")
    print("-" * 60)

    start = time.perf_counter()
    answer = chat(prompt, system)
    elapsed = time.perf_counter() - start

    print(f"Ответ:\n{answer}")
    print(f"\n  {elapsed:.1f}s")
    return {"label": label, "prompt": prompt, "answer": answer, "elapsed": elapsed}


def main():
    if not check_server():
        print("Ollama сервер не запущен. Выполните: ollama serve")
        return

    print(f"Модель: {MODEL}")
    print(f"API: {OLLAMA_URL}")

    results = []

    # Запрос 1 — простой: базовый диалог
    results.append(run_query(
        label="1. Простой — диалог",
        prompt="Привет! Как тебя зовут и что ты умеешь?",
    ))

    # Запрос 2 — средний: объяснение концепции
    results.append(run_query(
        label="2. Средний — концепция",
        prompt="Объясни разницу между RAG и fine-tuning в 3 предложениях. Когда использовать каждый подход?",
        system="Ты эксперт по ML. Отвечай кратко и по делу.",
    ))

    # Запрос 3 — сложный: генерация кода
    results.append(run_query(
        label="3. Сложный — генерация кода",
        prompt=(
            "Напиши функцию на Python, которая рекурсивно обходит вложенный словарь "
            "и возвращает список всех ключей в формате 'parent.child.key'. "
            "Добавь docstring и пример использования."
        ),
        system="Ты опытный Python-разработчик. Пиши чистый, читаемый код.",
    ))

    # Итог
    print(f"\n{'='*60}")
    print("ИТОГ")
    print("=" * 60)
    total = sum(r["elapsed"] for r in results)
    for r in results:
        print(f"  {r['label']}: {r['elapsed']:.1f}s")
    print(f"  Всего: {total:.1f}s")


if __name__ == "__main__":
    main()
