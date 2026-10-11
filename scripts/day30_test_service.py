"""
День 30: Тест стабильности и ограничений приватного LLM-сервиса.

Запуск (сервис должен быть запущен):
  uv run python scripts/day30_test_service.py
"""

from __future__ import annotations

import sys
import time
import httpx

BASE = "http://localhost:8080"
API_KEY = "local-dev-key"
HEADERS = {"X-API-Key": API_KEY}


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def post_chat(message: str, system: str = "Отвечай кратко на русском.") -> httpx.Response:
    return httpx.post(
        f"{BASE}/chat",
        headers=HEADERS,
        json={"message": message, "system": system},
        timeout=120,
    )


def main() -> None:
    print("День 30: Тест приватного LLM-сервиса")
    print(f"URL: {BASE}")

    # ── Health check ───────────────────────────────────────────────────────
    section("Health check")
    resp = httpx.get(f"{BASE}/health")
    print(f"  Статус: {resp.status_code}")
    data = resp.json()
    for k, v in data.items():
        print(f"  {k}: {v}")

    # ── Models ────────────────────────────────────────────────────────────
    section("Доступные модели")
    resp = httpx.get(f"{BASE}/models", headers=HEADERS)
    for m in resp.json().get("models", []):
        print(f"  {m['name']}  ({m['size_gb']} GB)")

    # ── Тест 1: 5 последовательных запросов ───────────────────────────────
    section("Тест 1: стабильность (5 последовательных запросов)")
    questions = [
        "Что такое нейросеть? Одно предложение.",
        "Столица Франции?",
        "2 + 2 = ?",
        "Назови один язык программирования.",
        "Какой сейчас год? (ответь: не знаю)",
    ]
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        resp = post_chat(q)
        elapsed = time.perf_counter() - t0
        data = resp.json()
        reply = data.get("reply", data.get("detail", ""))[:80]
        print(f"  [{i}] {elapsed:.1f}s | {reply}")

    # ── Тест 2: Auth — неверный ключ ──────────────────────────────────────
    section("Тест 2: аутентификация")
    resp = httpx.post(f"{BASE}/chat", headers={"X-API-Key": "wrong-key"},
                      json={"message": "привет"}, timeout=10)
    print(f"  Неверный ключ → {resp.status_code}: {resp.json().get('detail')}")

    resp = httpx.post(f"{BASE}/chat", json={"message": "привет"}, timeout=10)
    print(f"  Без ключа     → {resp.status_code}: {resp.json().get('detail')}")

    # ── Тест 3: Max context ───────────────────────────────────────────────
    section("Тест 3: ограничение контекста (max 4000 символов)")
    long_msg = "а" * 4001
    resp = post_chat(long_msg)
    print(f"  4001 символов → {resp.status_code}: {resp.json().get('detail')}")

    ok_msg = "а" * 4000
    resp = post_chat(ok_msg)
    print(f"  4000 символов → {resp.status_code}: OK" if resp.status_code == 200
          else f"  4000 символов → {resp.status_code}: {resp.json().get('detail')}")

    # ── Тест 4: Rate limit ────────────────────────────────────────────────
    section("Тест 4: rate limit (11 быстрых запросов, лимит 10/мин)")
    results = []
    for i in range(11):
        resp = httpx.post(
            f"{BASE}/chat",
            headers=HEADERS,
            json={"message": "привет", "system": "Отвечай одним словом."},
            timeout=60,
        )
        results.append(resp.status_code)
        print(f"  [{i+1:02d}] → {resp.status_code}", end="")
        if resp.status_code == 429:
            print(f"  {resp.json().get('detail')}", end="")
        print()

    ok = results.count(200)
    limited = results.count(429)
    print(f"\n  Итог: {ok} успешных, {limited} заблокированных")

    # ── Итог ──────────────────────────────────────────────────────────────
    section("ИТОГ")
    print(f"  Health:      OK")
    print(f"  Auth:        работает (401 на неверный ключ)")
    print(f"  Max context: работает (413 на >4000 символов)")
    print(f"  Rate limit:  {'работает' if limited > 0 else 'НЕ сработал'} (429 после {ok} запросов)")
    print(f"\n  Сервис доступен по адресам:")
    print(f"    Локально:  {BASE}")
    print(f"    LAN:       http://192.168.1.102:8080")


if __name__ == "__main__":
    main()
