"""
Day 22 — RAG CLI диалог
========================
Агент с двумя режимами: с RAG и без RAG.

Команды в диалоге:
  /rag   — переключить в режим RAG (по умолчанию)
  /plain — переключить в режим без RAG
  /bench — прогнать 10 контрольных вопросов и сравнить
  /quit  — выйти

Usage:
    uv run python scripts/day22_rag.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "data" / "index"
EVAL_FILE = ROOT / "data" / "eval" / "questions.json"

load_dotenv(ROOT / ".envrc")

# ── colours ────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM   = "\033[2m"
RED   = "\033[31m"

def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


# ── helpers ─────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 88, indent: str = "  ") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width, initial_indent=indent,
                                       subsequent_indent=indent))
    return "\n".join(lines)


def print_response(resp, show_sources: bool = True) -> None:
    mode_label = c(" RAG ", BOLD, GREEN) if resp.mode == "rag" else c(" PLAIN ", BOLD, YELLOW)
    tokens = f"({resp.prompt_tokens}+{resp.completion_tokens} tok)"
    print(f"\n{mode_label}  {c(tokens, DIM)}\n")
    print(wrap(resp.answer))

    if show_sources and resp.sources:
        print(f"\n{c('Источники:', BOLD)}")
        seen = {}
        for s in resp.sources:
            key = s.title
            if key not in seen:
                seen[key] = []
            seen[key].append(f"{s.section or '—'} (score {s.score:.2f})")
        for title, sections in seen.items():
            print(f"  {c('•', CYAN)} {title}")
            for sec in sections[:3]:
                print(f"      {c(sec, DIM)}")


def print_separator(char: str = "─", width: int = 72) -> None:
    print(c(char * width, DIM))


# ── benchmark ───────────────────────────────────────────────────────────────

def run_benchmark(agent) -> None:
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    print(f"\n{c('BENCHMARK: 10 контрольных вопросов', BOLD, CYAN)}\n")

    results = []
    for q in questions:
        print_separator()
        qid = q["id"]
        print(f"\n{c(f'[{qid}]', BOLD)} {q['question']}")
        print(f"{c('Ожидаемые источники:', DIM)} {', '.join(q['expected_sources'])}\n")

        plain = agent.ask_plain(q["question"])
        rag   = agent.ask_rag(q["question"])

        print(c("Без RAG:", BOLD, YELLOW))
        print(wrap(plain.answer))

        print(f"\n{c('С RAG:', BOLD, GREEN)}")
        print(wrap(rag.answer))

        used_sources = {s.title for s in rag.sources}
        expected = set(q["expected_sources"])
        hit = len(used_sources & expected)
        hit_kw = sum(
            1 for kw in q["expected_keywords"]
            if kw.lower() in rag.answer.lower()
        )

        print(f"\n{c('Оценка RAG:', BOLD)}")
        print(f"  Источники попали: {hit}/{len(expected)}")
        print(f"  Ключевых слов:   {hit_kw}/{len(q['expected_keywords'])}")

        results.append({
            "id": q["id"],
            "source_hits": hit,
            "source_total": len(expected),
            "kw_hits": hit_kw,
            "kw_total": len(q["expected_keywords"]),
        })

    # Summary
    print_separator("═")
    print(f"\n{c('ИТОГ BENCHMARK', BOLD)}\n")
    avg_src = sum(r["source_hits"] for r in results) / (sum(r["source_total"] for r in results) or 1)
    avg_kw  = sum(r["kw_hits"] for r in results) / (sum(r["kw_total"] for r in results) or 1)

    table = [("ID", "Источники", "Ключ. слова")]
    for r in results:
        src_str = f"{r['source_hits']}/{r['source_total']}"
        kw_str  = f"{r['kw_hits']}/{r['kw_total']}"
        table.append((str(r["id"]), src_str, kw_str))

    col_w = [max(len(row[i]) for row in table) for i in range(3)]
    for i, row in enumerate(table):
        line = "  ".join(cell.ljust(col_w[j]) for j, cell in enumerate(row))
        if i == 0:
            print(c("  " + line, BOLD))
            print(c("  " + "─" * (sum(col_w) + 4), DIM))
        else:
            print("  " + line)

    print(f"\n  Средняя точность источников: {c(f'{avg_src:.0%}', BOLD, GREEN)}")
    print(f"  Покрытие ключевых слов:      {c(f'{avg_kw:.0%}', BOLD, GREEN)}")
    print()


# ── main loop ───────────────────────────────────────────────────────────────

def main() -> None:
    print(c("\n╔══════════════════════════════════════╗", CYAN))
    print(c("║   RAG-ассистент по пошиву одежды     ║", CYAN))
    print(c("╚══════════════════════════════════════╝\n", CYAN))

    print("Загружаю индекс и модель…")
    from bot_ai_patterns.rag import RagAgent
    agent = RagAgent(INDEX_DIR, top_k=5, strategy="structure")
    print(c("Готово!\n", GREEN))

    print(f"Команды: {c('/rag', BOLD)} | {c('/plain', BOLD)} | {c('/bench', BOLD)} | {c('/quit', BOLD)}")
    print(f"Текущий режим: {c('RAG', BOLD, GREEN)}\n")

    mode = "rag"

    while True:
        try:
            user = input(c("Вопрос> ", BOLD, CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user:
            continue

        if user == "/quit":
            print("Пока!")
            break
        elif user == "/rag":
            mode = "rag"
            print(c("Режим: RAG (с поиском по документам)\n", GREEN))
            continue
        elif user == "/plain":
            mode = "plain"
            print(c("Режим: PLAIN (без поиска, только LLM)\n", YELLOW))
            continue
        elif user == "/bench":
            run_benchmark(agent)
            continue

        print(c("Думаю…", DIM), end="\r")
        try:
            resp = agent.ask_rag(user) if mode == "rag" else agent.ask_plain(user)
        except Exception as e:
            print(c(f"Ошибка: {e}", RED))
            continue

        print(" " * 20, end="\r")  # clear "Думаю…"
        print_response(resp)
        print()


if __name__ == "__main__":
    main()
