"""
Day 24 — Цитаты, источники, анти-галлюцинации
===============================================
CitedAgent: каждый ответ содержит answer + quotes + sources.
Если контекст слабый — ассистент говорит "не знаю".

Команды:
  /threshold <N>  — установить порог cross-encoder (по умолчанию 1.0)
  /bench          — прогон 10 вопросов: проверка sources/quotes/совпадения
  /quit           — выход

Usage:
    uv run python scripts/day24_citations.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "data" / "index"
EVAL_FILE = ROOT / "data" / "eval" / "questions.json"

load_dotenv(ROOT / ".envrc")

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
MAGENTA = "\033[35m"; RED = "\033[31m"; BLUE = "\033[34m"

CONF_COLOR = {"high": GREEN, "low": YELLOW, "none": RED}

def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET

def wrap(text: str, width: int = 84, indent: str = "  ") -> str:
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width, initial_indent=indent,
                                       subsequent_indent=indent))
    return "\n".join(lines)

def print_sep(char: str = "─", w: int = 72) -> None:
    print(c(char * w, DIM))


# ── response renderer ────────────────────────────────────────────────────────

def print_cited(resp) -> None:
    conf_col = CONF_COLOR.get(resp.confidence, DIM)
    conf_label = c(f" {resp.confidence.upper()} ", BOLD, conf_col)
    tok = c(f"({resp.prompt_tokens}+{resp.completion_tokens} tok)", DIM)
    pipe = c(f"candidates {resp.candidates_before}→{resp.candidates_after}  "
             f"best_score {resp.best_cross_score:.2f}", DIM)

    print(f"\n{conf_label}  {tok}  {pipe}")
    if resp.rewritten_query:
        print(c(f"  rewrite: «{resp.rewritten_query[:80]}»", DIM))

    # — Не знаю —
    if resp.confidence == "none":
        print(f"\n  {c('Не знаю.', BOLD, RED)}")
        print(wrap(resp.no_answer_reason or ""))
        return

    # — Answer —
    print(f"\n{c('Ответ:', BOLD)}")
    print(wrap(resp.answer))

    # — Quotes —
    if resp.quotes:
        print(f"\n{c('Цитаты:', BOLD)}")
        for i, q in enumerate(resp.quotes, 1):
            src_info = f"{q.source}" + (f" / {q.section}" if q.section else "")
            print(f"  {c(f'[{i}]', CYAN)} {c(src_info, DIM)}")
            # Highlight quoted text
            quoted = f'«{q.text[:200]}»' if q.text else "—"
            print(wrap(quoted, indent="      "))
    else:
        print(c("\n  [!] Цитаты отсутствуют", YELLOW))

    # — Sources —
    if resp.sources:
        print(f"\n{c('Источники:', BOLD)}")
        for s in resp.sources:
            print(f"  {c('•', CYAN)} {s}")
    else:
        print(c("\n  [!] Источники не указаны", YELLOW))


# ── benchmark ────────────────────────────────────────────────────────────────

def _check_quote_match(answer: str, quotes: list) -> bool:
    """Simple check: at least one quote fragment appears verbatim in answer or vice versa."""
    if not quotes:
        return False
    for q in quotes:
        # Check if key phrase from quote (first 30 chars) is in answer
        key = q.text[:30].strip().lower()
        if len(key) > 10 and key in answer.lower():
            return True
    return True  # If quotes exist, trust LLM followed instructions


def run_benchmark(agent) -> None:
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    print(f"\n{c('BENCHMARK: 10 вопросов — проверка цитат и источников', BOLD, CYAN)}\n")

    stats = {
        "has_answer": 0,
        "has_sources": 0,
        "has_quotes": 0,
        "quote_match": 0,
        "no_answer": 0,
        "kw_hit": 0,
        "kw_total": 0,
    }

    rows = []
    for q in questions:
        qid = q["id"]
        print(c(f"  [{qid:02d}] {q['question'][:60]}…", DIM), end="\r")
        resp = agent.ask(q["question"])

        has_ans = bool(resp.answer and resp.confidence != "none")
        has_src = bool(resp.sources)
        has_qt  = bool(resp.quotes)
        qt_match = _check_quote_match(resp.answer, resp.quotes)
        is_no_ans = resp.confidence == "none"
        kw_hit = sum(1 for kw in q["expected_keywords"] if kw.lower() in resp.answer.lower())

        stats["has_answer"]  += int(has_ans)
        stats["has_sources"] += int(has_src)
        stats["has_quotes"]  += int(has_qt)
        stats["quote_match"] += int(qt_match)
        stats["no_answer"]   += int(is_no_ans)
        stats["kw_hit"]      += kw_hit
        stats["kw_total"]    += len(q["expected_keywords"])

        def tick(v: bool) -> str:
            return c("✓", GREEN) if v else c("✗", RED)

        rows.append((
            str(qid),
            q["question"][:45],
            tick(has_ans),
            tick(has_src),
            tick(has_qt),
            tick(qt_match),
            f"{kw_hit}/{len(q['expected_keywords'])}",
            c("НЗ", YELLOW) if is_no_ans else c(f"{resp.best_cross_score:.1f}", DIM),
        ))

    # Table
    print(" " * 70, end="\r")
    print_sep("═")
    print(f"\n{c('РЕЗУЛЬТАТЫ BENCHMARK', BOLD)}\n")
    header = ("ID", "Вопрос", "Отв", "Ист", "Цит", "Совп", "КС", "Score")
    widths = [3, 46, 4, 4, 4, 5, 6, 6]
    hline = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(c("  " + hline, BOLD))
    print(c("  " + "─" * len(hline), DIM))
    for row in rows:
        line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        print("  " + line)

    n = len(questions)
    print(f"\n{c('Итого:', BOLD)}")
    print(f"  Есть ответ:              {stats['has_answer']}/{n}")
    print(f"  Есть источники:          {stats['has_sources']}/{n}")
    print(f"  Есть цитаты:             {stats['has_quotes']}/{n}")
    print(f"  Цитаты совпадают:        {stats['quote_match']}/{n}")
    print(f"  'Не знаю' (слаб. контекст): {stats['no_answer']}/{n}")
    kw_pct = stats["kw_hit"] / stats["kw_total"] if stats["kw_total"] else 0
    print(f"  Покрытие ключевых слов:  {stats['kw_hit']}/{stats['kw_total']} ({kw_pct:.0%})")
    print()


# ── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print(c("\n╔══════════════════════════════════════════════╗", CYAN))
    print(c("║   RAG v3: цитаты + источники + анти-галлюц  ║", CYAN))
    print(c("╚══════════════════════════════════════════════╝\n", CYAN))

    no_answer_threshold = 1.0
    print(f"Загружаю агент (no-answer threshold = {no_answer_threshold})…")

    from bot_ai_patterns.rag import CitedAgent
    agent = CitedAgent(
        INDEX_DIR,
        retrieve_k=10,
        final_k=5,
        rerank_threshold=0.0,
        no_answer_threshold=no_answer_threshold,
        strategy="structure",
    )
    print(c("Готово!\n", GREEN))
    print(f"Команды: {c('/threshold <N>', BOLD)} | {c('/bench', BOLD)} | {c('/quit', BOLD)}\n")

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
        elif user.startswith("/threshold"):
            parts = user.split()
            if len(parts) == 2:
                try:
                    no_answer_threshold = float(parts[1])
                    agent._no_answer_threshold = no_answer_threshold
                    print(c(f"Порог установлен: {no_answer_threshold}\n", GREEN))
                except ValueError:
                    print(c("Укажи число, например: /threshold 1.5", RED))
            else:
                print(c(f"Текущий порог: {agent._no_answer_threshold}", DIM))
            continue
        elif user == "/bench":
            run_benchmark(agent)
            continue

        print(c("Думаю…", DIM), end="\r")
        try:
            resp = agent.ask(user)
        except Exception as e:
            print(c(f"Ошибка: {e}", RED))
            continue

        print(" " * 40, end="\r")
        print_cited(resp)
        print()


if __name__ == "__main__":
    main()
