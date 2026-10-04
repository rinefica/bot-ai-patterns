"""
Day 23 — Реранкинг и фильтрация
=================================
4 режима:
  /plain   — только LLM, без поиска
  /rag     — bi-encoder retrieval (day22)
  /rerank  — retrieval + cross-encoder reranker + фильтр по порогу
  /full    — query rewrite + retrieval + rerank  ← по умолчанию

  /compare — сравнить все 4 режима на одном вопросе
  /bench   — прогон 10 контрольных вопросов по всем режимам
  /quit    — выход

Usage:
    uv run python scripts/day23_rerank.py
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

# ── colours ────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
MAGENTA = "\033[35m"; RED = "\033[31m"; BLUE = "\033[34m"

def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET

MODE_COLORS = {
    "plain":  YELLOW,
    "rag":    CYAN,
    "rerank": GREEN,
    "full":   MAGENTA,
}
MODE_LABELS = {
    "plain":  " PLAIN  ",
    "rag":    "  RAG   ",
    "rerank": " RERANK ",
    "full":   "  FULL  ",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 86, indent: str = "  ") -> str:
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width, initial_indent=indent,
                                       subsequent_indent=indent))
    return "\n".join(lines)


def print_response(resp, verbose: bool = True) -> None:
    col = MODE_COLORS.get(resp.mode, "")
    label = c(MODE_LABELS.get(resp.mode, resp.mode), BOLD, col)
    tok = c(f"({resp.prompt_tokens}+{resp.completion_tokens} tok)", DIM)

    extra = ""
    if resp.candidates_before:
        extra = c(f"  candidates: {resp.candidates_before}→{resp.candidates_after}", DIM)

    print(f"\n{label}  {tok}{extra}")

    if resp.rewritten_query:
        print(c(f"  query rewrite: «{resp.rewritten_query}»", DIM))

    print()
    print(wrap(resp.answer))

    if verbose and resp.ranked:
        print(f"\n{c('Источники (cross-score | bi-score | файл / секция):', BOLD)}")
        for r in resp.ranked:
            cs = c(f"{r.cross_score:+.2f}", GREEN if r.cross_score > 0 else RED)
            bs = c(f"{r.bi_score:.2f}", DIM)
            title = r.original.title
            sec = (r.original.section or "—")[:45]
            print(f"  {cs}  bi={bs}  {title} / {sec}")
    elif verbose and resp.sources:
        print(f"\n{c('Источники:', BOLD)}")
        for s in resp.sources:
            print(f"  {c('•', CYAN)} {s.title} / {s.section or '—'} (score {s.score:.2f})")


def print_sep(char: str = "─", w: int = 72) -> None:
    print(c(char * w, DIM))


# ── compare mode ────────────────────────────────────────────────────────────

def run_compare(agent, question: str) -> None:
    print(f"\n{c('Вопрос:', BOLD)} {question}\n")
    for mode in ("plain", "rag", "rerank", "full"):
        print(c("Думаю…", DIM), end="\r")
        fn = getattr(agent, f"ask_{mode}")
        resp = fn(question)
        print_response(resp, verbose=(mode in ("rerank", "full")))
        print_sep()


# ── benchmark ────────────────────────────────────────────────────────────────

def _score(resp, expected_kw: list[str], expected_src: list[str]) -> dict:
    used_src = {s.title for s in resp.sources}
    src_hit = len(used_src & set(expected_src))
    kw_hit = sum(1 for kw in expected_kw if kw.lower() in resp.answer.lower())
    return {
        "src_hit": src_hit, "src_total": len(expected_src),
        "kw_hit": kw_hit, "kw_total": len(expected_kw),
    }


def run_benchmark(agent) -> None:
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    modes = ("plain", "rag", "rerank", "full")
    totals = {m: {"src_hit": 0, "src_total": 0, "kw_hit": 0, "kw_total": 0} for m in modes}

    print(f"\n{c('BENCHMARK: 10 вопросов × 4 режима', BOLD, CYAN)}\n")

    for q in questions:
        qid = q["id"]
        print_sep("═")
        print(f"\n{c(f'[{qid}]', BOLD)} {q['question']}")
        print(c(f"Ожидаемые источники: {', '.join(q['expected_sources'])}", DIM))

        for mode in modes:
            print(c(f"  [{mode}] думаю…", DIM), end="\r")
            fn = getattr(agent, f"ask_{mode}")
            resp = fn(q["question"])
            sc = _score(resp, q["expected_keywords"], q["expected_sources"])
            for k in sc:
                totals[mode][k] += sc[k]
            col = MODE_COLORS[mode]
            label = c(MODE_LABELS[mode], BOLD, col)
            src_str = c(f"src {sc['src_hit']}/{sc['src_total']}", GREEN if sc['src_hit'] else RED)
            kw_str  = c(f"kw {sc['kw_hit']}/{sc['kw_total']}", GREEN if sc['kw_hit'] >= sc['kw_total']//2 else YELLOW)
            tok = c(f"({resp.prompt_tokens}+{resp.completion_tokens}tok)", DIM)
            print(f"  {label}  {src_str}  {kw_str}  {tok}          ")
            if resp.rewritten_query:
                print(c(f"           rewrite: «{resp.rewritten_query[:70]}»", DIM))

    # Summary table
    print_sep("═")
    print(f"\n{c('ИТОГ', BOLD, CYAN)}\n")
    header = f"  {'Режим':<8}  {'Источники':>12}  {'Ключ. слова':>13}"
    print(c(header, BOLD))
    print(c("  " + "─" * (len(header) - 2), DIM))
    for mode in modes:
        t = totals[mode]
        src_pct = t["src_hit"] / t["src_total"] if t["src_total"] else 0
        kw_pct  = t["kw_hit"]  / t["kw_total"]  if t["kw_total"]  else 0
        col = MODE_COLORS[mode]
        label = c(f"{mode:<8}", col)
        src_s = f"{t['src_hit']}/{t['src_total']} ({src_pct:.0%})"
        kw_s  = f"{t['kw_hit']}/{t['kw_total']} ({kw_pct:.0%})"
        print(f"  {label}  {src_s:>12}  {kw_s:>13}")
    print()


# ── main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print(c("\n╔═══════════════════════════════════════════╗", CYAN))
    print(c("║   RAG v2: реранкинг + query rewrite       ║", CYAN))
    print(c("╚═══════════════════════════════════════════╝\n", CYAN))

    print("Загружаю индекс, модели эмбеддингов и reranker…")
    from bot_ai_patterns.rag import RagAgent
    agent = RagAgent(
        INDEX_DIR,
        retrieve_k=10,   # top-10 кандидатов для reranker
        final_k=5,       # top-5 после фильтрации
        threshold=0.0,   # порог cross-encoder (0 = отсекаем только отрицательные)
        strategy="structure",
    )
    print(c("Готово!\n", GREEN))

    cmds = "/plain | /rag | /rerank | /full | /compare | /bench | /quit"
    print(f"Команды: {c(cmds, BOLD)}")
    print(f"Текущий режим: {c('full', BOLD, MAGENTA)} (rewrite + rerank)\n")

    mode = "full"

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
        elif user in ("/plain", "/rag", "/rerank", "/full"):
            mode = user[1:]
            col = MODE_COLORS[mode]
            print(c(f"Режим: {mode}\n", col))
            continue
        elif user == "/compare":
            try:
                q = input(c("Вопрос для сравнения> ", DIM)).strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if q:
                run_compare(agent, q)
            continue
        elif user == "/bench":
            run_benchmark(agent)
            continue

        print(c("Думаю…", DIM), end="\r")
        try:
            fn = getattr(agent, f"ask_{mode}")
            resp = fn(user)
        except Exception as e:
            print(c(f"Ошибка: {e}", RED))
            continue

        print(" " * 30, end="\r")
        print_response(resp)
        print()


if __name__ == "__main__":
    main()
