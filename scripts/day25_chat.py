"""
Day 25 — Мини-чат RAG + память задачи
=======================================
Команды:
  /state    — показать текущее состояние задачи
  /history  — показать историю диалога
  /reset    — начать новый диалог
  /scenario1 — запустить сценарий «Пошив брюк с нуля» (10 сообщений)
  /scenario2 — запустить сценарий «Разбор ошибок при пошиве блузки» (10 сообщений)
  /quit     — выход

Usage:
    uv run python scripts/day25_chat.py
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "data" / "index"
load_dotenv(ROOT / ".envrc")

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
MAGENTA = "\033[35m"; RED = "\033[31m"

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

def sep(char: str = "─", w: int = 72) -> None:
    print(c(char * w, DIM))


# ── Scenarios ────────────────────────────────────────────────────────────────

SCENARIO_1 = [
    "Хочу сшить свободные брюки с резинкой на поясе. С чего начать?",
    "Какую ткань лучше выбрать? Я начинающая.",
    "Как правильно снять мерки для брюк?",
    "Нужно ли декатировать трикотаж перед раскроем?",
    "Как разложить выкройку на ткани?",
    "Как соединить детали брюк — в какой последовательности?",
    "Как правильно стачать боковые швы?",
    "Как сделать пояс с кулиской для резинки?",
    "Как отрезать резинку нужной длины?",
    "Как обработать низ брюк?",
]

SCENARIO_2 = [
    "Шью блузку, но не понимаю как обработать горловину. Какие варианты есть?",
    "У меня есть обтачка горловины. Как её правильно притачать?",
    "Шов обтачивания морщит и стягивает горловину. Что делаю не так?",
    "Надо ли обрезать припуски после обтачивания? Насколько?",
    "Как закрепить обтачку изнутри, чтобы она не вываливалась?",
    "Бретели блузки тоже нужно обтачивать?",
    "Как обработать проймы без рукавов — тем же способом?",
    "Что такое ВТО и когда его делать при пошиве блузки?",
    "Как проверить, что горловина сидит правильно перед финальной строчкой?",
    "Какие финальные работы нужно сделать после сборки блузки?",
]


def run_scenario(agent, session, messages: list[str], title: str) -> None:
    print(f"\n{c('═' * 72, CYAN)}")
    print(c(f"  СЦЕНАРИЙ: {title}", BOLD, CYAN))
    print(c('═' * 72, CYAN))

    for i, question in enumerate(messages, 1):
        print(f"\n{c(f'[{i}/{len(messages)}]', BOLD, MAGENTA)} "
              f"{c('Пользователь:', BOLD)} {question}")
        sep()

        print(c("  думаю…", DIM), end="\r")
        resp = agent.chat(session, question)
        print(" " * 30, end="\r")

        tok = c(f"({resp.prompt_tokens}+{resp.completion_tokens} tok)", DIM)
        score_info = c(f"score {resp.best_score:.1f}  {resp.candidates_before}→{resp.candidates_after}", DIM)
        print(c(f"  Ассистент:", BOLD, GREEN) + f"  {tok}  {score_info}")
        if resp.rewritten_query and resp.rewritten_query[:40] not in question[:40]:
            print(c(f"  rewrite: «{resp.rewritten_query[:70]}»", DIM))
        print()
        print(wrap(resp.answer))

        if resp.sources:
            print(f"\n  {c('Источники:', BOLD)}")
            for s in resp.sources:
                print(f"    {c('📌', '')} {s}")
        print()

    # Show task state at the end
    sep()
    print(c(f"\n  СОСТОЯНИЕ ЗАДАЧИ после сценария:", BOLD, YELLOW))
    print(wrap(session.state.summary(), indent="    "))
    print()


# ── Main loop ────────────────────────────────────────────────────────────────

def print_response(resp, session) -> None:
    tok = c(f"({resp.prompt_tokens}+{resp.completion_tokens} tok)", DIM)
    score = c(f"score {resp.best_score:.1f}", DIM)
    print(f"\n{c('Ассистент:', BOLD, GREEN)}  {tok}  {score}")
    if resp.rewritten_query:
        print(c(f"  rewrite: «{resp.rewritten_query[:70]}»", DIM))
    print()
    print(wrap(resp.answer))
    if resp.sources:
        print(f"\n  {c('Источники:', BOLD)}")
        for s in resp.sources:
            print(f"    {c('📌', '')} {s}")
    print()


def main() -> None:
    print(c("\n╔══════════════════════════════════════════════════╗", CYAN))
    print(c("║   RAG-чат с памятью задачи  (Day 25)             ║", CYAN))
    print(c("╚══════════════════════════════════════════════════╝\n", CYAN))

    print("Загружаю агент…")
    from bot_ai_patterns.rag import ChatAgent, ChatSession
    agent = ChatAgent(
        INDEX_DIR,
        retrieve_k=10,
        final_k=4,
        rerank_threshold=0.0,
        no_context_threshold=1.0,
        strategy="structure",
    )
    session = ChatSession(max_history=20)
    print(c("Готово!\n", GREEN))

    cmds = "/state | /history | /reset | /scenario1 | /scenario2 | /quit"
    print(f"Команды: {c(cmds, BOLD)}\n")

    while True:
        try:
            user = input(c("Вы> ", BOLD, CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user:
            continue

        if user == "/quit":
            print("Пока!")
            break

        elif user == "/state":
            print(f"\n{c('Состояние задачи:', BOLD, YELLOW)}")
            print(wrap(session.state.summary(), indent="  "))
            print()

        elif user == "/history":
            print(f"\n{c('История диалога:', BOLD)}")
            for m in session.history[-10:]:
                role_label = c("Вы", CYAN) if m.role == "user" else c("Ассист.", GREEN)
                print(f"  [{m.ts}] {role_label}: {m.content[:100]}")
                if m.sources:
                    print(f"    {c('→ ' + ', '.join(m.sources[:2]), DIM)}")
            print()

        elif user == "/reset":
            session.reset()
            print(c("Сессия сброшена.\n", YELLOW))

        elif user == "/scenario1":
            session.reset()
            run_scenario(agent, session, SCENARIO_1, "Пошив свободных брюк с нуля")

        elif user == "/scenario2":
            session.reset()
            run_scenario(agent, session, SCENARIO_2, "Разбор ошибок при пошиве блузки")

        else:
            print(c("думаю…", DIM), end="\r")
            try:
                resp = agent.chat(session, user)
            except Exception as e:
                print(c(f"Ошибка: {e}", RED))
                continue
            print(" " * 30, end="\r")
            print_response(resp, session)


if __name__ == "__main__":
    main()
