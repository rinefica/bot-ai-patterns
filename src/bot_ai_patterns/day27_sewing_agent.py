"""
День 27: CLI-агент для подбора выкроек.

Читает PDF из /Users/kate_yal/sewing_docs, создаёт описания через
локальную LLM (qwen2.5:7b / Ollama), хранит в долговременной памяти,
отвечает на запросы пользователя в интерактивном чате.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pymupdf

DOCS_DIR = Path("/Users/kate_yal/sewing_docs")
INDEX_FILE = Path.home() / ".sewing_agent" / "index.json"
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"
PAGES_TO_READ = 3

# Оптимизированные параметры (день 29: temperature=0.1 даёт x2.5 ускорение
# при лучшем качестве; num_ctx=4096 достаточно для задачи подбора выкроек)
GENERATE_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 512,
    "num_ctx": 4096,
    "repeat_penalty": 1.1,
    "top_k": 40,
}

SYSTEM_SEARCH = """\
Ты — специализированный ассистент по подбору выкроек. Работаешь только с предоставленным каталогом.

Правила:
1. Подбирай ТОЛЬКО из каталога — не придумывай несуществующие файлы.
2. Для каждой подходящей выкройки указывай: имя файла, уровень сложности, почему подходит.
3. Если ни одна не подходит — честно скажи об этом.
4. Отвечай кратко и структурированно: сначала список файлов, потом пояснения.
5. Не повторяй вопрос пользователя в ответе.\
"""


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    """Извлекает текст первых PAGES_TO_READ страниц из PDF."""
    doc = pymupdf.open(str(pdf_path))
    pages = min(PAGES_TO_READ, len(doc))
    text = "\n\n".join(doc[i].get_text() for i in range(pages))
    doc.close()
    return text.strip()


# ---------------------------------------------------------------------------
# Ollama HTTP API
# ---------------------------------------------------------------------------

def _chat(messages: list[dict], options: dict | None = None) -> str:
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": options or GENERATE_OPTIONS,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def describe_pattern(filename: str, text: str) -> str:
    """Просит LLM составить описание выкройки по тексту первых страниц."""
    prompt = (
        f"Перед тобой начало PDF-файла с выкройкой «{filename}».\n\n"
        f"{text[:3000]}\n\n"
        "Составь краткое описание выкройки (3-5 предложений):\n"
        "- тип изделия\n"
        "- для кого (женщины, мужчины, дети, пол не указан)\n"
        "- уровень сложности (начинающий / средний / продвинутый) если понятно\n"
        "- размеры если указаны\n"
        "- особенности кроя или материала если есть\n\n"
        "Отвечай только описанием, без заголовков и вводных фраз."
    )
    return _chat([
        {"role": "system", "content": "Ты помощник по шитью и выкройкам. Отвечай на русском."},
        {"role": "user", "content": prompt},
    ])


def search_patterns(index: dict, query: str) -> str:
    """Передаёт запрос и все описания в LLM, получает рекомендацию."""
    if not index:
        return "Память пуста — сначала проиндексируйте папку."

    catalog = "\n\n".join(
        f"Файл: {name}\nОписание: {data['description']}"
        for name, data in index.items()
    )

    prompt = (
        f"Каталог выкроек:\n\n{catalog}\n\n"
        f"Запрос: {query}\n\n"
        "Подбери подходящие выкройки из каталога."
    )
    return _chat([
        {"role": "system", "content": SYSTEM_SEARCH},
        {"role": "user", "content": prompt},
    ])


# ---------------------------------------------------------------------------
# Индекс (долговременная память)
# ---------------------------------------------------------------------------

def load_index() -> dict:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return {}


def save_index(index: dict) -> None:
    INDEX_FILE.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def index_docs(index: dict) -> dict:
    """Сканирует DOCS_DIR, индексирует новые PDF, возвращает обновлённый индекс."""
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"  Нет PDF-файлов в {DOCS_DIR}")
        return index

    for pdf in pdfs:
        name = pdf.name
        if name in index:
            print(f"  [=] {name} (уже в памяти)")
            continue

        print(f"  [+] {name} — извлекаю текст...", end=" ", flush=True)
        text = extract_text(pdf)
        if not text:
            print("пропущен (пустой текст)")
            continue

        print("описываю через LLM...", end=" ", flush=True)
        description = describe_pattern(name, text)
        index[name] = {
            "description": description,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_index(index)
        print("готово")

    return index


# ---------------------------------------------------------------------------
# CLI чат
# ---------------------------------------------------------------------------

def check_server() -> bool:
    try:
        with httpx.Client(timeout=5) as client:
            return client.get(f"{OLLAMA_URL}/api/tags").status_code == 200
    except httpx.ConnectError:
        return False


def main() -> None:
    if not check_server():
        print(f"Ollama не запущен. Выполните: ollama serve")
        sys.exit(1)

    print(f"Модель: {MODEL} | API: {OLLAMA_URL}")
    print(f"Папка с выкройками: {DOCS_DIR}")

    # Загрузка + индексирование
    print(f"\nИндексирование {DOCS_DIR}...")
    index = load_index()
    index = index_docs(index)
    print(f"В памяти {len(index)} выкроек. Индекс: {INDEX_FILE}\n")

    if not index:
        print("Нет выкроек для поиска. Завершение.")
        return

    # Интерактивный чат
    print("Спросите что хотите сшить (или 'выход' для выхода):\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not query:
            continue
        if query.lower() in {"выход", "exit", "quit", "q"}:
            print("Выход.")
            break

        print()
        answer = search_patterns(index, query)
        print(answer)
        print()


if __name__ == "__main__":
    main()
