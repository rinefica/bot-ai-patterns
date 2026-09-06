import asyncio
import html
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from bot_ai_patterns.client import get_client
from bot_ai_patterns.strategies import StrategyResult, run as run_strategy
from bot_ai_patterns.utils import sanitize

load_dotenv()

RESPONSES_DIR = Path("responses")

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 900px; margin: 40px auto; padding: 0 20px;
            background: #f5f5f5; color: #222; }}
    h1 {{ font-size: 1.4rem; background: #222; color: #fff;
          padding: 16px 20px; border-radius: 8px; }}
    .variant {{ background: #fff; border-radius: 8px; padding: 20px 24px;
                margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    .variant h2 {{ margin: 0 0 16px; font-size: 1.1rem; color: #1a73e8; }}
    .content {{ white-space: pre-wrap; line-height: 1.6; font-size: .95rem; }}
    .error {{ color: #c62828; font-style: italic; }}
  </style>
</head>
<body>
  <h1>Запрос: {query}</h1>
  {body}
</body>
</html>"""


def _result_to_html(result: StrategyResult) -> str:
    if result.error:
        return f'<div class="variant"><p class="error">Ошибка: {html.escape(result.error)}</p></div>'
    parts = []
    for label, content in result.sections.items():
        parts.append(
            f'<div class="variant"><h2>{html.escape(label)}</h2>'
            f'<div class="content">{content}</div></div>'
        )
    return "\n".join(parts)


def _save_and_open(user_message: str, result: StrategyResult) -> None:
    body = _result_to_html(result)
    page = HTML_TEMPLATE.format(
        title=f"Запрос: {user_message}",
        query=html.escape(user_message),
        body=body,
    )
    RESPONSES_DIR.mkdir(exist_ok=True)
    filepath = RESPONSES_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    filepath.write_text(page, encoding="utf-8")
    subprocess.run(["open", str(filepath)])


def _print_result(result: StrategyResult) -> None:
    if result.error:
        print(f"Ошибка: {result.error}")
        return
    for label, content in result.sections.items():
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        print(content)


def main() -> None:
    client = get_client()
    print("Введите задачу (или 'exit' для выхода).\n")

    while True:
        raw = input("Вы: ").strip()
        if raw.lower() in ("exit", "quit", "стоп", "выход"):
            break
        user_message = sanitize(raw)
        if not user_message:
            continue

        print("\nЗапускаю 3 варианта параллельно...\n")
        result = asyncio.run(run_strategy(client, user_message))
        _print_result(result)
        _save_and_open(user_message, result)
        print()
