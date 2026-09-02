import html
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from bot_ai_patterns.client import get_client
from bot_ai_patterns.strategies import STRATEGIES, StrategyResult
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
    .strategy {{ background: #fff; border-radius: 8px; padding: 20px 24px;
                 margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
    .strategy h2 {{ margin: 0 0 16px; font-size: 1.1rem; color: #1a73e8; }}
    .section h3 {{ font-size: .9rem; text-transform: uppercase; letter-spacing: .05em;
                   color: #888; margin: 16px 0 6px; }}
    .content {{ white-space: pre-wrap; line-height: 1.6; font-size: .95rem; }}
    .error {{ color: #c62828; font-style: italic; }}
  </style>
</head>
<body>
  <h1>Запрос: {query}</h1>
  {body}
</body>
</html>"""


def _strategy_to_html(result: StrategyResult) -> str:
    inner = f'<div class="strategy"><h2>{html.escape(result.name)}</h2>'
    if result.error:
        inner += f'<p class="error">Ошибка: {html.escape(result.error)}</p>'
    else:
        for label, content in result.sections.items():
            inner += (
                f'<div class="section"><h3>{html.escape(label)}</h3>'
                f'<div class="content">{content}</div></div>'
            )
    return inner + "</div>"


def _save_and_open(user_message: str, results: dict[str, StrategyResult]) -> None:
    body = "\n".join(_strategy_to_html(results[s.name]) for s in STRATEGIES)
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
    print(f"\n{'=' * 60}")
    print(f"  Стратегия: {result.name}")
    print(f"{'=' * 60}")
    if result.error:
        print(f"Ошибка: {result.error}")
        return
    for label, content in result.sections.items():
        print(f"\n[{label}]\n{content}")


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

        print(f"\nЗапускаю {len(STRATEGIES)} стратегии параллельно...\n")
        results: dict[str, StrategyResult] = {}
        with ThreadPoolExecutor(max_workers=len(STRATEGIES)) as executor:
            futures = {}
            for strategy in STRATEGIES:
                print(f"  ▶ {strategy.name}...", flush=True)
                futures[executor.submit(strategy.run, client, user_message)] = strategy.name

            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
                print(f"  ✓ {name} завершена", flush=True)

        print()
        for strategy in STRATEGIES:
            _print_result(results[strategy.name])

        _save_and_open(user_message, results)
        print()
