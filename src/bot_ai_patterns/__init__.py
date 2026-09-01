import json

import openai
from dotenv import load_dotenv

from bot_ai_patterns.client import get_client
from bot_ai_patterns.config import MAX_RETRIES, MAX_TOKENS, YANDEX_CLOUD_FOLDER, YANDEX_CLOUD_MODEL
from bot_ai_patterns.prompts import SYSTEM_PROMPT

load_dotenv()


def _sanitize(s: str) -> str:
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _ask_model(client: openai.OpenAI, messages: list[dict]) -> dict | None:
    model_uri = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}"
    for attempt in range(MAX_RETRIES):
        response = client.chat.completions.create(
            model=model_uri,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        content = response.choices[0].message.content
        if content is None:
            print(f"Пустой ответ от модели (попытка {attempt + 1}/{MAX_RETRIES}), повторяю...")
            continue
        raw = _sanitize(content)
        try:
            return json.loads(raw), raw
        except json.JSONDecodeError:
            print(f"Невалидный JSON (попытка {attempt + 1}/{MAX_RETRIES}), повторяю...")
    return None, None


def _print_response(data: dict) -> None:
    print(f"\nОбъяснение: {data['explanation']}")
    print(f"Пример: {data['example']}")
    print(f"Вопрос: {data['check_question']}")
    if data.get("tip"):
        print(f"Совет: {data['tip']}")
    print()


def _is_session_complete(data: dict) -> bool:
    return data.get("session_complete") is True or data.get("session_complete") == "true"


def main() -> None:
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        text = input("Вы: ")
        messages.append({"role": "user", "content": _sanitize(text)})

        data, raw = _ask_model(client, messages)
        if data is None:
            print("Не удалось получить валидный ответ от модели.")
            break

        messages.append({"role": "assistant", "content": raw})
        _print_response(data)

        if _is_session_complete(data):
            print("Сессия завершена. Удачи в обучении!")
            break
