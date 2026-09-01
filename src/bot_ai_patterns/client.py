import os

import openai

from bot_ai_patterns.config import YANDEX_BASE_URL


def get_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=os.environ["YANDEX_CLOUD_API_KEY"],
        base_url=YANDEX_BASE_URL,
    )
