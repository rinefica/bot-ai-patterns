import os

import openai

from bot_ai_patterns.config import DEEPSEEK_BASE_URL


def get_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
    )
