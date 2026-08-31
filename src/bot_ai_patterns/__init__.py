import os
import openai
from dotenv import load_dotenv

load_dotenv()

YANDEX_CLOUD_FOLDER = "b1gesnd8o5f6co3dkvij"
YANDEX_CLOUD_MODEL = "qwen3.6-35b-a3b/latest"
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"


def get_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=os.environ["YANDEX_CLOUD_API_KEY"],
        base_url=YANDEX_BASE_URL,
    )


def main() -> None:
    text = input("Введите текст: ")
    client = get_client()
    model_uri = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}"
    response = client.chat.completions.create(
        model=model_uri,
        messages=[{"role": "user", "content": text}],
    )
    print(response.choices[0].message.content)
