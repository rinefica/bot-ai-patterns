YANDEX_CLOUD_FOLDER = "b1gesnd8o5f6co3dkvij"
YANDEX_CLOUD_MODEL = "yandexgpt-5-pro/latest"
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
MODEL_URI = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}"
MAX_TOKENS = 1024
MAX_RETRIES = 3


def _uri(model: str) -> str:
    return f"gpt://{YANDEX_CLOUD_FOLDER}/{model}"


# Модели для сравнения: слабая → средняя → сильная
# Цены: https://yandex.cloud/ru/prices#foundation-models (руб. за 1000 токенов)
COMPARE_MODELS = [
    {"label": "🟢 Lite",     "uri": _uri("yandexgpt-lite/latest"), "price_per_1k": 0.20},
    {"label": "🟡 GPT-4 Pro", "uri": _uri("yandexgpt/latest"),     "price_per_1k": 1.20},
    {"label": "🔴 GPT-5 Pro", "uri": _uri("yandexgpt-5-pro/latest"), "price_per_1k": 6.00},
]
