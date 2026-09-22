DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_URI = "deepseek-chat"
MAX_TOKENS = 1024
MAX_RETRIES = 3

# Лимит контекста DeepSeek-V3 (токены)
CONTEXT_LIMIT = 64000

# Порог предупреждения — 80% от лимита
CONTEXT_WARN_THRESHOLD = int(CONTEXT_LIMIT * 0.8)

# Цена основной модели ($ за 1000 токенов, output)
MAIN_MODEL_PRICE_PER_1K = 0.28

# Стратегии управления контекстом
# Sliding Window: число сообщений в скользящем окне
WINDOW_SIZE = 6
# Sticky Facts: размер окна последних сообщений + блок фактов
FACTS_WINDOW_SIZE = 6
