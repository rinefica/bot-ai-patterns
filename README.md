# bot-ai-patterns

Примеры паттернов для AI-ботов на базе Anthropic Claude.

## Быстрый старт

```bash
# Установить uv (если нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установить зависимости
uv sync

# Задать API-ключ в .envrc
# ANTHROPIC_API_KEY=sk-...

# Запустить
uv run bot-ai-patterns
```

## Структура

```
src/bot_ai_patterns/   # основной пакет
pyproject.toml         # зависимости и метаданные
.envrc                 # переменные окружения (не коммитить)
```
