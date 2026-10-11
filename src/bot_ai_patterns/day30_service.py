"""
День 30: Локальная LLM как приватный HTTP-сервис.

Запуск:
  uv run uvicorn bot_ai_patterns.day30_service:app --host 0.0.0.0 --port 8080

Эндпоинты:
  GET  /health   — статус сервиса и модели
  GET  /models   — список доступных моделей
  POST /chat     — отправить сообщение, получить ответ

Аутентификация: заголовок X-API-Key
Rate limit: 10 запросов / минута на ключ
Max context: 4000 символов на запрос
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
API_KEY = os.getenv("SERVICE_API_KEY", "local-dev-key")

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))       # запросов в минуту
RATE_WINDOW = 60                                         # секунд
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "4000"))    # символов

GENERATE_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 512,
    "num_ctx": 4096,
    "repeat_penalty": 1.1,
}

# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per API key)
# ---------------------------------------------------------------------------

_rate_store: dict[str, deque] = defaultdict(deque)


def check_rate_limit(key: str) -> None:
    now = time.time()
    window = _rate_store[key]
    # Удаляем устаревшие отметки
    while window and window[0] < now - RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        reset_in = int(RATE_WINDOW - (now - window[0]))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT} req/min). Retry in {reset_in}s.",
        )
    window.append(now)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(api_key_header)) -> str:
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key


# ---------------------------------------------------------------------------
# Модели запроса / ответа
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="Сообщение пользователя")
    system: str = Field(default="Ты полезный ассистент. Отвечай на русском.", description="Системный промпт")
    model: str = Field(default=DEFAULT_MODEL, description="Имя модели Ollama")


class ChatResponse(BaseModel):
    reply: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed: float


class HealthResponse(BaseModel):
    status: str
    ollama: str
    model: str
    rate_limit: str
    max_context: int


# ---------------------------------------------------------------------------
# Ollama клиент
# ---------------------------------------------------------------------------

def ollama_chat(model: str, system: str, message: str) -> dict:
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
                "stream": False,
                "options": GENERATE_OPTIONS,
            },
        )
        resp.raise_for_status()
        return resp.json()


def ollama_available() -> bool:
    try:
        with httpx.Client(timeout=3) as client:
            return client.get(f"{OLLAMA_URL}/api/tags").status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local LLM Service",
    description="Приватный AI-сервис на базе Ollama + qwen2.5:7b",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        ollama="available" if ollama_available() else "unavailable",
        model=DEFAULT_MODEL,
        rate_limit=f"{RATE_LIMIT} req/min",
        max_context=MAX_CONTEXT,
    )


@app.get("/models")
def models(key: str = Depends(require_api_key)):
    with httpx.Client(timeout=5) as client:
        resp = client.get(f"{OLLAMA_URL}/api/tags")
        resp.raise_for_status()
    return {
        "models": [
            {"name": m["name"], "size_gb": round(m["size"] / 1e9, 1)}
            for m in resp.json().get("models", [])
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, key: str = Depends(require_api_key)):
    # Rate limit
    check_rate_limit(key)

    # Max context
    if len(body.message) > MAX_CONTEXT:
        raise HTTPException(
            status_code=413,
            detail=f"Message too long: {len(body.message)} chars (max {MAX_CONTEXT}).",
        )

    # Запрос к Ollama
    t0 = time.perf_counter()
    try:
        data = ollama_chat(body.model, body.system, body.message)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    elapsed = time.perf_counter() - t0

    return ChatResponse(
        reply=data["message"]["content"],
        model=body.model,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        elapsed=round(elapsed, 2),
    )
