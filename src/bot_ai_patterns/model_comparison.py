import asyncio
import time
from dataclasses import dataclass

import openai

from bot_ai_patterns.config import COMPARE_MODELS, MAX_TOKENS
from bot_ai_patterns.prompts import STEP_BY_STEP_PROMPT


@dataclass
class ModelResult:
    label: str
    uri: str
    response: str
    elapsed_sec: float
    prompt_tokens: int
    completion_tokens: int
    cost_rub: float
    error: str | None = None


def _request(client: openai.OpenAI, model_info: dict, user_message: str) -> ModelResult:
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_info["uri"],
            messages=[
                {"role": "system", "content": STEP_BY_STEP_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.perf_counter() - start

        content = resp.choices[0].message.content or ""
        content = content.strip()
        if content.startswith("```html"):
            content = content[len("```html"):].lstrip("\n")
        if content.endswith("```"):
            content = content[:-3].rstrip("\n")

        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        cost = total_tokens / 1000 * model_info["price_per_1k"]

        return ModelResult(
            label=model_info["label"],
            uri=model_info["uri"],
            response=content,
            elapsed_sec=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_rub=cost,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return ModelResult(
            label=model_info["label"],
            uri=model_info["uri"],
            response="",
            elapsed_sec=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            cost_rub=0.0,
            error=str(e),
        )


async def compare_models(client: openai.OpenAI, user_message: str) -> list[ModelResult]:
    return await asyncio.gather(
        *[asyncio.to_thread(_request, client, m, user_message) for m in COMPARE_MODELS]
    )


def format_summary(results: list[ModelResult]) -> str:
    rows = []
    for r in results:
        if r.error:
            rows.append(f"{r.label}: ошибка — {r.error}")
        else:
            rows.append(
                f"{r.label}: {r.elapsed_sec:.1f}с | "
                f"{r.prompt_tokens}+{r.completion_tokens} tok | "
                f"{r.cost_rub:.4f} ₽"
            )
    return "\n".join(rows)
