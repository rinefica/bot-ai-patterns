import asyncio
from dataclasses import dataclass, field

import openai

from bot_ai_patterns.config import MAX_TOKENS, MODEL_URI
from bot_ai_patterns.prompts import STEP_BY_STEP_PROMPT

TEMPERATURES = [0.0, 0.7, 1.0]


@dataclass
class StrategyResult:
    sections: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def _request(client: openai.OpenAI, user_message: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=MODEL_URI,
        messages=[
            {"role": "system", "content": STEP_BY_STEP_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```html"):
        content = content[len("```html"):].lstrip("\n")
    if content.endswith("```"):
        content = content[:-3].rstrip("\n")
    return content


async def run(client: openai.OpenAI, user_message: str) -> StrategyResult:
    try:
        results = await asyncio.gather(
            *[asyncio.to_thread(_request, client, user_message, t) for t in TEMPERATURES]
        )
        sections = {
            f"Вариант {i + 1} (t={t})": text
            for i, (t, text) in enumerate(zip(TEMPERATURES, results))
        }
        return StrategyResult(sections=sections)
    except Exception as e:
        return StrategyResult(error=str(e))
