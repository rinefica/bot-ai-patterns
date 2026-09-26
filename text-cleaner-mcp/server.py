"""text-cleaner-mcp: очищает сырой PDF-текст и генерирует описание швейного изделия через DeepSeek."""
import json
import os
from typing import Annotated

import openai
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

load_dotenv()

mcp = MCPServer("text-cleaner-mcp")

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = """\
Ты — эксперт по швейному делу и текстильной промышленности.
Тебе передаётся текст, извлечённый из PDF-файла с описанием швейного изделия или выкройки.

Твои задачи:
1. Очисти текст: убери номера страниц, колонтитулы, лишние пробелы и артефакты PDF-парсинга.
2. Определи предмет пошива (платье, брюки, юбка, блуза и т.д.).
3. Напиши краткое профессиональное описание изделия на русском языке (3-5 предложений):
   укажи тип изделия, его особенности кроя, назначение и характерные детали.

Верни строго JSON без markdown-обёртки:
{
  "cleaned_text": "очищенный текст без мусора",
  "item_type": "тип изделия одним словом или коротко",
  "summary": "профессиональное описание 3-5 предложений"
}
"""


class SummaryResult(BaseModel):
    cleaned_text: str
    item_type: str
    summary: str


def _get_client() -> openai.OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return openai.OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)


@mcp.tool(
    name="clean_and_summarize",
    description=(
        "Cleans raw PDF text and generates a professional sewing item description "
        "using DeepSeek. Returns cleaned text, detected item type, and a summary."
    ),
)
def clean_and_summarize(
    text: Annotated[str, Field(description="Raw text extracted from a PDF sewing pattern or garment description.")],
) -> SummaryResult:
    if not text.strip():
        raise ValueError("Input text is empty")

    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Текст из PDF:\n\n{text[:4000]}"},
        ],
        max_tokens=2048,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek returned invalid JSON: {exc}") from exc

    return SummaryResult(
        cleaned_text=data.get("cleaned_text", text),
        item_type=data.get("item_type", "неизвестно"),
        summary=data.get("summary", ""),
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
