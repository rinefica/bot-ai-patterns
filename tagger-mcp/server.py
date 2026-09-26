"""tagger-mcp: извлекает структурированные теги швейного изделия из описания через DeepSeek."""
import json
import os
from typing import Annotated

import openai
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

load_dotenv()

mcp = MCPServer("tagger-mcp")

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = """\
Ты — эксперт по классификации швейных изделий.
На основе описания изделия извлеки структурированные теги.

Верни строго JSON без markdown-обёртки:
{
  "clothing_type": "тип одежды (платье / брюки / юбка / блуза / жакет / пальто / ...)",
  "fabric": ["рекомендуемые ткани, например: хлопок, лён, шёлк, трикотаж, ..."],
  "silhouette": "силуэт (прямой / А-силуэт / приталенный / оверсайз / трапеция / ...)",
  "season": ["подходящие сезоны: лето / зима / весна / осень / демисезон"],
  "length": "длина (мини / до колена / миди / макси / в пол / ...)",
  "style": "стиль (casual / деловой / вечерний / спортивный / романтичный / ...)",
  "closure": "застёжка (молния / пуговицы / крючки / без застёжки / ...)",
  "additional_tags": ["прочие характеристики: рукав, вырез, карманы, отделка и т.д."]
}

Если параметр невозможно определить из описания — используй "не определено".
"""


class TagResult(BaseModel):
    clothing_type: str
    fabric: list[str]
    silhouette: str
    season: list[str]
    length: str
    style: str
    closure: str
    additional_tags: list[str]


def _get_client() -> openai.OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return openai.OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)


@mcp.tool(
    name="extract_tags",
    description=(
        "Extracts structured garment tags (clothing type, fabric, silhouette, season, "
        "length, style, closure, additional tags) from a sewing item summary using DeepSeek."
    ),
)
def extract_tags(
    summary: Annotated[str, Field(description="Professional description of the sewing item.")],
) -> TagResult:
    if not summary.strip():
        raise ValueError("Summary is empty")

    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Описание изделия:\n\n{summary}"},
        ],
        max_tokens=512,
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek returned invalid JSON: {exc}") from exc

    return TagResult(
        clothing_type=data.get("clothing_type", "не определено"),
        fabric=data.get("fabric", []),
        silhouette=data.get("silhouette", "не определено"),
        season=data.get("season", []),
        length=data.get("length", "не определено"),
        style=data.get("style", "не определено"),
        closure=data.get("closure", "не определено"),
        additional_tags=data.get("additional_tags", []),
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
