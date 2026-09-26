"""Client for text-cleaner-mcp: cleans PDF text and summarises the sewing item."""
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_PATH = Path(__file__).parent.parent.parent / "text-cleaner-mcp" / "server.py"


async def clean_and_summarize(text: str) -> dict:
    """Call text-cleaner-mcp and return {cleaned_text, item_type, summary}."""
    extra_env = {}
    if key := os.environ.get("DEEPSEEK_API_KEY"):
        extra_env["DEEPSEEK_API_KEY"] = key
    params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_PATH)], env=extra_env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("clean_and_summarize", {"text": text})
            if result.is_error:
                raise RuntimeError(f"clean_and_summarize error: {result.content}")
            for item in result.content:
                raw = getattr(item, "text", None)
                if raw:
                    return json.loads(raw)
    return {}
