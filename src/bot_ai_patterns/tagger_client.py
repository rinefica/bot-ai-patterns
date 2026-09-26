"""Client for tagger-mcp: extracts structured garment tags from a sewing item summary."""
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_PATH = Path(__file__).parent.parent.parent / "tagger-mcp" / "server.py"


async def extract_tags(summary: str) -> dict:
    """Call tagger-mcp and return structured tag dict."""
    extra_env = {}
    if key := os.environ.get("DEEPSEEK_API_KEY"):
        extra_env["DEEPSEEK_API_KEY"] = key
    params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_PATH)], env=extra_env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("extract_tags", {"summary": summary})
            if result.is_error:
                raise RuntimeError(f"extract_tags error: {result.content}")
            for item in result.content:
                raw = getattr(item, "text", None)
                if raw:
                    return json.loads(raw)
    return {}
