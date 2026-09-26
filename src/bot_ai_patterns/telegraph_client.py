"""Client for telegraph-mcp: creates Telegraph pages for sewing pattern collections."""
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_PATH = Path(__file__).parent.parent.parent / "telegraph-mcp" / "server.py"


async def _call(tool: str, args: dict) -> dict:
    params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.is_error:
                raise RuntimeError(f"{tool} error: {result.content}")
            for item in result.content:
                raw = getattr(item, "text", None)
                if raw:
                    return json.loads(raw)
    return {}


async def upload_photo(png_bytes: bytes) -> str:
    """Upload PNG to Telegraph, return public URL."""
    import base64
    b64 = base64.b64encode(png_bytes).decode()
    result = await _call("upload_photo", {"image_b64": b64})
    return result.get("url", "")


async def create_page(title: str, items: list[dict]) -> str:
    """Create a Telegraph page with pattern items. Returns page URL."""
    result = await _call("create_page", {
        "title": title,
        "items_json": json.dumps(items, ensure_ascii=False),
    })
    return result.get("url", "")
