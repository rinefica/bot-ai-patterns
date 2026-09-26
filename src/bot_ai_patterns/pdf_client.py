"""Client for the first-mcp MCP server — PDF processing."""
import json
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_PATH = Path(__file__).parent.parent.parent / "first-mcp" / "server.py"


async def process_pdf(pdf_bytes: bytes) -> tuple[bytes, str]:
    """Process PDF via first-mcp MCP server.

    Calls:
      - pdf_extract_first_page → renders page 1 as PNG
      - pdf_read_text          → extracts text from pages 1-5

    Returns:
        (png_bytes, text) — PNG bytes of the first page and
        concatenated text from the first 5 pages.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pdf_path = str(tmp / "input.pdf")
        png_path = str(tmp / "page1.png")

        (tmp / "input.pdf").write_bytes(pdf_bytes)

        params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_PATH)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Text for pages 1-5
                texts: list[str] = []
                for page_num in range(1, 6):
                    result = await session.call_tool(
                        "pdf_read_text",
                        {"pdf_path": pdf_path, "page": page_num},
                    )
                    if result.is_error:
                        break
                    for item in result.content:
                        raw = getattr(item, "text", None)
                        if raw:
                            try:
                                data = json.loads(raw)
                                page_text = data.get("text", "").strip()
                                if page_text:
                                    texts.append(f"--- Страница {page_num} ---\n{page_text}")
                            except (json.JSONDecodeError, AttributeError):
                                pass
                            break

                # First page → PNG
                await session.call_tool(
                    "pdf_extract_first_page",
                    {"pdf_path": pdf_path, "output_path": png_path},
                )

                png_bytes = Path(png_path).read_bytes() if Path(png_path).exists() else b""

        return png_bytes, "\n\n".join(texts)
