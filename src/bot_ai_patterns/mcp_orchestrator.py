"""
Multi-MCP orchestrator — координирует four MCP-серверов для обработки коллекции
швейных выкроек и публикации результатов на Telegraph.

Серверы:
  first-mcp        — scan_for_description, pdf_extract_first_page
  text-cleaner-mcp — is_sewing_pattern, clean_and_summarize
  tagger-mcp       — extract_tags
  telegraph-mcp    — upload_photo, create_page
"""
import base64
import json
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_FIRST_MCP = Path(__file__).parent.parent.parent / "first-mcp" / "server.py"
_CLEANER_MCP = Path(__file__).parent.parent.parent / "text-cleaner-mcp" / "server.py"
_TAGGER_MCP = Path(__file__).parent.parent.parent / "tagger-mcp" / "server.py"
_TELEGRAPH_MCP = Path(__file__).parent.parent.parent / "telegraph-mcp" / "server.py"


def _deepseek_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if key := os.environ.get("DEEPSEEK_API_KEY"):
        env["DEEPSEEK_API_KEY"] = key
    return env


async def _call(
    server_path: Path,
    tool: str,
    args: dict,
    env: dict[str, str] | None = None,
) -> dict | None:
    """Spawn a server subprocess, call one tool, return parsed result or None."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=env or {},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.is_error:
                raise RuntimeError(
                    f"[{server_path.parent.name}:{tool}] "
                    f"{getattr(result.content[0], 'text', result.content) if result.content else 'error'}"
                )
            for item in result.content:
                raw = getattr(item, "text", None)
                if raw:
                    return json.loads(raw)
    return None


async def process_collection(
    files: list[tuple[bytes, str, str]],
    *,
    max_pages: int = 20,
    page_title: str = "Выкройки",
    on_progress: Callable[[str], Awaitable[None]] | None = None,
    on_cover_ready: Callable[[bytes], Awaitable[str]] | None = None,
) -> str:
    """
    Run the full multi-MCP pipeline for a collection of PDF files.

    Routing logic (agent selects tool based on context):
      • Every file → first-mcp:scan_for_description
      • If text found → text-cleaner-mcp:is_sewing_pattern  (route: skip non-patterns)
      • If is pattern → text-cleaner-mcp:clean_and_summarize (route: summarise)
      • If summary    → tagger-mcp:extract_tags              (route: tag)
      • Always        → first-mcp:pdf_extract_first_page
      • If PNG        → telegraph-mcp:upload_photo
      After all files → telegraph-mcp:create_page

    Returns the Telegraph page URL.
    """
    deepseek_env = _deepseek_env()

    async def _progress(msg: str) -> None:
        if on_progress:
            await on_progress(msg)

    main_item: dict | None = None          # first sewing pattern — full pipeline
    attachments: list[str] = []            # all other file names

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for idx, (pdf_bytes, file_name, file_url) in enumerate(files, start=1):
            # Files after the first pattern are attached without processing
            if main_item is not None:
                attachments.append({"name": file_name, "url": file_url})
                await _progress(f"📎 [{idx}/{len(files)}] {file_name}: прикрепляю как доп. файл.")
                continue

            await _progress(f"📄 [{idx}/{len(files)}] {file_name}: сканирую страницы...")

            pdf_path = str(tmp / f"input_{idx}.pdf")
            png_path = str(tmp / f"cover_{idx}.png")
            Path(pdf_path).write_bytes(pdf_bytes)

            # ── Tool 1: first-mcp → scan_for_description ────────────────────
            scan = await _call(
                _FIRST_MCP,
                "scan_for_description",
                {"pdf_path": pdf_path, "max_pages": max_pages},
            )
            raw_text: str = (scan or {}).get("text", "")
            pages_scanned: int = (scan or {}).get("pages_scanned", 0)

            if not raw_text.strip():
                await _progress(
                    f"⚠️ [{idx}/{len(files)}] {file_name}: текст не найден — "
                    "прерываю цепочку для этого файла."
                )
                attachments.append({"name": file_name, "url": file_url})
                continue

            await _progress(
                f"🔍 [{idx}/{len(files)}] {file_name}: "
                f"просканировано {pages_scanned} стр. — определяю тип..."
            )

            # ── Tool 2: text-cleaner-mcp → is_sewing_pattern ────────────────
            detect = await _call(
                _CLEANER_MCP,
                "is_sewing_pattern",
                {"text": raw_text[:3000]},
                env=deepseek_env,
            )
            if not (detect or {}).get("is_pattern", False):
                reason = (detect or {}).get("reason", "")
                await _progress(
                    f"⏭ [{idx}/{len(files)}] {file_name}: не выкройка ({reason}) — "
                    "прерываю цепочку, прикрепляю как файл."
                )
                attachments.append({"name": file_name, "url": file_url})
                continue

            await _progress(f"✂️ [{idx}/{len(files)}] {file_name}: выкройка найдена — генерирую описание...")

            # ── Tool 3: text-cleaner-mcp → clean_and_summarize ──────────────
            cleaned = await _call(
                _CLEANER_MCP,
                "clean_and_summarize",
                {"text": raw_text},
                env=deepseek_env,
            )
            summary: str = (cleaned or {}).get("summary", "")
            item_type: str = (cleaned or {}).get("item_type", "неизвестно")

            # ── Tool 4: tagger-mcp → extract_tags ───────────────────────────
            tags: dict = {}
            if summary:
                await _progress(f"🏷 [{idx}/{len(files)}] {file_name}: извлекаю теги...")
                tags_result = await _call(
                    _TAGGER_MCP,
                    "extract_tags",
                    {"summary": summary},
                    env=deepseek_env,
                )
                tags = tags_result or {}

            # ── Tool 5: first-mcp → pdf_extract_first_page ──────────────────
            cover_url = ""
            img_result = await _call(
                _FIRST_MCP,
                "pdf_extract_first_page",
                {"pdf_path": pdf_path, "output_path": png_path},
            )
            if img_result and Path(png_path).exists():
                png_bytes_data = Path(png_path).read_bytes()
                if on_cover_ready:
                    # ── Callback: handler sends PNG to Telegram, returns CDN URL ──
                    await _progress(f"🖼 [{idx}/{len(files)}] {file_name}: отправляю обложку в чат...")
                    cover_url = await on_cover_ready(png_bytes_data)
                else:
                    # ── Tool 6: telegraph-mcp → upload_photo (fallback) ───────────
                    await _progress(f"🖼 [{idx}/{len(files)}] {file_name}: загружаю обложку...")
                    png_b64 = base64.b64encode(png_bytes_data).decode()
                    upload = await _call(
                        _TELEGRAPH_MCP,
                        "upload_photo",
                        {"image_b64": png_b64},
                    )
                    cover_url = (upload or {}).get("url", "")

            main_item = {
                "title": file_name,
                "item_type": item_type,
                "summary": summary,
                "tags_json": json.dumps(tags, ensure_ascii=False),
                "cover_url": cover_url,
                "file_name": file_name,
                "file_url": file_url,
            }

    if main_item is None:
        raise RuntimeError(
            "Среди загруженных файлов не найдено описание выкройки. "
            "Загрузи PDF с описанием швейного изделия."
        )

    # Merge attachments into the main item
    if attachments:
        main_item["attachments"] = attachments

    await _progress(
        f"📰 Создаю Telegraph-страницу"
        + (f" + {len(attachments)} прикреплённых файл(а)..." if attachments else "...")
    )

    # ── Tool 7: telegraph-mcp → create_page ─────────────────────────────────
    page = await _call(
        _TELEGRAPH_MCP,
        "create_page",
        {
            "title": page_title,
            "items_json": json.dumps([main_item], ensure_ascii=False),
        },
    )
    if not page:
        raise RuntimeError("Не удалось создать страницу Telegraph.")
    return page.get("url", "")
