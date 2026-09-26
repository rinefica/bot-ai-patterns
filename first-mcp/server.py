from pathlib import Path
from typing import Annotated

import pymupdf as fitz
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field
from pypdf import PdfReader

mcp = MCPServer("first-mcp")


class PdfTextResult(BaseModel):
    pdf_path: str
    total_pages: int
    page: int | None
    text: str


class PdfPageImageResult(BaseModel):
    pdf_path: str
    output_path: str
    dpi: int
    width_px: int
    height_px: int


class ScanResult(BaseModel):
    pdf_path: str
    total_pages: int
    pages_scanned: int
    text: str


@mcp.tool(
    name="pdf_read_text",
    description="Extract text content from a PDF file — either a single page or all pages.",
)
def pdf_read_text(
    pdf_path: Annotated[str, Field(description="Absolute path to the PDF file.")],
    page: Annotated[
        int | None,
        Field(description="Page number to read (1-based). Omit to read all pages.", ge=1),
    ] = None,
) -> PdfTextResult:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    reader = PdfReader(str(path))
    total = len(reader.pages)

    if page is not None:
        if page > total:
            raise ValueError(f"Page {page} out of range (document has {total} pages)")
        text = reader.pages[page - 1].extract_text() or ""
    else:
        parts = []
        for i, p in enumerate(reader.pages, start=1):
            t = p.extract_text() or ""
            parts.append(f"--- Page {i} ---\n{t.strip()}")
        text = "\n\n".join(parts)

    return PdfTextResult(
        pdf_path=str(path),
        total_pages=total,
        page=page,
        text=text.strip(),
    )


@mcp.tool(
    name="scan_for_description",
    description=(
        "Scan PDF pages one by one, accumulating text until min_words is reached "
        "or max_pages is exhausted. Returns all accumulated text and scan metadata."
    ),
)
def scan_for_description(
    pdf_path: Annotated[str, Field(description="Absolute path to the PDF file.")],
    max_pages: Annotated[
        int, Field(description="Maximum number of pages to scan.", ge=1, le=50)
    ] = 20,
    min_words: Annotated[
        int,
        Field(description="Stop scanning early once this many words are accumulated.", ge=50),
    ] = 300,
) -> ScanResult:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    reader = PdfReader(str(path))
    total = len(reader.pages)
    limit = min(max_pages, total)

    parts: list[str] = []
    word_count = 0

    for i in range(limit):
        page_text = (reader.pages[i].extract_text() or "").strip()
        if page_text:
            parts.append(f"--- Страница {i + 1} ---\n{page_text}")
            word_count += len(page_text.split())
        if word_count >= min_words:
            break

    return ScanResult(
        pdf_path=str(path),
        total_pages=total,
        pages_scanned=len(parts),
        text="\n\n".join(parts).strip(),
    )


@mcp.tool(
    name="pdf_extract_first_page",
    description="Render the first page of a PDF as a PNG image and save it to disk.",
)
def pdf_extract_first_page(
    pdf_path: Annotated[str, Field(description="Absolute path to the source PDF file.")],
    output_path: Annotated[
        str, Field(description="Absolute path where the PNG will be saved (must end with .png).")
    ],
    dpi: Annotated[
        int, Field(description="Render resolution in DPI.", ge=72, le=600)
    ] = 150,
) -> PdfPageImageResult:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    doc = fitz.open(str(path))
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

    zoom = dpi / 72
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out))

    return PdfPageImageResult(
        pdf_path=str(path),
        output_path=str(out),
        dpi=dpi,
        width_px=pix.width,
        height_px=pix.height,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
