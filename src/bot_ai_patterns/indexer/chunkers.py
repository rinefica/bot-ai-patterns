"""Two chunking strategies: fixed-size and structure-based."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str          # absolute file path
    title: str           # file name
    section: str         # heading / class / function / "" for fixed
    strategy: str        # "fixed" | "structure"
    char_start: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def char_len(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking with overlap
# ---------------------------------------------------------------------------

class FixedSizeChunker:
    """Split text into fixed-size character windows with overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> list[Chunk]:
        title = Path(source).name
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        pos = 0
        idx = 0
        while pos < len(text):
            end = pos + self.chunk_size
            window = text[pos:end]
            if window.strip():
                chunks.append(Chunk(
                    chunk_id=f"fixed::{title}::{idx}",
                    text=window,
                    source=source,
                    title=title,
                    section="",
                    strategy="fixed",
                    char_start=pos,
                ))
                idx += 1
            pos += step
        return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Structure-based chunking
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r'^(#{1,6})\s+(.+)', re.MULTILINE)
_PY_BLOCK = re.compile(
    r'^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)',
    re.MULTILINE,
)
# Leading "09 " / "09\n" page numbers in PDF text
_PDF_PAGE_NUM = re.compile(r'^\d{1,3}\s+')
# ALL-CAPS heading: ≥4 caps letters, possibly Cyrillic, spaces allowed
_PDF_CAPS_HEADING = re.compile(r'^[А-ЯЁA-Z][А-ЯЁA-Z\s\-/,]{3,}$')
# Numbered step like "01. " or "1. "
_PDF_STEP = re.compile(r'^(\d{1,2})\.\s+(.{4,})')


def _pdf_label(text: str, fallback_idx: int, current_heading: str) -> str:
    """Extract a human-readable section label from a PDF paragraph."""
    # Work line by line; skip blank lines and lone page-number lines
    lines = [ln.strip() for ln in text.splitlines()]
    meaningful: list[str] = []
    for ln in lines:
        if not ln:
            continue
        # Skip bare page numbers like "09" or "10"
        if re.fullmatch(r'\d{1,3}', ln):
            continue
        # Strip leading "09 SOME TEXT" → "SOME TEXT"
        ln = _PDF_PAGE_NUM.sub("", ln).strip()
        if ln:
            meaningful.append(ln)

    if not meaningful:
        return current_heading or f"раздел {fallback_idx + 1}"

    first = meaningful[0]

    # 1. ALL-CAPS heading line (section title)
    if _PDF_CAPS_HEADING.match(first) and len(first) < 80:
        return first.rstrip(".")

    # 2. Numbered step "01. Стачать боковые срезы"
    m = _PDF_STEP.match(first)
    if m:
        step_num = m.group(1).lstrip("0") or "0"
        step_text = m.group(2)[:55].rstrip(" .,")
        prefix = f"Шаг {step_num}: {step_text}"
        return prefix if not current_heading else f"{current_heading} — шаг {step_num}"

    # 3. Inherit current heading + first words of content
    snippet = first[:55].rstrip(" .,")
    if current_heading and current_heading.lower() not in snippet.lower():
        return f"{current_heading} — {snippet}"

    return snippet


class StructureChunker:
    """
    Split by document structure:
    - Markdown: split on headings (##, ###, …)
    - Python: split on top-level class / def blocks
    - Other: fallback to paragraph splitting
    """

    def __init__(self, min_chunk: int = 100, max_chunk: int = 2000):
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk

    def chunk(self, text: str, source: str) -> list[Chunk]:
        title = Path(source).name
        ext = Path(source).suffix.lower()

        if ext == ".md":
            segments = self._split_markdown(text)
        elif ext == ".py":
            segments = self._split_python(text)
        elif ext == ".pdf":
            segments = self._split_pdf(text)
        else:
            segments = self._split_paragraphs(text)

        chunks: list[Chunk] = []
        for idx, (section, body) in enumerate(segments):
            # If body is too large, sub-split it
            for sub_idx, sub_text in enumerate(self._sub_split(body)):
                if not sub_text.strip():
                    continue
                section_label = section if sub_idx == 0 else f"{section} [part {sub_idx+1}]"
                chunks.append(Chunk(
                    chunk_id=f"struct::{title}::{idx}::{sub_idx}",
                    text=sub_text,
                    source=source,
                    title=title,
                    section=section_label,
                    strategy="structure",
                ))
        return chunks

    # ------------------------------------------------------------------

    def _split_markdown(self, text: str) -> list[tuple[str, str]]:
        """Split on any heading level."""
        positions = [m.start() for m in _MD_HEADING.finditer(text)]
        if not positions:
            return [("", text)]

        segments: list[tuple[str, str]] = []
        # Preamble before first heading
        if positions[0] > 0:
            segments.append(("preamble", text[:positions[0]]))

        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            block = text[pos:end]
            heading_match = _MD_HEADING.match(block)
            heading = heading_match.group(2).strip() if heading_match else ""
            body = block[heading_match.end():].strip() if heading_match else block
            segments.append((heading, body))

        return segments

    def _split_python(self, text: str) -> list[tuple[str, str]]:
        """Split on top-level class and def statements."""
        positions = [m.start() for m in _PY_BLOCK.finditer(text)]
        if not positions:
            return [("module", text)]

        segments: list[tuple[str, str]] = []
        if positions[0] > 0:
            segments.append(("module-level", text[:positions[0]]))

        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            block = text[pos:end]
            name_match = re.match(r'(?:async\s+)?(?:class|def)\s+(\w+)', block)
            name = name_match.group(1) if name_match else f"block_{i}"
            segments.append((name, block))

        return segments

    def _split_paragraphs(self, text: str) -> list[tuple[str, str]]:
        """Fallback: split on blank lines."""
        paragraphs = re.split(r'\n{2,}', text)
        return [(f"para_{i}", p) for i, p in enumerate(paragraphs)]

    def _split_pdf(self, text: str) -> list[tuple[str, str]]:
        """Split PDF text on blank lines, then derive an informative section label."""
        paragraphs = re.split(r'\n{2,}', text)
        segments: list[tuple[str, str]] = []
        current_heading = ""
        for i, para in enumerate(paragraphs):
            label = _pdf_label(para, i, current_heading)
            # If the paragraph IS a pure heading (very short, all-caps), carry it
            # forward so the next content chunk inherits it.
            stripped = para.strip()
            if _PDF_CAPS_HEADING.match(stripped) and len(stripped) < 80:
                current_heading = stripped
            segments.append((label, para))
        return segments

    def _sub_split(self, text: str) -> list[str]:
        """If a section is too large, split it at sentence boundaries."""
        if len(text) <= self.max_chunk:
            return [text]
        parts: list[str] = []
        while len(text) > self.max_chunk:
            cut = text.rfind('. ', 0, self.max_chunk)
            if cut == -1:
                cut = self.max_chunk
            parts.append(text[:cut + 1])
            text = text[cut + 1:].lstrip()
        if text:
            parts.append(text)
        return parts
