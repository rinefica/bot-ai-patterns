"""Unit tests for first-mcp/server.py tools (called directly, no MCP transport)."""
import importlib.util
from pathlib import Path

import pymupdf
import pytest

# ---------------------------------------------------------------------------
# Import server module without going through the package system
# ---------------------------------------------------------------------------
_SERVER_FILE = Path(__file__).parent.parent.parent / "first-mcp" / "server.py"
_spec = importlib.util.spec_from_file_location("first_mcp_server", _SERVER_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pdf_read_text = _mod.pdf_read_text
pdf_extract_first_page = _mod.pdf_extract_first_page
scan_for_description = _mod.scan_for_description
PdfTextResult = _mod.PdfTextResult
PdfPageImageResult = _mod.PdfPageImageResult
ScanResult = _mod.ScanResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pdf_3pages(tmp_path) -> Path:
    """3-page PDF with extractable text on each page."""
    doc = pymupdf.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i} content line one\nLine two on page {i}")
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    return path


@pytest.fixture
def pdf_1page(tmp_path) -> Path:
    """Single-page PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Only page")
    path = tmp_path / "one.pdf"
    doc.save(str(path))
    return path


# ---------------------------------------------------------------------------
# pdf_read_text
# ---------------------------------------------------------------------------

class TestPdfReadText:
    def test_returns_pdftext_result(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages), page=1)
        assert isinstance(result, PdfTextResult)

    def test_specific_page_text(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages), page=2)
        assert result.page == 2
        assert "Page 2" in result.text

    def test_total_pages_reported(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages), page=1)
        assert result.total_pages == 3

    def test_all_pages_when_page_is_none(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages))
        assert result.page is None
        assert result.total_pages == 3
        for i in range(1, 4):
            assert f"Page {i}" in result.text

    def test_all_pages_contains_separators(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages))
        assert "--- Page 1 ---" in result.text
        assert "--- Page 2 ---" in result.text

    def test_pdf_path_stored_in_result(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages), page=1)
        assert result.pdf_path == str(pdf_3pages)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pdf_read_text(str(tmp_path / "missing.pdf"))

    def test_raises_value_error_for_page_out_of_range(self, pdf_3pages):
        with pytest.raises(ValueError, match="out of range"):
            pdf_read_text(str(pdf_3pages), page=99)

    def test_last_page_is_valid(self, pdf_3pages):
        result = pdf_read_text(str(pdf_3pages), page=3)
        assert "Page 3" in result.text

    def test_single_page_doc(self, pdf_1page):
        result = pdf_read_text(str(pdf_1page), page=1)
        assert result.total_pages == 1
        assert "Only page" in result.text


# ---------------------------------------------------------------------------
# pdf_extract_first_page
# ---------------------------------------------------------------------------

class TestPdfExtractFirstPage:
    def test_creates_png_file(self, pdf_3pages, tmp_path):
        out = tmp_path / "out.png"
        pdf_extract_first_page(str(pdf_3pages), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_pdfpageimage_result(self, pdf_3pages, tmp_path):
        out = tmp_path / "out.png"
        result = pdf_extract_first_page(str(pdf_3pages), str(out))
        assert isinstance(result, PdfPageImageResult)

    def test_result_fields(self, pdf_3pages, tmp_path):
        out = tmp_path / "out.png"
        result = pdf_extract_first_page(str(pdf_3pages), str(out))
        assert result.dpi == 150
        assert result.width_px > 0
        assert result.height_px > 0
        assert result.output_path == str(out)
        assert result.pdf_path == str(pdf_3pages)

    def test_higher_dpi_produces_larger_image(self, pdf_3pages, tmp_path):
        lo = tmp_path / "lo.png"
        hi = tmp_path / "hi.png"
        pdf_extract_first_page(str(pdf_3pages), str(lo), dpi=72)
        pdf_extract_first_page(str(pdf_3pages), str(hi), dpi=300)
        assert hi.stat().st_size > lo.stat().st_size

    def test_creates_nested_output_directory(self, pdf_3pages, tmp_path):
        out = tmp_path / "a" / "b" / "page.png"
        pdf_extract_first_page(str(pdf_3pages), str(out))
        assert out.exists()

    def test_output_is_valid_png(self, pdf_3pages, tmp_path):
        out = tmp_path / "out.png"
        pdf_extract_first_page(str(pdf_3pages), str(out))
        # PNG files start with the 8-byte PNG signature
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pdf_extract_first_page(
                str(tmp_path / "missing.pdf"),
                str(tmp_path / "out.png"),
            )

    def test_custom_dpi_reflected_in_result(self, pdf_3pages, tmp_path):
        out = tmp_path / "out.png"
        result = pdf_extract_first_page(str(pdf_3pages), str(out), dpi=72)
        assert result.dpi == 72


# ---------------------------------------------------------------------------
# scan_for_description
# ---------------------------------------------------------------------------

@pytest.fixture
def pdf_long(tmp_path) -> Path:
    """10-page PDF with substantial text on each page."""
    doc = pymupdf.open()
    for i in range(1, 11):
        page = doc.new_page()
        words = " ".join([f"word{j}" for j in range(60)])
        page.insert_text((72, 100), f"Page {i}: {words}")
    path = tmp_path / "long.pdf"
    doc.save(str(path))
    return path


class TestScanForDescription:
    def test_returns_scan_result(self, pdf_3pages):
        result = scan_for_description(str(pdf_3pages))
        assert isinstance(result, ScanResult)

    def test_total_pages_correct(self, pdf_3pages):
        result = scan_for_description(str(pdf_3pages))
        assert result.total_pages == 3

    def test_text_not_empty(self, pdf_3pages):
        result = scan_for_description(str(pdf_3pages))
        assert result.text.strip()

    def test_respects_max_pages(self, pdf_long):
        result = scan_for_description(str(pdf_long), max_pages=2)
        assert result.pages_scanned <= 2

    def test_stops_early_when_min_words_reached(self, pdf_long):
        result = scan_for_description(str(pdf_long), max_pages=10, min_words=50)
        assert result.pages_scanned < 10

    def test_full_scan_when_text_short(self, pdf_3pages):
        # Each page has ~8 words → won't reach min_words=1000 → scans all
        result = scan_for_description(str(pdf_3pages), max_pages=3, min_words=1000)
        assert result.pages_scanned == 3

    def test_pdf_path_in_result(self, pdf_3pages):
        result = scan_for_description(str(pdf_3pages))
        assert result.pdf_path == str(pdf_3pages)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_for_description(str(tmp_path / "missing.pdf"))

    def test_single_page_doc(self, pdf_1page):
        result = scan_for_description(str(pdf_1page))
        assert result.total_pages == 1
        assert "Only page" in result.text

    def test_max_pages_capped_at_total(self, pdf_3pages):
        result = scan_for_description(str(pdf_3pages), max_pages=100)
        assert result.pages_scanned <= 3
