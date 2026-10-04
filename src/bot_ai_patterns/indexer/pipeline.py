"""Main indexing pipeline: load docs → chunk → embed → save."""

from __future__ import annotations

from pathlib import Path

from .chunkers import Chunk, FixedSizeChunker, StructureChunker
from .embedder import Embedder
from .store import IndexStore

_TEXT_EXTENSIONS = {".py", ".md", ".txt", ".rst"}


def collect_documents(root: str | Path) -> list[Path]:
    root = Path(root)
    docs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _TEXT_EXTENSIONS | {".pdf"}:
            docs.append(path)
    return docs


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _read_pdf(path: Path) -> str:
    try:
        import pymupdf  # fitz

        doc = pymupdf.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"  [warn] cannot read PDF {path.name}: {e}")
        return ""


class IndexPipeline:
    def __init__(
        self,
        docs_dir: str | Path,
        index_dir: str | Path,
        chunk_size: int = 600,
        overlap: int = 120,
    ):
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        self.struct_chunker = StructureChunker()
        self.embedder = Embedder()

    def run(self) -> dict[str, IndexStore]:
        docs = collect_documents(self.docs_dir)
        print(f"Found {len(docs)} documents in {self.docs_dir}")

        fixed_chunks: list[Chunk] = []
        struct_chunks: list[Chunk] = []

        for path in docs:
            print(f"  Reading: {path.name}")
            text = read_file(path)
            if not text.strip():
                print(f"    [skip] empty")
                continue
            source = str(path)
            fixed_chunks.extend(self.fixed_chunker.chunk(text, source))
            struct_chunks.extend(self.struct_chunker.chunk(text, source))

        print(f"\nFixed chunks:     {len(fixed_chunks)}")
        print(f"Structure chunks: {len(struct_chunks)}")

        stores: dict[str, IndexStore] = {}
        for name, chunks in [("fixed", fixed_chunks), ("structure", struct_chunks)]:
            print(f"\nEmbedding [{name}] ({len(chunks)} chunks)…")
            vectors = self.embedder.embed_chunks(chunks)
            store = IndexStore(self.index_dir)
            store.build(chunks, vectors)
            store.save(name)
            stores[name] = store
            print(f"{name} stats: {store.stats()}")

        return stores
