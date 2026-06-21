"""Chunk source documents into retrieval-friendly passages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from src.document_loader import DocumentPage, LoadedDocument


@dataclass(slots=True)
class DocumentChunk:
    """A text chunk with metadata preserved for citations."""

    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict[str, object]:
        """Serialize chunk to a plain dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DocumentChunk":
        """Deserialize a chunk from a plain dictionary."""

        return cls(
            chunk_id=str(payload["chunk_id"]),
            source_file=str(payload["source_file"]),
            page_start=int(payload["page_start"]),
            page_end=int(payload["page_end"]),
            chunk_index=int(payload["chunk_index"]),
            text=str(payload["text"]),
            char_start=int(payload["char_start"]),
            char_end=int(payload["char_end"]),
        )


@dataclass(slots=True)
class PageSpan:
    """Character span occupied by a page in the concatenated document text."""

    page_number: int
    start: int
    end: int


def chunk_document(document: LoadedDocument, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Split a loaded document into chunks with overlap and page metadata."""

    full_text, page_spans = _build_document_text(document.pages)
    if not full_text:
        return []

    total_length = len(full_text)
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0
    stem = Path(document.source_file).stem

    while start < total_length:
        target_end = min(start + chunk_size, total_length)
        end = _find_breakpoint(full_text, start, target_end)
        if end <= start:
            end = target_end

        raw_slice = full_text[start:end]
        stripped = raw_slice.strip()
        if not stripped:
            start = min(end + 1, total_length)
            continue

        left_trim = len(raw_slice) - len(raw_slice.lstrip())
        right_trim = len(raw_slice) - len(raw_slice.rstrip())
        char_start = start + left_trim
        char_end = end - right_trim
        page_numbers = [
            span.page_number
            for span in page_spans
            if span.end > char_start and span.start < char_end
        ]
        if not page_numbers:
            page_numbers = [1]

        chunks.append(
            DocumentChunk(
                chunk_id=f"{stem}_chunk_{chunk_index:04d}",
                source_file=document.source_file,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                chunk_index=chunk_index,
                text=stripped,
                char_start=char_start,
                char_end=char_end,
            )
        )
        chunk_index += 1

        if end >= total_length:
            break

        next_start = max(char_end - chunk_overlap, 0)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _build_document_text(pages: list[DocumentPage]) -> tuple[str, list[PageSpan]]:
    non_empty_pages = [(page.page_number, page.text.strip()) for page in pages if page.text and page.text.strip()]
    if not non_empty_pages:
        return "", []

    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for page_index, (page_number, text) in enumerate(non_empty_pages):
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(PageSpan(page_number=page_number, start=start, end=cursor))
        if page_index != len(non_empty_pages) - 1:
            parts.append("\n\n")
            cursor += 2

    return "".join(parts), spans


def _find_breakpoint(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    search_window = text[start:target_end]
    priorities = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    for token in priorities:
        position = search_window.rfind(token)
        if position > max(0, len(search_window) // 2):
            return start + position + len(token)
    return target_end

