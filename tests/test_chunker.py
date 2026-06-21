from pathlib import Path

from src.chunker import chunk_document
from src.document_loader import DocumentPage, LoadedDocument


def test_chunker_splits_long_text_and_preserves_metadata() -> None:
    text = ("CREATE TABLE demo (id int);\n\n" + ("lorem ipsum dolor sit amet " * 40)).strip()
    document = LoadedDocument(
        source_path=Path("postgresql-18-US.pdf"),
        source_file="postgresql-18-US.pdf",
        pages=[DocumentPage(page_number=12, text=text)],
    )

    chunks = chunk_document(document, chunk_size=180, chunk_overlap=40)

    assert len(chunks) >= 2
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.source_file == "postgresql-18-US.pdf" for chunk in chunks)
    assert all(chunk.page_start == 12 and chunk.page_end == 12 for chunk in chunks)
    assert chunks[0].char_end > chunks[1].char_start


def test_chunker_keeps_chunk_indices_in_order() -> None:
    text = "Paragraph one.\n\n" + ("data " * 80) + "\n\nParagraph two."
    document = LoadedDocument(
        source_path=Path("doc.txt"),
        source_file="doc.txt",
        pages=[DocumentPage(page_number=1, text=text)],
    )

    chunks = chunk_document(document, chunk_size=160, chunk_overlap=30)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id.endswith("0000")
