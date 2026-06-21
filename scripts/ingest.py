"""Build the hybrid retrieval index from source documents."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunker import chunk_document
from src.config import Settings, load_settings
from src.document_loader import DocumentLoadingError, load_documents
from src.embeddings import EmbeddingService
from src.keyword_search import BM25Index
from src.text_cleaner import clean_text
from src.utils import setup_logging, write_json, write_jsonl
from src.vector_store import FaissVectorStore


LOGGER = logging.getLogger(__name__)


def build_index(settings: Settings | None = None) -> dict[str, Any]:
    """Load source documents, chunk them, and build FAISS/BM25 indexes."""

    settings = settings or load_settings()
    settings.ensure_runtime_directories()

    documents = load_documents(settings.raw_data_dir)
    LOGGER.info("Found %s source document(s) in %s", len(documents), settings.raw_data_dir)

    total_pages = sum(len(document.pages) for document in documents)
    total_raw_characters = sum(len(page.text) for document in documents for page in document.pages)
    LOGGER.info("Extracted %s page(s) and %s raw characters", total_pages, total_raw_characters)

    if settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    processed_documents: list[dict[str, Any]] = []
    total_clean_characters = 0

    for document in tqdm(documents, desc="Documents"):
        cleaned_pages = []
        for page in document.pages:
            cleaned_text = clean_text(page.text)
            if cleaned_text:
                cleaned_pages.append(type(page)(page_number=page.page_number, text=cleaned_text))
                total_clean_characters += len(cleaned_text)

        cleaned_document = type(document)(
            source_path=document.source_path,
            source_file=document.source_file,
            pages=cleaned_pages,
        )
        chunks = chunk_document(
            cleaned_document,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)

        processed_path = settings.processed_data_dir / f"{document.source_path.stem}.txt"
        processed_path.write_text(
            "\n\n".join(page.text for page in cleaned_pages),
            encoding="utf-8",
        )
        processed_documents.append(
            {
                "source_file": document.source_file,
                "pages": len(document.pages),
                "cleaned_pages": len(cleaned_pages),
                "processed_path": str(processed_path.relative_to(ROOT)),
            }
        )

    if not all_chunks:
        raise DocumentLoadingError("No chunks were created from the source documents. Check PDF extraction output.")

    LOGGER.info("Created %s chunks from cleaned text (%s characters)", len(all_chunks), total_clean_characters)

    embedder = EmbeddingService(settings.embedding_model)
    LOGGER.info("Embedding model loaded: %s", settings.embedding_model)
    embeddings = embedder.encode_documents(
        [chunk.text for chunk in all_chunks],
        batch_size=settings.embedding_batch_size,
        show_progress=True,
    )
    LOGGER.info("Embedding dimension: %s", embedder.dimension)

    vector_store = FaissVectorStore.build(embeddings)
    vector_store.save(settings.index_files["faiss"])

    bm25_index = BM25Index.build([chunk.text for chunk in all_chunks])
    bm25_index.save(settings.index_files["bm25"])

    write_jsonl(settings.index_files["chunks"], [chunk.to_dict() for chunk in all_chunks])

    metadata = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_files": [document.source_file for document in documents],
        "documents": processed_documents,
        "page_count": total_pages,
        "raw_character_count": total_raw_characters,
        "clean_character_count": total_clean_characters,
        "chunk_count": len(all_chunks),
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embedder.dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "vector_weight": settings.vector_weight,
        "keyword_weight": settings.keyword_weight,
        "min_relevance_score": settings.min_relevance_score,
    }
    write_json(settings.index_files["meta"], metadata)

    LOGGER.info("Index artifacts saved to %s", settings.index_dir)
    LOGGER.info("Index build finished successfully. Run `python main.py` to start the bot.")
    return metadata


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    try:
        metadata = build_index(settings)
    except Exception as exc:
        LOGGER.error("Index build failed: %s", exc)
        raise

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nNext step: python main.py")


if __name__ == "__main__":
    main()
