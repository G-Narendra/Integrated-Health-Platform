"""
Ingest medical documents into the knowledge base.
Supports text files, PDFs, and structured medical data.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import VectorStore
from src.retrieval.chunking import TextChunker
from src.utils.logger import logger


def ingest_text_file(file_path: str, collection: str, source_name: str = ""):
    """Ingest a text file into a collection."""
    store = VectorStore()
    chunker = TextChunker()

    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return 0

    text = path.read_text(encoding="utf-8")
    metadata = {
        "source_type": collection,
        "source": source_name or path.stem,
        "title": path.stem,
        "filename": path.name,
    }

    chunks = chunker.chunk_text(text, metadata)
    doc_ids = [f"{collection[:3].upper()}-{i:04d}" for i in range(len(chunks))]

    count = store.add_documents(collection, chunks, ids=doc_ids)
    logger.info(f"Ingested {count} chunks from '{file_path}' into '{collection}'")
    return count


def ingest_directory(dir_path: str, collection: str):
    """Ingest all text files in a directory into a collection."""
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        logger.error(f"Directory not found: {dir_path}")
        return 0

    total = 0
    for file_path in path.glob("*.*"):
        if file_path.suffix in [".txt", ".md", ".json", ".csv"]:
            count = ingest_text_file(str(file_path), collection, path.name)
            total += count

    logger.info(f"Ingested {total} total chunks from directory '{dir_path}'")
    return total


def verify_knowledge_base():
    """Verify the knowledge base has expected data."""
    store = VectorStore()
    collections = store.list_collections()
    print("\nKnowledge Base Status:")
    print("-" * 40)
    for name in ["clinical_guidelines", "drug_database", "moh_regulations", "medical_literature"]:
        count = store.count(name) if name in collections else 0
        status = "✅" if count > 0 else "⚠️"
        print(f"  {status} {name}: {count} documents")
    print("-" * 40)
    return collections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest medical documents into the knowledge base")
    parser.add_argument("--file", type=str, help="Text file to ingest")
    parser.add_argument("--dir", type=str, help="Directory of text files to ingest")
    parser.add_argument("--collection", type=str, default="clinical_guidelines",
                        choices=["clinical_guidelines", "drug_database", "moh_regulations", "medical_literature"],
                        help="Target collection")
    parser.add_argument("--verify", action="store_true", help="Verify knowledge base status")

    args = parser.parse_args()

    if args.verify:
        verify_knowledge_base()
    elif args.file:
        ingest_text_file(args.file, args.collection)
    elif args.dir:
        ingest_directory(args.dir, args.collection)
    else:
        print("Please specify a file, directory, or use --verify")
        print("Or run build_vector_db.py first to populate the database.")
