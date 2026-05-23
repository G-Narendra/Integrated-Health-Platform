"""
Text chunking for document processing and RAG.
"""

import re
from typing import List


class TextChunker:
    """Splits documents into overlapping chunks for embedding."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """Split text into chunks with metadata."""
        if not text:
            return []

        chunks = []
        paragraphs = re.split(r"\n\s*\n", text.strip())
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(self._make_chunk(current_chunk.strip(), chunk_idx, metadata))
                    chunk_idx += 1
                current_chunk = para + "\n"

                # Handle very long paragraphs
                while len(current_chunk) > self.chunk_size:
                    chunks.append(self._make_chunk(
                        current_chunk[:self.chunk_size].strip(), chunk_idx, metadata
                    ))
                    chunk_idx += 1
                    current_chunk = current_chunk[self.chunk_size - self.chunk_overlap:]

        if current_chunk.strip():
            chunks.append(self._make_chunk(current_chunk.strip(), chunk_idx, metadata))

        return chunks

    def _make_chunk(self, text: str, index: int, metadata: dict = None) -> dict:
        return {
            "text": text,
            "chunk_index": index,
            "metadata": metadata or {},
        }
