"""
Build the ONSU Kitchen Assistant vector database.

Run locally when the corpus changes:

    python tools/build_index.py

This reads documents from /corpus and writes
the Chroma vector database to /vector_db/chroma.
"""

import os
import shutil
from pathlib import Path

import chromadb
from openai import OpenAI

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.config import (
    CORPUS_PATH,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


client = OpenAI()


# -----------------------------------------------------------------------------
# Document loading
# -----------------------------------------------------------------------------

def load_documents(folder: Path):
    """
    Load markdown files from corpus folder.
    """

    documents = []

    for path in sorted(folder.glob("*.md")):
        print(f"Loading: {path.name}")

        text = path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "filename": path.name,
                "text": text,
            }
        )

    if not documents:
        raise RuntimeError(
            f"No markdown files found in {folder}"
        )

    return documents


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------

import re


def chunk_by_headers(text, min_words=25):
    """
    Split markdown by ## headers.

    Keeps recipes, SOPs and product documents
    as self-contained chunks.
    """

    parts = re.split(
        r"\n(?=##\s)",
        text
    )

    parts = [
        p.strip()
        for p in parts
        if p.strip()
    ]

    merged = []
    buf = ""

    for part in parts:

        buf = (
            f"{buf}\n\n{part}"
            if buf
            else part
        )

        if len(buf.split()) >= min_words:
            merged.append(buf)
            buf = ""

    if buf:
        merged.append(buf)

    return merged


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

import time


def create_embeddings(texts, batch_size=10):
    """
    Create embeddings in batches with retry handling
    to respect OpenAI token-per-minute limits.
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        while True:
            try:
                print(
                    f"Embedding batch {i//batch_size + 1} "
                    f"({len(batch)} chunks)"
                )

                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )

                all_embeddings.extend(
                    [
                        item.embedding
                        for item in response.data
                    ]
                )

                break

            except Exception as e:
                if "RateLimitError" in str(type(e)):
                    print(
                        "Rate limit reached. Waiting 10 seconds..."
                    )
                    time.sleep(10)
                else:
                    raise e

        # small pause between batches
        time.sleep(2)

    return all_embeddings

# -----------------------------------------------------------------------------
# Build database
# -----------------------------------------------------------------------------

def build_index():

    print("Starting ONSU index build...")
    
    documents = load_documents(
        CORPUS_PATH
    )

    all_chunks = []
    metadata = []

    for doc in documents:

        chunks = chunk_by_headers(
    doc["text"]
)

        for i, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            metadata.append(
                {
                    "source": doc["filename"],
                    "chunk": i,
                }
            )


    print(
        f"Created {len(all_chunks)} chunks"
    )


    print("Generating embeddings...")

    embeddings = create_embeddings(
        all_chunks
    )


    # Reset database
    if VECTOR_DB_PATH.exists():
        shutil.rmtree(
            VECTOR_DB_PATH
        )

    VECTOR_DB_PATH.mkdir(
        parents=True,
        exist_ok=True
    )


    db = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )


    collection = db.get_or_create_collection(
        name=COLLECTION_NAME
    )


    collection.add(
        ids=[
            str(i)
            for i in range(len(all_chunks))
        ],
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )


    print(
        "Index build complete!"
    )

    print(
        f"Stored {collection.count()} chunks"
    )


if __name__ == "__main__":
    build_index()
    

