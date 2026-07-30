"""
ONSU Kitchen Assistant - RAG runtime.

Loads the existing Chroma vector database
and answers questions using retrieved corpus context.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from openai import OpenAI
import chromadb

from app.config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
)


client = OpenAI()


# -----------------------------------------------------------------------------
# Load Chroma database
# -----------------------------------------------------------------------------

print("[rag] Loading Chroma database...")

chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(
    f"[rag] Loaded {collection.count()} chunks"
)


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

def get_query_embedding(query):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    return response.data[0].embedding


# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------

def retrieve(query, k=TOP_K):

    embedding = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
    )

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]

    return list(
        zip(chunks, metadata)
    )


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are ONSU's internal kitchen operations assistant.

You help chefs, bakers and supervisors with:
- recipes
- production methods
- SOPs
- equipment settings
- troubleshooting
- costing information

Rules:

1. Only answer using the provided context.
2. If the answer is not in the context, say:
   "That's not in our documented recipes/SOPs — check with a supervisor."
3. Never invent temperatures, weights, costs or ingredients.
4. If multiple products could match, ask for clarification.
5. Keep answers concise and practical for kitchen staff.
6. If calculating numbers, show the calculation.
"""


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

def answer(query):

    retrieved = retrieve(query)

    context_parts = []

    for chunk, meta in retrieved:

        context_parts.append(
            f"""
SOURCE: {meta['source']}

{chunk}
"""
        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    response = client.chat.completions.create(

        model=CHAT_MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
CONTEXT:

{context}


QUESTION:

{query}
"""
            },
        ],

    )


    return response.choices[0].message.content
    
