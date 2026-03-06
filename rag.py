import cohere
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import re
import os
import hashlib

load_dotenv()

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
qdrant = QdrantClient(":memory:")
COLLECTION = "research_docs"
VECTOR_SIZE = 1024

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# Track uploaded documents
uploaded_docs = {}  # doc_id -> metadata
point_counter = [0]  # mutable counter for qdrant point IDs

def get_embeddings(texts: list[str]) -> list:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return response.embeddings.float_

def get_query_embedding(query: str) -> list:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings.float_[0]

def load_pdf(path: str) -> int:
    """Load a PDF file into Qdrant"""
    reader = PdfReader(path)
    text = " ".join([p.extract_text() for p in reader.pages])
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []
    for i in range(0, len(words), 450):
        chunk = " ".join(words[i:i+450])
        chunks.append(chunk)

    print(f"Embedding {len(chunks)} chunks with Cohere...")
    embeddings = get_embeddings(chunks)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": chunks[i], "chunk_id": i, "doc_id": "default"}
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    point_counter[0] = len(chunks)
    return len(chunks)

def load_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """Load a PDF from uploaded bytes"""
    import io

    # Generate unique doc ID
    doc_id = hashlib.md5(file_bytes).hexdigest()[:8]

    # Check if already uploaded
    if doc_id in uploaded_docs:
        return uploaded_docs[doc_id]

    # Extract text
    reader = PdfReader(io.BytesIO(file_bytes))
    text = " ".join([p.extract_text() for p in reader.pages])
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []
    for i in range(0, len(words), 450):
        chunk = " ".join(words[i:i+450])
        chunks.append(chunk)

    # Embed and store
    embeddings = get_embeddings(chunks)

    start_id = point_counter[0]
    points = [
        PointStruct(
            id=start_id + i,
            vector=embeddings[i],
            payload={
                "text": chunks[i],
                "chunk_id": i,
                "doc_id": doc_id,
                "filename": filename
            }
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    point_counter[0] += len(chunks)

    # Save metadata
    doc_meta = {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "words": len(words)
    }
    uploaded_docs[doc_id] = doc_meta
    return doc_meta

def search_docs(query: str, top_k: int = 3, doc_id: str = None) -> str:
    """Search documents — optionally filter by doc_id"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    embedding = get_query_embedding(query)

    # Filter by specific document if doc_id provided
    search_filter = None
    if doc_id:
        search_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=embedding,
        query_filter=search_filter,
        limit=top_k
    ).points

    if not results or results[0].score < 0.3:
        return "No relevant information found in documents."

    context = "\n---\n".join([r.payload["text"] for r in results])
    return f"Document results (score: {results[0].score:.3f}):\n{context}"

def list_documents() -> list:
    """List all uploaded documents"""
    return list(uploaded_docs.values())