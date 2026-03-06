import cohere
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import re
import os

load_dotenv()

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
qdrant = QdrantClient(":memory:")
COLLECTION = "research_docs"
VECTOR_SIZE = 1024  # Cohere embed-english-v3.0 outputs 1024 dimensions

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

def get_embeddings(texts: list[str]) -> list:
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings

def get_query_embedding(query: str) -> list:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]

def load_pdf(path: str) -> int:
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
            payload={"text": chunks[i], "chunk_id": i}
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(chunks)

def search_docs(query: str, top_k: int = 3) -> str:
    embedding = get_query_embedding(query)
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=top_k
    ).points

    if not results or results[0].score < 0.3:
        return "No relevant information found in documents."

    context = "\n---\n".join([r.payload["text"] for r in results])
    return f"Document results (score: {results[0].score:.3f}):\n{context}"