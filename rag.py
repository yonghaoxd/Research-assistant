from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import re

model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(":memory:")
COLLECTION = "research_docs"
VECTOR_SIZE = 384

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

def load_pdf(path: str) -> int:
    reader = PdfReader(path)
    text = " ".join([p.extract_text() for p in reader.pages])
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []
    for i in range(0, len(words), 450):
        chunk = " ".join(words[i:i+450])
        chunks.append(chunk)
    embeddings = model.encode(chunks, show_progress_bar=False)
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i], "chunk_id": i}
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(chunks)

def search_docs(query: str, top_k: int = 3) -> str:
    embedding = model.encode([query])[0].tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=embedding,
        limit=top_k
    ).points

    if not results or results[0].score < 0.3:
        return "No relevant information found in documents."

    context = "\n---\n".join([r.payload["text"] for r in results])
    return f"Document results (score: {results[0].score:.3f}):\n{context}"