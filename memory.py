import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from datetime import datetime
import json
import os

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
qdrant = QdrantClient(":memory:")
MEMORY_COLLECTION = "memory"
VECTOR_SIZE = 1024

qdrant.create_collection(
    collection_name=MEMORY_COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

MEMORY_FILE = "memory_store.json"

class MemorySystem:
    def __init__(self):
        self.short_term = []
        self.counter = 0
        self.load_long_term()

    def _embed(self, texts: list[str], input_type: str = "search_document") -> list:
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type=input_type
        )
        return response.embeddings

    def load_long_term(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                self.counter = data.get("counter", 0)
                memories = data.get("memories", [])
                if memories:
                    embeddings = self._embed([m["content"] for m in memories])
                    points = [
                        PointStruct(
                            id=memories[i]["id"],
                            vector=embeddings[i],
                            payload=memories[i]
                        )
                        for i in range(len(memories))
                    ]
                    qdrant.upsert(collection_name=MEMORY_COLLECTION, points=points)

    def save(self, content: str):
        memory = {
            "id": self.counter,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session": datetime.now().strftime("%Y-%m-%d")
        }
        embedding = self._embed([content])[0]
        qdrant.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[PointStruct(
                id=self.counter,
                vector=embedding,
                payload=memory
            )]
        )
        existing = []
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                existing = json.load(f).get("memories", [])
        existing.append(memory)
        with open(MEMORY_FILE, 'w') as f:
            json.dump({"counter": self.counter + 1, "memories": existing}, f, indent=2)
        self.counter += 1

    def add_short_term(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > 10:
            self.short_term = self.short_term[-10:]

    def get_short_term(self) -> str:
        if not self.short_term:
            return "No previous messages."
        return "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in self.short_term[-6:]
        ])

    def search(self, query: str) -> str:
        recall_phrases = ["last session", "previous", "earlier", "we discussed", "what did"]
        if any(p in query.lower() for p in recall_phrases):
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, 'r') as f:
                    memories = json.load(f).get("memories", [])
                    if memories:
                        return "Past memories:\n" + "\n---\n".join(
                            [f"[{m['session']}] {m['content']}" for m in memories[-5:]]
                        )
            return "No past memories."

        embedding = self._embed([query], input_type="search_query")[0]
        try:
            results = qdrant.query_points(
                collection_name=MEMORY_COLLECTION,
                query=embedding,
                limit=3
            ).points
            if not results or results[0].score < 0.2:
                return "No relevant memories."
            return "Relevant memories:\n" + "\n---\n".join(
                [f"[{r.payload['session']}] {r.payload['content'][:200]}" for r in results]
            )
        except:
            return "No relevant memories."