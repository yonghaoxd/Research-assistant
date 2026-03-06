from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import supervisor
from memory import MemorySystem
from rag import load_pdf
import asyncio
import os

load_dotenv()

app = FastAPI(title="AI Research Assistant")
memory = MemorySystem()

# Load PDF on startup
PDF_PATH = "sample.pdf"

@app.on_event("startup")
async def startup():
    chunks = load_pdf(PDF_PATH)
    print(f"✅ Loaded {chunks} chunks from PDF")

# --- MODELS ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class QueryResponse(BaseModel):
    question: str
    answer: str
    agents_used: list[str]
    status: str = "success"

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "AI Research Assistant running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        short_term = memory.get_short_term()
        long_term = memory.search(request.question)

        answer, results = supervisor(
            request.question,
            short_term,
            long_term
        )

        memory.add_short_term("user", request.question)
        memory.add_short_term("assistant", answer)

        important = ["what is", "how", "explain", "compare", "difference"]
        if any(kw in request.question.lower() for kw in important):
            memory.save(f"User asked: {request.question[:100]} | Answer: {answer[:150]}")

        return QueryResponse(
            question=request.question,
            answer=answer,
            agents_used=list(results.keys())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def generate():
        short_term = memory.get_short_term()
        long_term = memory.search(request.question)

        answer, results = supervisor(
            request.question,
            short_term,
            long_term
        )

        memory.add_short_term("user", request.question)
        memory.add_short_term("assistant", answer)

        # Stream word by word
        words = answer.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/memory")
async def get_memory():
    return {
        "short_term_messages": len(memory.short_term),
        "short_term": memory.short_term
    }

@app.delete("/memory")
async def clear_memory():
    memory.short_term = []
    return {"status": "Short-term memory cleared"}