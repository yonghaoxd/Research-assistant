from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import supervisor
from memory import MemorySystem
from rag import load_pdf
from datetime import datetime
from collections import defaultdict
import asyncio
import os
import re
import time

load_dotenv()

app = FastAPI(title="AI Research Assistant")
memory = MemorySystem()

# --- AUTH ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Store valid API keys — in production this would be a database
VALID_API_KEYS = {
    "dev-key-123": "developer",
    "user-key-456": "user",
}

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Include your key in the X-API-Key header."
        )
    return api_key

# --- RATE LIMITER ---
request_counts = defaultdict(list)  # key -> list of timestamps
RATE_LIMIT = 10      # max requests
RATE_WINDOW = 60     # per 60 seconds

def check_rate_limit(api_key: str):
    now = time.time()
    window_start = now - RATE_WINDOW

    # Remove old requests outside the window
    request_counts[api_key] = [
        t for t in request_counts[api_key] if t > window_start
    ]

    if len(request_counts[api_key]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per minute."
        )

    request_counts[api_key].append(now)

# --- INPUT SANITIZER ---
BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"you are now",
    r"forget your instructions",
    r"act as",
    r"jailbreak",
    r"<script>",
    r"system prompt",
]

def sanitize_input(text: str) -> str:
    text_lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text_lower):
            raise HTTPException(
                status_code=400,
                detail="Input contains blocked content."
            )
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- MODELS ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class QueryResponse(BaseModel):
    question: str
    answer: str
    agents_used: list[str]
    status: str = "success"

# --- LOAD PDF ON STARTUP ---
PDF_PATH = "sample.pdf"

@app.on_event("startup")
async def startup():
    chunks = load_pdf(PDF_PATH)
    print(f"✅ Loaded {chunks} chunks from PDF")

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "AI Research Assistant running"}

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: str = Security(verify_api_key)  # requires valid API key
):
    # Rate limit check
    check_rate_limit(api_key)

    # Sanitize input
    clean_question = sanitize_input(request.question)

    try:
        short_term = memory.get_short_term()
        long_term = memory.search(clean_question)

        answer, results = supervisor(
            clean_question,
            short_term,
            long_term
        )

        memory.add_short_term("user", clean_question)
        memory.add_short_term("assistant", answer)

        important = ["what is", "how", "explain", "compare", "difference"]
        if any(kw in clean_question.lower() for kw in important):
            memory.save(f"User asked: {clean_question[:100]} | Answer: {answer[:150]}")

        return QueryResponse(
            question=clean_question,
            answer=answer,
            agents_used=list(results.keys())
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }