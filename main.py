from fastapi import FastAPI, HTTPException, Security, Request, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import supervisor
from memory import MemorySystem
from rag import load_pdf, load_pdf_bytes, list_documents
from datetime import datetime
from collections import defaultdict
import asyncio
import logging
import os
import re
import time
import json

load_dotenv()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # saves to file
        logging.StreamHandler()          # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Research Assistant")
memory = MemorySystem()

# --- AUTH ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

VALID_API_KEYS = {
    "dev-key-123": "developer",
    "user-key-456": "user",
}

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )
    return api_key

# --- RATE LIMITER ---
request_counts = defaultdict(list)
RATE_LIMIT = 10
RATE_WINDOW = 60

def check_rate_limit(api_key: str):
    now = time.time()
    window_start = now - RATE_WINDOW
    request_counts[api_key] = [
        t for t in request_counts[api_key] if t > window_start
    ]
    if len(request_counts[api_key]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for key: {api_key[:8]}...")
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
            logger.warning(f"Blocked prompt injection attempt: {text[:50]}")
            raise HTTPException(
                status_code=400,
                detail="Input contains blocked content."
            )
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- MIDDLEWARE: Log every request ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    logger.info(
        f"METHOD={request.method} "
        f"PATH={request.url.path} "
        f"STATUS={response.status_code} "
        f"DURATION={duration:.2f}s"
    )
    
    return response

# --- MODELS ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

class QueryResponse(BaseModel):
    question: str
    answer: str
    agents_used: list[str]
    response_time: float
    status: str = "success"

# --- LOAD PDF ON STARTUP ---
PDF_PATH = "sample.pdf"

@app.on_event("startup")
async def startup():
    logger.info("Starting AI Research Assistant...")
    chunks = load_pdf(PDF_PATH)
    logger.info(f"Loaded {chunks} chunks from PDF")

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "AI Research Assistant running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_messages": len(memory.short_term)
    }

@app.get("/metrics")
async def metrics(api_key: str = Security(verify_api_key)):
    """Show API usage metrics"""
    return {
        "active_api_keys": len(request_counts),
        "request_counts": {
            key[:8] + "...": len(times)
            for key, times in request_counts.items()
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: str = Security(verify_api_key)
):
    start_time = time.time()
    check_rate_limit(api_key)
    clean_question = sanitize_input(request.question)

    logger.info(f"Query received | key={api_key[:8]}... | question={clean_question[:50]}")

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

        response_time = time.time() - start_time
        logger.info(f"Query completed | agents={list(results.keys())} | time={response_time:.2f}s")

        return QueryResponse(
            question=clean_question,
            answer=answer,
            agents_used=list(results.keys()),
            response_time=round(response_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

    # --- UPLOAD ENDPOINT ---

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    check_rate_limit(api_key)

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Validate file size (max 10MB)
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    logger.info(f"Upload received | file={file.filename} | size={len(file_bytes)} bytes")

    try:
        doc_meta = load_pdf_bytes(file_bytes, file.filename)
        logger.info(f"Document indexed | doc_id={doc_meta['doc_id']} | chunks={doc_meta['chunks']}")

        return {
            "message": "Document uploaded and indexed successfully",
            "doc_id": doc_meta["doc_id"],
            "filename": doc_meta["filename"],
            "chunks_indexed": doc_meta["chunks"],
            "words_indexed": doc_meta["words"]
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- LIST DOCUMENTS ---
@app.get("/documents")
async def get_documents(api_key: str = Security(verify_api_key)):
    docs = list_documents()
    return {
        "total_documents": len(docs),
        "documents": docs
    }

# --- QUERY SPECIFIC DOCUMENT ---
class DocQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    doc_id: str = Field(..., description="Document ID from upload response")

@app.post("/query/document", response_model=QueryResponse)
async def query_document(
    request: DocQueryRequest,
    api_key: str = Security(verify_api_key)
):
    start_time = time.time()
    check_rate_limit(api_key)
    clean_question = sanitize_input(request.question)

    logger.info(f"Doc query | doc_id={request.doc_id} | question={clean_question[:50]}")

    try:
        # Search only in the specified document
        from rag import search_docs as _search
        context = _search(clean_question, doc_id=request.doc_id)

        from agents import groq_client, MODEL
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions based ONLY on the provided context. Be precise."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {clean_question}"
                }
            ],
            max_tokens=400
        )

        answer = response.choices[0].message.content
        response_time = time.time() - start_time

        return QueryResponse(
            question=clean_question,
            answer=answer,
            agents_used=["document"],
            response_time=round(response_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Doc query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))