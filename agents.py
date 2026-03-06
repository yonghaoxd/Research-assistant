from groq import Groq
from rag import search_docs
from dotenv import load_dotenv
import requests
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

def document_agent(question: str) -> str:
    context = search_docs(question)
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a document analysis specialist. Answer based ONLY on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def web_agent(question: str) -> str:
    try:
        response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + question.replace(" ", "_"),
            timeout=5
        )
        if response.status_code == 200:
            raw = response.json().get('extract', '')[:1000]
            response2 = groq_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Extract the most relevant information to answer the question."},
                    {"role": "user", "content": f"Info:\n{raw}\n\nQuestion: {question}"}
                ],
                max_tokens=200
            )
            return response2.choices[0].message.content
    except:
        pass
    return "Could not retrieve web information."

def math_agent(problem: str) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a math specialist. Show calculations clearly.
GPU reference: P100=9.5 TFLOPS, V100=14 TFLOPS, A100=19.5 TFLOPS, H100=51.2 TFLOPS"""
            },
            {"role": "user", "content": problem}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content

def supervisor(question: str, short_term: str, long_term: str) -> tuple[str, dict]:
    plan_response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a supervisor delegating to specialist agents.
Agents available:
- document_agent: searches research paper
- web_agent: finds current web info
- math_agent: does calculations

ALWAYS use document_agent for paper questions.

Respond in EXACT format:
DOCUMENT_TASK: task or SKIP
WEB_TASK: task or SKIP
MATH_TASK: task or SKIP"""
            },
            {"role": "user", "content": f"Recent conversation:\n{short_term}\n\nQuestion: {question}"}
        ],
        max_tokens=150
    )

    plan = plan_response.choices[0].message.content

    import re
    doc_task = re.search(r'DOCUMENT_TASK:\s*(.+?)(?=WEB_TASK:|$)', plan, re.DOTALL)
    web_task = re.search(r'WEB_TASK:\s*(.+?)(?=MATH_TASK:|$)', plan, re.DOTALL)
    math_task = re.search(r'MATH_TASK:\s*(.+?)$', plan, re.DOTALL)

    doc_task = doc_task.group(1).strip() if doc_task else "SKIP"
    web_task = web_task.group(1).strip() if web_task else "SKIP"
    math_task = math_task.group(1).strip() if math_task else "SKIP"

    results = {}
    if doc_task != "SKIP":
        results["document"] = document_agent(doc_task)
    if web_task != "SKIP":
        results["web"] = web_agent(web_task)
    if math_task != "SKIP":
        results["math"] = math_agent(math_task)

    combined = ""
    if "document" in results:
        combined += f"Document findings:\n{results['document']}\n\n"
    if "web" in results:
        combined += f"Web findings:\n{results['web']}\n\n"
    if "math" in results:
        combined += f"Math results:\n{results['math']}\n\n"

    final = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Combine specialist findings into a clear comprehensive answer."},
            {"role": "user", "content": f"Past context:\n{long_term}\n\nQuestion: {question}\n\nFindings:\n{combined}"}
        ],
        max_tokens=400
    )

    return final.choices[0].message.content, results