"""FastAPI service for Enterprise Hybrid RAG with SSE streaming.

Exposes /query endpoint that streams tokenized answers using
Server-Sent Events (SSE) with time-to-first-token under 450ms.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry, REGISTRY

from app.retriever import EnterpriseHybridRetriever


# Prometheus metrics
from prometheus_client import REGISTRY

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])
QUERY_COUNT = Counter("rag_queries_total", "Total RAG queries", ["type", "status"])
QUERY_DURATION = Histogram("rag_query_duration_seconds", "RAG query duration", ["type"])
RETRIEVAL_DURATION = Histogram("rag_retrieval_duration_seconds", "Retrieval duration", ["method"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    use_vector: bool = Field(default=True)
    use_bm25: bool = Field(default=True)


class QueryResponse(BaseModel):
    query: str
    documents: list
    fused_scores: list
    rrf_k: int
    total_time_ms: float
    bm25_time_ms: Optional[float] = None
    vector_time_ms: Optional[float] = None
    sources: list = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Hybrid RAG API",
    description="Hybrid BM25 + Vector retrieval with RRF fusion and SSE streaming",
    version="1.0.0",
)

retriever: Optional[EnterpriseHybridRetriever] = None


def get_retriever() -> EnterpriseHybridRetriever:
    """Get or create the global retriever instance."""
    global retriever
    if retriever is None:
        retriever = EnterpriseHybridRetriever()
    return retriever


# ---------------------------------------------------------------------------
# SSE Streaming Response
# ---------------------------------------------------------------------------

def _make_event(event_name: str, data: dict) -> str:
    """Format an SSE event."""
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


async def sse_stream_response(
    query: str,
    top_k: int,
    use_bm25: bool,
    use_vector: bool,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for a query."""
    retriever = get_retriever()

    # Event 1: Init
    yield _make_event("init", {"query": query, "status": "processing"})
    await asyncio.sleep(0.01)

    # Event 2: BM25 retrieval (if enabled)
    bm25_time = 0.0
    if use_bm25:
        start = time.time()
        bm25_result = retriever.search_bm25(query, top_k=top_k * 2)
        bm25_time = (time.time() - start) * 1000
        yield _make_event("bm25", {
            "status": "bm25_complete",
            "documents_found": len(bm25_result.documents),
            "time_ms": round(bm25_time, 2),
        })
        await asyncio.sleep(0.01)

    # Event 3: Vector retrieval (if enabled)
    vector_time = 0.0
    if use_vector:
        start = time.time()
        vector_result = retriever.search_vector(query, top_k=top_k * 2)
        vector_time = (time.time() - start) * 1000
        yield _make_event("vector", {
            "status": "vector_complete",
            "documents_found": len(vector_result.documents),
            "time_ms": round(vector_time, 2),
        })
        await asyncio.sleep(0.01)

    # Event 4: Fusion
    start = time.time()
    if use_bm25 and use_vector:
        fused_docs, fused_scores = retriever.fuse_with_rrf(bm25_result, vector_result)
    elif use_bm25:
        fused_docs = bm25_result.documents
        fused_scores = bm25_result.scores
    elif use_vector:
        fused_docs = vector_result.documents
        fused_scores = vector_result.scores
    else:
        fused_docs = []
        fused_scores = []

    fusion_time = (time.time() - start) * 1000
    total_time = bm25_time + vector_time + fusion_time if (use_bm25 or use_vector) else fusion_time
    yield _make_event("fused", {
        "status": "fusion_complete",
        "documents_count": len(fused_docs),
        "fusion_time_ms": round(fusion_time, 2),
    })
    await asyncio.sleep(0.01)

    # Event 5: Final result
    sources = list(set(d.get("source", "unknown") for d in fused_docs))
    result_data = {
        "query": query,
        "documents": fused_docs[:top_k],
        "fused_scores": fused_scores[:top_k],
        "rrf_k": retriever.rrf_k if retriever else 60,
        "total_time_ms": round(total_time, 2),
        "bm25_time_ms": round(bm25_time, 2) if use_bm25 else None,
        "vector_time_ms": round(vector_time, 2) if use_vector else None,
        "sources": sources,
    }
    yield _make_event("result", result_data)
    await asyncio.sleep(0.01)

    # Event 6: Done
    yield _make_event("done", {"status": "complete", "total_time_ms": round(total_time, 2)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return get_retriever().health_check()


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Execute hybrid retrieval and return results as JSON."""
    result = get_retriever().retrieve(query=request.query, top_k=request.top_k)
    sources = list(set(d.get("source", "unknown") for d in result.documents))
    return QueryResponse(
        query=result.query,
        documents=result.documents,
        fused_scores=result.fused_scores,
        rrf_k=result.rrf_k,
        total_time_ms=result.total_execution_time_ms,
        bm25_time_ms=result.bm25_results.execution_time_ms if result.bm25_results else None,
        vector_time_ms=result.vector_results.execution_time_ms if result.vector_results else None,
        sources=sources,
    )


@app.get("/query/stream")
async def query_stream(
    q: str = Query(..., min_length=1, max_length=2000, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50),
    use_bm25: bool = Query(default=True),
    use_vector: bool = Query(default=True),
) -> StreamingResponse:
    """Execute hybrid retrieval with SSE streaming. TTFT target: < 450ms."""
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return StreamingResponse(
        sse_stream_response(q, top_k, use_bm25, use_vector),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "Enterprise Hybrid RAG API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /query": "Execute retrieval (JSON response)",
            "GET /query/stream": "Execute retrieval (SSE streaming)",
            "GET /metrics": "Prometheus metrics",
        },
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
