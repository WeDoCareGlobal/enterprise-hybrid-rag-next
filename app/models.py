"""Pydantic models for Enterprise Hybrid RAG API.

Request/response schemas for FastAPI endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    use_vector: bool = Field(default=True, description="Enable vector search")
    use_bm25: bool = Field(default=True, description="Enable BM25 search")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 1:
            raise ValueError("Query cannot be empty")
        return v

    class Config:
        frozen = True


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""

    query: str
    documents: list[dict[str, Any]]
    fused_scores: list[float]
    rrf_k: int
    total_time_ms: float
    bm25_time_ms: float | None = None
    vector_time_ms: float | None = None
    sources: list[str] = Field(default_factory=list)

    class Config:
        frozen = True


class StreamQueryRequest(BaseModel):
    """Request schema for the /query/stream endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    use_bm25: bool = Field(default=True, description="Enable BM25 search")
    use_vector: bool = Field(default=True, description="Enable vector search")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 1:
            raise ValueError("Query cannot be empty")
        return v

    class Config:
        frozen = True


class StreamQueryResponse(BaseModel):
    """Response schema for SSE stream events (internal use)."""

    event: str
    data: dict[str, Any]

    class Config:
        frozen = True


class HealthResponse(BaseModel):
    """Health check response schema."""

    bm25: dict[str, Any] = Field(default_factory=dict)
    qdrant: dict[str, Any] = Field(default_factory=dict)
    nim: dict[str, Any] = Field(default_factory=dict)
    overall_status: str = Field(default="unknown", description="overall_status")

    class Config:
        frozen = True


class MetricsResponse(BaseModel):
    """Prometheus metrics response schema."""

    content_type: str = Field(default="text/plain; version=0.0.4; charset=utf-8")
    data: str
