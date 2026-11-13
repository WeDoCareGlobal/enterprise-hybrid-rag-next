"""Enterprise Hybrid RAG - Application Module.

Exports retrieval core plus the agentic RAG, routing and redaction layers so
retrieval-only consumers do not need to import the FastAPI application.
"""

from app.corrective_rag import CorrectiveRAG, CorrectiveRAGResult
from app.query_router import QueryRouter, RouteDecision, get_query_router
from app.redaction import RedactionEngine, RedactedText, configure_redaction, get_redaction_engine
from app.retriever import EnterpriseHybridRetriever

__all__ = [
    "EnterpriseHybridRetriever",
    "CorrectiveRAG",
    "CorrectiveRAGResult",
    "QueryRouter",
    "RouteDecision",
    "get_query_router",
    "RedactionEngine",
    "RedactedText",
    "configure_redaction",
    "get_redaction_engine",
]


def __getattr__(name):
    # Retrieval-only consumers must not import the HTTP application or metrics.
    if name == "app":
        from app.main import app
        return app
    raise AttributeError(name)
