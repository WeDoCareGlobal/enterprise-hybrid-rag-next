"""Enterprise Hybrid RAG - Application Module."""

from app.retriever import EnterpriseHybridRetriever

__all__ = ["EnterpriseHybridRetriever"]


def __getattr__(name):
    # Retrieval-only consumers must not import the HTTP application or metrics.
    if name == "app":
        from app.main import app
        return app
    raise AttributeError(name)

