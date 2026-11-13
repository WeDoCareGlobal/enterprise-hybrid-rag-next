"""Unit tests for the Corrective RAG agentic loop."""

from unittest.mock import MagicMock, patch

import pytest

from app.corrective_rag import CorrectiveRAG
from app.query_router import QueryRouter
from app.redaction import RedactionEngine
from app.retriever import (
    EnterpriseHybridRetriever,
    HybridRetrievalResult,
    RetrievalResult,
)


def _result(method, ids, scores=None):
    return RetrievalResult(
        method=method,
        documents=[{"id": i, "text": f"text about {i}"} for i in ids],
        scores=scores or [1.0] * len(ids),
        execution_time_ms=0.0,
    )


def _hybrid(doc_ids, scores=None, top_score=0.5):
    bm = _result("bm25", doc_ids, scores)
    vt = _result("vector", doc_ids, scores)
    return HybridRetrievalResult(
        query="q",
        documents=bm.documents,
        fused_scores=scores or [top_score] * len(doc_ids),
        rrf_k=60,
        bm25_results=bm,
        vector_results=vt,
        total_execution_time_ms=10.0,
    )


def _retriever(hit_result):
    with patch("app.retriever.QdrantClient"):
        retriever = EnterpriseHybridRetriever(bm25_corpus=["doc"])
    retriever.retrieve = MagicMock(return_value=hit_result)
    return retriever


def test_accepts_on_first_pass_high_relevance():
    retriever = _retriever(_hybrid(["shared"], scores=[1.0], top_score=1.0))
    cr = CorrectiveRAG(retriever, max_iterations=3, relevance_threshold=0.2)
    out = cr.run("what is hybrid retrieval", top_k=5, redact=False, embedding_model=MagicMock())
    assert out.accepted is True
    assert out.iterations == 1
    assert out.relevance_score > 0.2


def test_rewrites_and_retries_on_low_relevance():
    first = _hybrid([], scores=[], top_score=0.0)
    second = _hybrid(["shared"], scores=[0.9], top_score=0.9)
    retriever = _retriever(first)
    retriever.retrieve = MagicMock(side_effect=[first, second])
    cr = CorrectiveRAG(retriever, max_iterations=3, relevance_threshold=0.5)
    out = cr.run("obscure quantum flux capacitor query", top_k=5, redact=False, embedding_model=MagicMock())
    assert out.iterations == 2
    assert out.accepted is True
    assert out.final_query != out.original_query


def test_stops_after_max_iterations_still_returns_best():
    first = _hybrid([], scores=[], top_score=0.0)
    retriever = _retriever(first)
    retriever.retrieve = MagicMock(return_value=first)
    cr = CorrectiveRAG(retriever, max_iterations=2, relevance_threshold=0.9)
    out = cr.run("zzz nowhere match", top_k=5, redact=False, embedding_model=MagicMock())
    assert out.iterations == 2
    assert out.accepted is False
    assert out.documents == []


def test_redacts_documents_in_output():
    retriever = _retriever(_hybrid(["shared"], scores=[1.0], top_score=1.0))
    cr = CorrectiveRAG(retriever, max_iterations=1, relevance_threshold=0.0)
    out = cr.run("contact ivanic.peric@example.com", top_k=5, redact=True)
    assert "ivanic.peric@example.com" not in str(out.documents)
    assert out.redacted is True


def test_route_is_recorded_in_trace():
    retriever = _retriever(_hybrid(["shared"], scores=[1.0], top_score=1.0))
    cr = CorrectiveRAG(retriever, max_iterations=1, relevance_threshold=0.0)
    out = cr.run("def foo(): pass", top_k=5, redact=False)
    assert out.route.route == "code"
    assert len(out.trace) == 1
    assert out.trace[0].decision == "accept"


def test_uses_default_router_when_none_given():
    with patch("app.retriever.QdrantClient"):
        retriever = EnterpriseHybridRetriever(bm25_corpus=["doc"])
    retriever.retrieve = MagicMock(return_value=_hybrid(["a"], scores=[1.0], top_score=1.0))
    cr = CorrectiveRAG(retriever)
    assert isinstance(cr.router, QueryRouter)


def test_min_documents_accepts_empty_high_score_docs():
    retriever = _retriever(_hybrid([], scores=[], top_score=1.0))
    cr = CorrectiveRAG(retriever, max_iterations=2, relevance_threshold=0.0, min_documents=0)
    out = cr.run("anything", top_k=5, redact=False, embedding_model=MagicMock())
    assert out.accepted is True


def test_dev_mode_without_embedding_uses_bm25_only():
    """CorrectiveRAG must not crash when no embedding model is configured."""
    with patch("app.retriever.QdrantClient"):
        retriever = EnterpriseHybridRetriever(bm25_corpus=["RRF fusion combines bm25 and vectors"])
    out = CorrectiveRAG(retriever, max_iterations=2, relevance_threshold=0.0).run(
        "what is rrf", top_k=5, redact=False
    )
    assert out.accepted is True
    assert out.documents
    assert "rrf" in str(out.documents).lower()
