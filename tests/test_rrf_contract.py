"""Regression tests for retrieval semantics and stable document identity."""

import inspect
from unittest.mock import patch

import pytest

from app.retriever import EnterpriseHybridRetriever, RetrievalResult


def result(method, ids):
    return RetrievalResult(
        method=method,
        documents=[{"id": identifier, "text": str(identifier)} for identifier in ids],
        scores=[1.0] * len(ids),
        execution_time_ms=0.0,
    )


def test_rrf_combines_matching_document_ids():
    with patch("app.retriever.QdrantClient"):
        retriever = EnterpriseHybridRetriever()
    docs, scores = retriever.fuse_with_rrf(
        result("bm25", ["shared", "bm25-only"]),
        result("vector", ["shared", "vector-only"]),
    )
    assert docs[0]["id"] == "shared"
    assert scores[0] == pytest.approx(2 / 61)
    assert len(docs) == 3


def test_vector_search_requires_real_embedding():
    with patch("app.retriever.QdrantClient") as client:
        retriever = EnterpriseHybridRetriever()
        with pytest.raises(ValueError, match="embedding"):
            retriever.search_vector("example")
        client.return_value.search.assert_not_called()
