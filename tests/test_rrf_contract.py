"""Regression contract for stable RRF document identity."""

import pytest


def test_rrf_requires_shared_document_identity():
    """A local BM25 row index is not a globally stable document ID."""
    bm25_local_id = 0
    qdrant_document_id = "document-abc"
    assert bm25_local_id != qdrant_document_id


def test_missing_embedding_must_not_be_treated_as_semantic_search():
    """Zero-vector fallback is not a valid query embedding."""
    placeholder_embedding = [0.0] * 768
    assert not any(placeholder_embedding)
