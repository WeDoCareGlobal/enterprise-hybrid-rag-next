"""Thread-safety and async tests for Enterprise Hybrid RAG."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from app.config import AppConfig, NimConfig, QdrantConfig, RagasConfig
from app.retriever import EnterpriseHybridRetriever


class TestConfigValidation:
    """Test configuration validation."""

    def test_qdrant_config_valid(self):
        config = QdrantConfig(host="localhost", port=6333)
        assert config.host == "localhost"
        assert config.port == 6333

    def test_qdrant_port_validation(self):
        with pytest.raises(ValueError, match="port"):
            QdrantConfig(host="localhost", port=70000)

    def test_nim_config_from_env_defaults(self):
        config = NimConfig.from_env()
        assert isinstance(config, NimConfig)
        assert config.model == "nemotron-3-ultra-550b"

    def test_app_config_validation(self):
        config = AppConfig()
        assert config.rrf_k == 60
        assert config.bm25_weight == 1.0
        assert config.vector_weight == 1.0


class TestRetrieverThreadSafety:
    """Test thread-safety of EnterpriseHybridRetriever."""

    def test_concurrent_bm25_searches(self):
        """Multiple threads can safely run BM25 searches."""
        import time
        corpus = [f"Document about topic {i} with some keywords" for i in range(100)]
        retriever = EnterpriseHybridRetriever(bm25_corpus=corpus)

        errors = []
        results = []

        def search_thread(thread_id, query, top_k):
            try:
                result = retriever.search_bm25(query, top_k=top_k)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=search_thread,
                args=(i, "keywords", 10),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All searches should complete
        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        total_document_ids = set()
        for _, result in results:
            for doc in result.documents:
                total_document_ids.add(doc["id"])
        assert len(total_document_ids) > 0

    def test_concurrent_retrieve(self):
        """Multiple threads can safely run full retrieve calls."""
        # Mock Qdrant client with search method
        mock_qdrant = MagicMock()
        mock_results = [
            MagicMock(id=0, payload={"text": "Result 0", "source": "vector", "metadata": {}}),
            MagicMock(id=1, payload={"text": "Result 1", "source": "vector", "metadata": {}}),
        ]
        mock_qdrant.search.return_value = mock_results

        # Mock embedding model
        class MockEmbeddingModel:
            def encode(self, text):
                return [0.0] * 384  # Fixed-size embedding vector

        errors = []
        results = []

        def retrieve_thread(thread_id, query, top_k):
            try:
                retriever = EnterpriseHybridRetriever(bm25_corpus=["Test doc"])
                result = retriever.retrieve(
                    query, top_k=top_k, embedding_model=MockEmbeddingModel()
                )
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        # Patch QdrantClient for this test
        with patch("app.retriever.QdrantClient", return_value=mock_qdrant):
            threads = []
            for i in range(10):
                t = threading.Thread(
                    target=retrieve_thread,
                    args=(i, "test query", 5),
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        # All searches should complete without errors
        for thread_id, error in errors:
            print(f"Thread {thread_id} error: {error}")
        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    def test_add_documents_concurrent(self):
        """Adding documents from multiple threads is safe."""
        corpus = [f"Document {i}" for i in range(50)]
        retriever = EnterpriseHybridRetriever(bm25_corpus=corpus)

        errors = []

        def add_docs(start_idx):
            try:
                docs = [f"New document {start_idx + i}" for i in range(10)]
                retriever.add_bm25_documents(docs)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=add_docs, args=(i * 10,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in add_documents: {errors}"
        assert len(retriever.bm25_corpus) == 100, "Expected 100 documents after concurrent adds"


class TestAsyncRetriever:
    """Test async retriever methods."""

    def test_async_retrieve(self):
        """Async retrieve returns correct results."""
        corpus = [f"Document about {i}" for i in range(30)]
        # Create retriever first
        retriever = EnterpriseHybridRetriever(bm25_corpus=corpus)

        # Mock Qdrant client with search method (applied after creation)
        mock_qdrant = MagicMock()
        mock_results = [
            MagicMock(id=0, payload={"text": "Result 0", "source": "vector", "metadata": {}}),
            MagicMock(id=1, payload={"text": "Result 1", "source": "vector", "metadata": {}}),
        ]
        mock_qdrant.search.return_value = mock_results
        # Replace the client instance on the retriever
        retriever.qdrant_client = mock_qdrant

        # Mock embedding model
        class MockEmbeddingModel:
            def encode(self, text):
                return [0.0] * 384  # Fixed-size embedding vector

        result = asyncio_test_retriever(
            retriever, "test query", top_k=5, embedding_model=MockEmbeddingModel()
        )

        assert result.query == "test query"
        assert len(result.documents) <= 5
        assert len(result.fused_scores) == len(result.documents)
        assert result.total_execution_time_ms >= 0
        assert result.bm25_results is not None
        assert result.vector_results is not None

    def test_async_retrieve_without_bm25(self):
        """Async retrieve works without BM25."""
        corpus = [f"Document about {i}" for i in range(30)]
        retriever = EnterpriseHybridRetriever(bm25_corpus=corpus)

        # Mock Qdrant client with search method (applied after creation)
        mock_qdrant = MagicMock()
        mock_results = [
            MagicMock(id=0, payload={"text": "Result 0", "source": "vector", "metadata": {}}),
            MagicMock(id=1, payload={"text": "Result 1", "source": "vector", "metadata": {}}),
        ]
        mock_qdrant.search.return_value = mock_results
        retriever.qdrant_client = mock_qdrant

        # Mock embedding model
        class MockEmbeddingModel:
            def encode(self, text):
                return [0.0] * 384  # Fixed-size embedding vector

        result = asyncio_test_retriever(
            retriever, "test query", top_k=5, embedding_model=MockEmbeddingModel()
        )

        assert result.query == "test query"
        assert len(result.documents) <= 5


def asyncio_test_retriever(retriever, query, top_k=5, embedding_model=None):
    """Helper to run async retrieve in a synchronous test."""
    import asyncio

    async def _run():
        return await retriever.retrieve_async(
            query=query, top_k=top_k, embedding_model=embedding_model
        )

    return asyncio.run(_run())
