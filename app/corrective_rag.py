"""Corrective (agentic) RAG loop for Enterprise Hybrid RAG.

Flow per query:

1. Route the query (factual / creative / code) via :class:`QueryRouter`.
2. Retrieve hybrid results (BM25 + vector, RRF fused) via
   :class:`EnterpriseHybridRetriever`.
3. Grade the relevance of the top fused documents against the query.
4. If relevance is below threshold, rewrite the query (paraphrase / expand)
   and re-retrieve, up to ``max_iterations``.
5. Return the best pass documents together with a full trace.

Grading is heuristic (fused-score proximity + lexical query-term coverage) so
the loop runs offline; an LLM grader can be plugged in later without changing
the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.query_router import QueryRouter, RouteDecision
from app.redaction import RedactedText, get_redaction_engine
from app.retriever import (
    EnterpriseHybridRetriever,
    HybridRetrievalResult,
    RetrievalResult,
)


@dataclass
class CorrectionTrace:
    """Single iteration of the corrective loop."""

    iteration: int
    query: str
    route: str
    documents_retrieved: int
    relevance_score: float
    decision: str  # "accept" | "rewrite" | "stop"
    rewritten_from: str | None = None


@dataclass
class CorrectiveRAGResult:
    """Outcome of the agentic corrective-RAG run."""

    original_query: str
    final_query: str
    route: RouteDecision
    documents: list[dict[str, Any]]
    fused_scores: list[float]
    iterations: int
    trace: list[CorrectionTrace]
    relevance_score: float
    accepted: bool
    redacted: bool
    total_time_ms: float


class CorrectiveRAG:
    """Agentic retrieval with self-correction on low relevance."""

    def __init__(
        self,
        retriever: EnterpriseHybridRetriever,
        *,
        max_iterations: int = 3,
        relevance_threshold: float = 0.25,
        min_documents: int = 1,
        router: QueryRouter | None = None,
    ) -> None:
        self.retriever = retriever
        self.max_iterations = max(1, max_iterations)
        self.relevance_threshold = relevance_threshold
        self.min_documents = min_documents
        self.router = router or QueryRouter()
        self.redaction = get_redaction_engine()

    # -- public API -------------------------------------------------------

    def run(
        self,
        query: str,
        top_k: int = 10,
        use_bm25: bool = True,
        use_vector: bool = True,
        embedding_model: Any = None,
        redact: bool = True,
    ) -> CorrectiveRAGResult:
        """Run the corrective-RAG loop for *query*.

        Returns :class:`CorrectiveRAGResult` with redacted documents when
        ``redact=True``.
        """
        import time

        start = time.time()
        route = self.router.route(query)
        current_query = query
        trace: list[CorrectionTrace] = []
        best_result: HybridRetrievalResult | None = None
        best_relevance = -1.0
        accepted = False

        for iteration in range(1, self.max_iterations + 1):
            if embedding_model is None:
                # Dev mode without an embedding model: BM25-only retrieval.
                bm25_result = self.retriever.search_bm25(current_query, top_k=top_k * 2)
                result = HybridRetrievalResult(
                    query=current_query,
                    documents=bm25_result.documents,
                    fused_scores=bm25_result.scores,
                    rrf_k=self.retriever.rrf_k,
                    bm25_results=bm25_result,
                    vector_results=None,
                    total_execution_time_ms=bm25_result.execution_time_ms,
                )
            else:
                result = self.retriever.retrieve(
                    current_query,
                    top_k=top_k * 2,
                    embedding_model=embedding_model,
                )
            relevance = self._grade(current_query, result)

            trace.append(
                CorrectionTrace(
                    iteration=iteration,
                    query=current_query,
                    route=route.route,
                    documents_retrieved=len(result.documents),
                    relevance_score=round(relevance, 4),
                    decision="accept",
                    rewritten_from=None if iteration == 1 else current_query,
                )
            )

            if relevance >= self.relevance_threshold or len(result.documents) >= self.min_documents:
                best_result = result
                best_relevance = relevance
                accepted = True
                trace[-1] = CorrectionTrace(
                    iteration=iteration,
                    query=current_query,
                    route=route.route,
                    documents_retrieved=len(result.documents),
                    relevance_score=round(relevance, 4),
                    decision="accept",
                    rewritten_from=trace[-1].rewritten_from,
                )
                break

            # Low relevance -> rewrite and retry (unless last iteration).
            if iteration < self.max_iterations:
                rewritten = self._rewrite_query(current_query, route.route, trace)
                trace[-1] = CorrectionTrace(
                    iteration=iteration,
                    query=current_query,
                    route=route.route,
                    documents_retrieved=len(result.documents),
                    relevance_score=round(relevance, 4),
                    decision="rewrite",
                    rewritten_from=current_query,
                )
                current_query = rewritten
            else:
                best_result = result
                best_relevance = relevance
                trace[-1] = CorrectionTrace(
                    iteration=iteration,
                    query=current_query,
                    route=route.route,
                    documents_retrieved=len(result.documents),
                    relevance_score=round(relevance, 4),
                    decision="stop",
                    rewritten_from=trace[-1].rewritten_from,
                )

        docs = best_result.documents[:top_k] if best_result else []
        scores = best_result.fused_scores[:top_k] if best_result else []

        if redact:
            docs = self._redact_documents(docs)

        return CorrectiveRAGResult(
            original_query=query,
            final_query=current_query,
            route=route,
            documents=docs,
            fused_scores=scores,
            iterations=len(trace),
            trace=trace,
            relevance_score=round(best_relevance, 4),
            accepted=accepted,
            redacted=redact,
            total_time_ms=round((time.time() - start) * 1000, 2),
        )

    # -- internals --------------------------------------------------------

    def _grade(self, query: str, result: HybridRetrievalResult) -> float:
        """Heuristic relevance grade in [0, 1].

        Combines:
        * normalised top fused score (vs. best possible for this retriever),
        * lexical coverage: share of query content terms present in top docs.
        """
        if not result.documents:
            return 0.0

        top_score = result.fused_scores[0] if result.fused_scores else 0.0
        # RRF max possible for a doc present in both lists at rank 1: 2 / (k + 1)
        rrf_max = 2.0 / (self.retriever.rrf_k + 1)
        score_component = min(top_score / rrf_max, 1.0) if rrf_max > 0 else 0.0

        query_terms = {
            t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2
        }
        if not query_terms:
            coverage = 0.5
        else:
            doc_text = " ".join(d.get("text", "") for d in result.documents[:5]).lower()
            hits = sum(1 for t in query_terms if t in doc_text)
            coverage = hits / len(query_terms)

        return round(0.6 * score_component + 0.4 * coverage, 4)

    def _rewrite_query(
        self,
        query: str,
        route: str,
        trace: list[CorrectionTrace],
    ) -> str:
        """Produce a rewritten query for the next retrieval pass."""
        terms = re.findall(r"\b\w+\b", query.lower())
        content = [t for t in terms if len(t) > 2][:6]

        if route == "code":
            rewritten = f"primer koda implementacija: {' '.join(content)}"
        elif route == "creative":
            rewritten = f"kreativno opisivanje: {' '.join(content)}"
        else:
            rewritten = f"definicija i podaci o: {' '.join(content)}"

        # Avoid exact loops: if the rewrite equals an earlier query, append a
        # broadening hint.
        previous = {t.query for t in trace}
        if rewritten in previous:
            rewritten = f"sirestore expanded: {query}"

        return rewritten

    def _redact_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.redaction.redact_document(doc) for doc in documents]
