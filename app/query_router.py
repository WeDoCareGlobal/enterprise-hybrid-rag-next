"""Query router for Enterprise Hybrid RAG.

Classifies an incoming query into one of three routes:

* ``factual``   -- fact-seeking questions (hybrid BM25 + vector retrieval)
* ``creative``  -- generation / ideation requests (vector-first, diverse top-k)
* ``code``      -- programming questions (BM25-heavy, exact-token priority)

The router is heuristic (no LLM call required) so it works offline and in
development mode. An optional LLM-backed classifier can be swapped in later
without changing the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RouteDecision:
    """Result of routing a single query."""

    route: str  # "factual" | "creative" | "code"
    confidence: float  # 0.0 - 1.0
    signals: dict[str, float] = field(default_factory=dict)
    suggested_top_k: int = 10
    rationale: str = ""

    @property
    def is_factual(self) -> bool:
        return self.route == "factual"

    @property
    def is_creative(self) -> bool:
        return self.route == "creative"

    @property
    def is_code(self) -> bool:
        return self.route == "code"


class QueryRouter:
    """Keyword/signal-based query router."""

    # Signals pointing to code queries.
    CODE_KEYWORDS = [
        "def ", "class ", "import ", "from ", "function", "return ", "=>",
        "{", "};", "public ", "private ", "void ", "int ", "string ",
        "console.log", "print(", "SELECT ", "INSERT ", "UPDATE ", "DELETE ",
        "git ", "docker", "kubernetes", "api endpoint", "regex", "syntax",
        "error", "exception", "stack trace", "compile", "bug", "refactor",
        "typescript", "javascript", "python", "java ", "cpp", "c++", "sql",
        "html", "css", "react", "fastapi", "pydantic",
    ]
    CODE_HINT_WORDS = [
        "kako", "how to", "implementiraj", "implement", "napisi kod",
        "write code", "skripta", "script", "primer", "example",
    ]

    # Signals pointing to creative queries.
    CREATIVE_KEYWORDS = [
        "napisi", "write", "priča", "story", "poema", "poem", "esej", "essay",
        "scenario", "script", "scenarij", "imaginiraj", "imagine", "kreiraj",
        "create", "dizajn", "design", "naslov", "title", "slogan", "slogan",
        "opis", "describe", "objasni", "explain", "pripovijest", "tale",
    ]

    # Signals pointing to factual queries.
    FACTUAL_KEYWORDS = [
        "sta", "what", "koji", "which", "koliko", "how many", "kada", "when",
        "gde", "where", "ko", "who", "zasto", "why", "definisanje", "definicija",
        "definition", "stanje", "status", "rezultat", "result", "podatak",
        "data", "informacija", "information", "fakt", "fact", "istina", "true",
    ]

    def __init__(self, *, creative_top_k: int = 12, code_top_k: int = 10, factual_top_k: int = 10) -> None:
        self.creative_top_k = creative_top_k
        self.code_top_k = code_top_k
        self.factual_top_k = factual_top_k

        self._code_re = re.compile("|".join(re.escape(k) for k in self.CODE_KEYWORDS), re.IGNORECASE)
        self._code_hint_re = re.compile("|".join(re.escape(k) for k in self.CODE_HINT_WORDS), re.IGNORECASE)
        self._creative_re = re.compile("|".join(re.escape(k) for k in self.CREATIVE_KEYWORDS), re.IGNORECASE)
        self._factual_re = re.compile("|".join(re.escape(k) for k in self.FACTUAL_KEYWORDS), re.IGNORECASE)

    def route(self, query: str) -> RouteDecision:
        """Classify *query* and return a :class:`RouteDecision`."""
        q = query.strip()
        signals: dict[str, float] = {}

        code_marks = len(self._code_re.findall(q)) + len(self._code_hint_re.findall(q))
        creative_marks = len(self._creative_re.findall(q))
        factual_marks = len(self._factual_re.findall(q))

        total = max(code_marks + creative_marks + factual_marks, 1)
        signals["code"] = round(code_marks / total, 3)
        signals["creative"] = round(creative_marks / total, 3)
        signals["factual"] = round(factual_marks / total, 3)

        # Structural code signals: looks like a snippet / error trace.
        if re.search(r"(?m)^\s*(def |class |import |from |#|//|/\*|SELECT |console\.log|print\()", q):
            signals["code"] += 0.25
        if re.search(r"(Traceback|Error:|Exception|SyntaxError|line \d+)", q, re.IGNORECASE):
            signals["code"] += 0.25

        # Question words lean factual unless overridden by creative/code hints.
        if re.search(r"\b(ko|sta|kako|gde|kada|koliko|zasto|what|how|where|when|why|who)\b", q, re.IGNORECASE):
            signals["factual"] += 0.15

        best = max(signals, key=lambda k: signals[k])
        best_score = signals[best]

        if best == "code":
            route, confidence, top_k = "code", min(best_score, 0.99), self.code_top_k
            rationale = "Code/snippet signals detected; lexical-weighted retrieval."
        elif best == "creative":
            route, confidence, top_k = "creative", min(best_score, 0.99), self.creative_top_k
            rationale = "Creative/generative signals detected; vector-first diverse retrieval."
        else:
            route, confidence, top_k = "factual", min(best_score, 0.99), self.factual_top_k
            rationale = "Fact-seeking signals detected; standard hybrid retrieval."

        # Low signal overall -> default factual with low confidence.
        if total <= 1 and best_score < 0.4:
            route, confidence = "factual", 0.35
            rationale = "No strong route signals; defaulting to factual."

        return RouteDecision(
            route=route,
            confidence=round(confidence, 3),
            signals=signals,
            suggested_top_k=top_k,
            rationale=rationale,
        )


# Default router used by the API layer.
_default_router: QueryRouter | None = None


def get_query_router() -> QueryRouter:
    global _default_router
    if _default_router is None:
        _default_router = QueryRouter()
    return _default_router
