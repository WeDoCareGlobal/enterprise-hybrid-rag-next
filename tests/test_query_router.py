"""Unit tests for the query router (factual / creative / code)."""

from app.query_router import QueryRouter, RouteDecision


def _router() -> QueryRouter:
    return QueryRouter()


def test_routes_code_query():
    r = _router()
    d = r.route("kako implementirati fastapi endpoint sa pydantic modelom?")
    assert d.is_code
    assert d.suggested_top_k == 10


def test_routes_code_from_snippet():
    r = _router()
    d = r.route("def calculate(x): return x + 1\n# SyntaxError on line 4")
    assert d.is_code


def test_routes_creative_query():
    r = _router()
    d = r.route("napisi kratku priču o roboticu koji voli kafu")
    assert d.is_creative
    assert d.suggested_top_k == 12


def test_routes_factual_query():
    r = _router()
    d = r.route("sta je RRF fusion i kako radi u hibridnom pretrazivanju?")
    assert d.is_factual


def test_routes_factual_default_with_low_confidence():
    r = _router()
    d = r.route("hello")
    assert d.is_factual
    assert d.confidence < 0.6


def test_returns_route_decision_type():
    r = _router()
    d = r.route("select * from users where id = 1")
    assert isinstance(d, RouteDecision)
    assert d.route in ("factual", "creative", "code")
    assert 0.0 <= d.confidence <= 1.0
    assert set(d.signals) == {"factual", "creative", "code"}
