"""Governance integration for Enterprise Hybrid RAG.

Inspects queries through OmniGuard before retrieval.
Fail-closed: any error or non-ALLOW decision blocks retrieval.
"""
import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class GovernanceDecision:
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class GovernanceError(Exception):
    """Raised when governance inspection fails or blocks."""
    pass


class OmniGuardGovernance:
    """Fail-closed governance client for RAG queries."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = (base_url or os.environ.get("OMNIGUARD_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        key = api_key or os.environ.get("OMNIGUARD_API_KEY")
        if key is None:
            raise ValueError("OMNIGUARD_API_KEY not configured")
        self.api_key = key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=self.timeout,
            )
        return self._client

    async def inspect_query(self, query: str) -> str:
        """Inspect a query through OmniGuard. Returns ALLOW or raises GovernanceError."""
        try:
            client = await self._get_client()
            resp = await client.post(
                "/chat/completions",
                json={"model": "omniguard-2.0", "messages": [{"role": "user", "content": query}]},
            )
        except Exception as exc:
            logger.warning("OmniGuard governance unreachable: %s", exc)
            raise GovernanceError("governance_unreachable") from exc

        if resp.status_code < 200 or resp.status_code >= 300:
            raise GovernanceError(f"governance_http_{resp.status_code}")

        decision = resp.headers.get("X-Governance-Decision", "BLOCK")
        if decision != GovernanceDecision.ALLOW:
            raise GovernanceError("governance_blocked")

        return GovernanceDecision.ALLOW

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
