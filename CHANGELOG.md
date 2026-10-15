# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Prometheus metrics endpoint (`/metrics`) with HTTP request counters, latency histograms, and RAG-specific metrics
- CI workflow with linting, typechecking, security scanning (bandit, safety, trivy), and coverage enforcement (≥80%)
- Python lockfile (`requirements.lock`) for reproducible builds
- Dockerfile with multi-stage build, non-root user, and base image digest pinning
- Security scanning in CI (bandit, safety for Python; trivy for container)

### Changed
- Updated app/main.py to include Prometheus metrics middleware and `/metrics` endpoint
- Hardened Dockerfile: non-root user (UID 1000), base image digest pinning, multi-stage build

### Security
- Added non-root user (UID 1000) in Dockerfile
- Pinned base image digests
- Added security audit steps in CI

## [1.0.0] - 2026-09-30

### Added
- Hybrid RAG pipeline: NVIDIA NeMo Retriever + custom embeddings, multi-modal retrieval, production deployment
- FastAPI service with SSE streaming (TTFT target: < 450ms)
- BM25 + Vector retrieval with RRF fusion
- Ragas evaluation framework integration
- Qdrant vector database integration
- Health check endpoint

---

**Full Changelog**: https://github.com/we-do-care-global/enterprise-hybrid-rag/commits/main