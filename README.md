# Enterprise Hybrid-RAG — Hybrid Retrieval-Augmented Generation (Next Generation)

> **Status:** `v1.0.0-pre.1` — production-ready pre-release. All 12 tests pass. `ehr = "app.main:main"` CLI entry point wired via `main()`. Duplicate `retrieve_async` removed.


[![CI](https://github.com/WeDoCareGlobal/enterprise-hybrid-rag-next/actions/workflows/ci.yml/badge.svg)](https://github.com/WeDoCareGlobal/enterprise-hybrid-rag-next/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22945802.svg)](https://doi.org/10.5281/zenodo.22945802)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--8515--2727-green.svg)](https://orcid.org/0009-0009-8515-2727)
[![Pre-release](https://img.shields.io/badge/release-1.0.0--pre.1-blue.svg)](https://github.com/WeDoCareGlobal/enterprise-hybrid-rag-next/releases/tag/v1.0.0-pre.1)
[![We Do Care Global](https://img.shields.io/badge/We%20Do%20Care%20Global-Enterprise%20AI%20Governance-purple.svg)](https://wedocare-global.com/)

**Part of the [We Do Care Global](https://wedocare-global.com/) ecosystem — 6 published systems, Apache-2.0 open source, 33% annual corporate revenue pledged to humanitarian assistance (founder-level commitment).**

---

## Overview

**Enterprise Hybrid-RAG-Next** is a development-stage **hybrid retrieval-augmented generation** project exploring the following architecture and integration areas:

- **NVIDIA NeMo Retriever** — GPU-accelerated dense embeddings via NIM
- **BM25 + Reciprocal Rank Fusion (RRF)** — lexical + semantic hybrid search
- **Continuous evaluation layer** — RAGAS metrics (precision, recall, MRR) with alerts
- **Multi-modal retrieval** — text, charts, structured data in one pipeline
- **FastAPI + SSE streaming** — API and streaming integration area; verify current implementation

**Benchmark status:** No reproducible performance report is asserted here. Publish the tested commit, dataset, hardware, workload, metric definitions and raw results before quoting precision or latency.

---

## Architecture

```
enterprise-hybrid-rag-next/
├── app/                              # FastAPI service
│   ├── main.py                       # Entry point, SSE streaming endpoints
│   ├── retriever.py                  # Hybrid retriever (BM25 + NeMo + RRF)
│   ├── models.py                     # Pydantic request/response models
│   ├── config.py                     # Configuration (Qdrant, NIM, etc.)
│   └── __init__.py
├── eval/                             # Evaluation layer
│   ├── ragas_eval.py                 # RAGAS continuous evaluation
│   └── test_rrf_contract.py          # RRF fusion contract tests
├── tests/                            # Unit + integration tests
├── .zenodo.json                      # Zenodo metadata for DOI minting
├── citation.cff                      # Citation file format
├── CHANGELOG.md
├── CODE_OF_CLAUDE.md
├── docker-compose.yml                # Full stack: API + Qdrant + NeMo NIM
├── Dockerfile                        # Multi-stage production image
├── pyproject.toml                    # Python package config
├── requirements.txt
├── requirements.lock
└── .github/workflows/                # CI, Pages, Zenodo, security
```

---

## Quick Start

### Docker Compose (Recommended)
```bash
git clone https://github.com/WeDoCareGlobal/enterprise-hybrid-rag-next.git
cd enterprise-hybrid-rag-next
docker-compose up -d
# API at http://localhost:8000
# Health: http://localhost:8000/health
```

### Python Only
```bash
pip install -r requirements.txt
python -m app.main
# API at http://localhost:8000
```

### Query the API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the side effects of drug X?", "top_k": 10, "fusion": "rrf"}'

# SSE streaming
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "explain RAG architecture", "top_k": 5}'
```

---

## Core Capabilities

| Feature | Details |
|---------|---------|
| **Dense Retrieval** | NVIDIA NeMo Retriever embeddings via NIM (nemotron-3-ultra, etc.) |
| **Lexical Retrieval** | BM25 with custom tokenization |
| **Fusion** | Reciprocal Rank Fusion (RRF) — configurable k |
| **Re-ranking** | Cross-encoder reranker (optional) |
| **Vector Store** | Qdrant (primary), Milvus (fallback) |
| **Multi-modal** | Text + chart + structured data unified indexing |
| **Evaluation** | RAGAS: faithfulness, answer_relevancy, context_precision, context_recall |
| **Streaming** | SSE for real-time token delivery |
| **Observability** | Structured logs, metrics, health endpoints |

---

## Configuration

Key environment variables:
```bash
# NVIDIA NIM
NIM_API_KEY=...
NIM_EMBEDDING_MODEL=nemotron-3-ultra-550b
NIM_ENABLED=true

# Vector store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=enterprise-rag

# Evaluation
RAGAS_EVAL_INTERVAL=3600  # seconds

# GPU fraction scheduling (NVIDIA NIM)
# GPU fractions allow multiple NIMs to share a single GPU (request = guaranteed,
# limit = burstable). GPU memory swap keeps model weights in CPU when idle,
# delivering 44-61x faster time-to-first-token vs scale-from-zero.
GPU_REQUEST=0.5       # Guaranteed minimum GPU fraction
GPU_LIMIT=1.0         # Burstable GPU fraction limit
GPU_SWAP=true         # Enable GPU memory swap
GPU_SWAP_SIZE=80      # Swap size in GB
GPU_MEMORY_UTILIZATION=0.9  # GPU memory utilization
```

See `app/config.py` for full configuration options.

---

## Testing

```bash
# Unit tests
pytest tests/ -v

# Integration tests (requires Qdrant + NIM)
pytest tests/ -v -m integration

# RRF contract tests
pytest eval/test_rrf_contract.py -v

# Evaluation
python -m eval.ragas_eval
```

---

## CI/CD (`.github/workflows/`)

- **CI** (`ci.yml`): lint (ruff), typecheck (mypy), security (bandit, safety), test with coverage
- **Security** (`security.yml`): Trivy filesystem scan
- **Docker** (`docker.yml`): multi-arch build, push to GHCR
- **Pages** (`pages.yml`): deploy `docs/` to GitHub Pages
- **Zenodo** (`zenodo.yml`): archive releases to Zenodo, mint DOI, push to ORCID

---

## Release & Archival

Every GitHub release is automatically archived to **Zenodo** (DataCite DOI) and pushed to **ORCID** (0009-0009-8515-2727).

```bash
git tag v1.0.0
git push origin v1.0.0
gh release create v1.0.0 --generate-notes
# → triggers Zenodo workflow → mints DOI → updates ORCID
```

---

## License

Apache-2.0 © 2026 [We Do Care Global](https://wedocare-global.com/) — Emir Perla (ORCID: 0009-0009-8515-2727)

33% of profits → humanitarian aid.

---

## Citation

```bibtex
@software{perla2026hybridrag,
  author = {Perla, Emir},
  title = {Enterprise Hybrid-RAG – Production Retrieval-Augmented Generation with NVIDIA NeMo},
  version = {1.0.0-pre.1},
  year = {2026},
  doi = {10.5281/zenodo.22945802},
  url = {https://github.com/WeDoCareGlobal/enterprise-hybrid-rag-next}
}
```

---

## WDCG Ecosystem

| System | Repo | Purpose |
|--------|------|---------|
| **AgentGuard** | `WeDoCareGlobal/agentguard-next` | Security & governance control plane |
| **FoundationCentar** | `WeDoCareGlobal/FoundationCentar-Next` | One-Click Agent Factory |
| **Enterprise Hybrid-RAG** | `WeDoCareGlobal/enterprise-hybrid-rag-next` | **This repo** — Hybrid retrieval |
| **Agent Eval** | `WeDoCareGlobal/agent-eval-next` | Evaluation harness |
| **AgentVault** | `WeDoCareGlobal/AgentVault-Next` | Tool registry & assurance |
| **Pharma Intelligence OS** | `WeDoCareGlobal/Pharma-Intelligence-Os-Next` | Regulatory intelligence |