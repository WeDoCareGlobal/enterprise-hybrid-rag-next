# Enterprise Hybrid RAG Next — source integration status

Original source: `WeDoCareGlobal-CC/enterprise-hybrid-rag`, `main` branch.
Staged without intentional functional changes: `app/main.py`,
`app/retriever.py`, `app/__init__.py`, `eval/ragas_eval.py`,
`eval/__init__.py`, `requirements.txt`, `requirements.lock`.

**Not yet validated:** dependency installation, import checks, FastAPI endpoints,
BM25/Qdrant integration, RRF correctness, SSE behavior, evaluation metrics,
security and production readiness.

**Known dependency discrepancy:** `app/main.py` imports
`prometheus_client`, but `requirements.txt` does not list it.
Resolve before running application tests.

No data sources, API credentials, deployment or public release were migrated.
