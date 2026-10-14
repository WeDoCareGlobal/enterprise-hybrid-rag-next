# Retrieval core repair

Base: 2da2fdef4e63b153e91b250ecd94f22866c493a8. HTTP application loading is lazy so standalone retriever tests do not require FastAPI. Evaluation package no longer imports a nonexistent test_dataset module; Optional typing is imported and optional datasets dependency is loaded inside the guarded evaluation path.

Unavailable evaluator returns BLOCKED, null metrics, all_passed false and zero evaluated samples; fabricated 0.975/0.965 mock PASS removed. Three regression tests passed locally with Python 3.12 in an isolated environment, without datasets/ragas.

Not yet verified: actual embedding provider, live Qdrant index, authenticated service integration, true Ragas quality metrics, SSE/API deployment or container build. Apache-2.0 source license preserved.
