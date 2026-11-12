# Pre-Release Consolidation Baseline — enterprise-hybrid-rag-next

## Issue Reference
- Issue #2: "Pre-release: original-code consolidation and 10-step completion plan" (open)
- Acceptance: All required tests on exact candidate SHA; no mocked success, undocumented skipped integrations, exposed credentials, missing attribution, or unproven product claims. Preserve donor repositories; adapt selected modules on development branches. No public release or production deployment in this planning task. OmniGuard remains excluded.

## Baseline SHAs (from issue)
| Reference | SHA | Description |
|-----------|-----|-------------|
| Baseline | bfb0a74 | Repair retrieval imports and fail closed when evaluator is unavailable |
| Current HEAD | 40fc4e4 | Resolve merge conflict in ci.yml: keep Bandit upload + safety scan from main with dependency-audit job from PR #6 |

## Delta
- Commits ahead of baseline (bfb0a74): 65

## Step 1 Status: IN PROGRESS
**Task:** Zaključati bfb0a74 baseline i testirati stvarnu /query rutu: ispraviti injekciju embedding modela i konfiguraciju.
(Lock bfb0a74 baseline and test real /query route: fix embedding model injection and configuration.)

**Resolution plan:**
1. Lock bfb0a74 as reviewed baseline
2. Test real /query route
3. Fix embedding model injection (app/main.py passes search_vector without embedding_model, retriever rejects it)
4. Fix configuration for Qdrant integration

## Source Selection (from issue)
| Source | Decision | Reason |
|--------|----------|--------|
| Original app/retriever.py | Keep | BM25/RRF structure, apply fixes |
| Private we-do-care-pharma-os/src/index.js | Don't copy | Real Workers AI calls but computes embedding per document per search |
| gd-scan/vectorDb.ts | Don't copy | Pseudo-embedding, not neural semantic model |

## Steps 2–10
Documented in issue #2 body. Not started.

## Verification
- [ ] Baseline documented (this file)
- [ ] Step 1: bfb0a74 baseline locked, /query route tested, embedding model injection fixed
- [ ] Steps 2–10 not started
