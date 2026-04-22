# Insurance AI Decision Platform — Project overview

## Purpose

**Insurance AI Decision Platform** is a **local MVP** for **micro-insurance claim triage**: it accepts a small-ticket claim and returns a structured outcome **`APPROVED`**, **`REJECTED`**, or **`INVESTIGATE`**, with fraud signals, policy checks, similarity to past claims, calibrated confidence, and optional human-in-the-loop (HITL) flags. It is built to run **without extra cloud infrastructure** beyond what you choose (default: **Ollama** + **ChromaDB** on disk).

## What the system does

1. **Claim processing pipeline** — An orchestrator runs **fraud analysis** (LLM → structured JSON) and **policy validation** (rules, e.g. amount vs limit) **in parallel**, then a **decision agent** merges them into a final triage decision and confidence.
2. **Vector memory** — Embeddings (Ollama) feed a **persistent ChromaDB** collection so the system can **retrieve similar past claims** using **review-aware weighted ranking** (reviewed items get a boost; hits coherent with the majority reviewed outcome get an additional boost).
3. **Compact RAG context** — Retrieved hits are converted into a **short, token-capped context block** (no raw metadata dump) and injected into the fraud prompt.
4. **Optional rerank** — A cheap reranker can boost retrieved hits that match the inbound `product_code` (useful when vector similarity alone is ambiguous).
5. **Multimodal image signals (new)** — Claims may include an optional **image** (JSON `image_base64` or multipart upload). The service extracts lightweight visual signals (heuristic CV by default; optional CNN classifier when configured).
6. **Calibration** — Raw **`confidence_score`** is kept; **`calibrated_confidence`** adjusts using human-reviewed similar cases when available.
7. **HITL** — Review is recommended when the decision is **`INVESTIGATE`** or when **`calibrated_confidence`** is below **0.75** (see `HitlService` in `app/core/dependencies.py`).
8. **Persistence** — Each successfully embedded claim is **upserted** into the vector store for future retrieval and analytics.
9. **Investigator workflow** — Cases on stored claims: list/filter, **assign** (`NEW` → `ASSIGNED`), and status updates (`ASSIGNED` → `IN_PROGRESS` → `RESOLVED`).
10. **Analytics** — Read-only aggregates: summary stats, heuristic anomaly-style alerts, fraud-score leaderboard.
11. **Observability & UI** — Structured logging, in-process **metrics** (`GET /metrics`), and a **minimal dashboard** at **`/ui`**.
12. **Post-decision analyze (2026-04-23)** — **`POST /analyze`** accepts a JSON **object** (passed through as a `dict` to the orchestrator—prefer the same fields as `ClaimRequest`), runs the **same** `InsurFlowOrchestrator` claim pipeline as `/claims`, then a **non-mutating enrichment layer**: step **trace**, conditional **RAG** over a **dedicated** Chroma collection (`claims_post_decision_sbert`, **sentence-transformers** / `all-MiniLM-L6-v2` when installed), reflection-driven **RAG retry** for low fraud scores, and a **best-effort LLM explanation** (`generate_explanation` via the `ollama` client, with a safe string fallback). Core **`decision` is unchanged** by this layer.

## Architecture (conceptual)

```
FastAPI (v0.1.0)
└── InsurFlowOrchestrator
    ├── EmbeddingService (Ollama) → VectorStore (Chroma) — weighted similar-claim retrieval
    │     └── ContextBuilder (token-capped) + optional LightweightReranker (product_code boost)
    ├── FraudAgent (LLM)     ─┐
    ├── PolicyAgent (rules)  ─┼─→ DecisionAgent → HITL
    │                          ├→ Optional DL fraud probability fusion (enabled via env)
    │                          └→ Optional image severity/signal fusion (enabled via env)
    └── store_claim (upsert when embedding succeeds)

Post-decision (optional consumer): `POST /analyze` → `enhance_after_decision` → `post_decision_agent` (plan/reflect) + `rag_service` (SBERT collection) + `generate_explanation`
```

**LLM execution** goes through **`LLMService` → `LLMRouter`** (timeouts, retries, optional fallbacks). Providers: **Ollama** (default), **OpenAI** and **OpenRouter** if API keys are set. **`POST /inference`** supports optional task hints (`cheap` → Ollama, `complex` → OpenAI when configured). Optional **USD cost** heuristics use `LLM_COST_USD_PER_1K_*` environment variables.

**Timeout defaults:** unless overridden via `LLM_TIMEOUT_S` / `LLM_TIMEOUT`, the default per-attempt wall-clock timeout is **60s** (with a safety floor of 60s even if set lower).

## Technology stack

| Layer | Choice |
|--------|--------|
| API | **FastAPI**, **Pydantic v2**, **httpx**, **python-dotenv** |
| Runtime | **Python 3.10+** |
| LLM / embeddings | **Ollama** (chat + embeddings); optional OpenAI / OpenRouter |
| Vector store | **ChromaDB** (embedded, persistent directory) |

Primary dependencies (see `requirements.txt`): `fastapi`, `uvicorn[standard]`, `httpx`, `python-dotenv`, `pydantic`, `chromadb`, **`sentence-transformers`** and **`ollama`** (post-decision RAG + explanation helpers; both degrade gracefully if unavailable).

## API surface (summary)

| Area | Highlights |
|------|------------|
| Service | `GET /`, `GET /health`, `GET /metrics` |
| Claims | `POST /claims`, `POST /claim`, `GET /claims`, `POST /claims/{claim_id}/review` |
| Claim images | `GET /claims/{claim_id}/image-preview`, `GET /claims/{claim_id}/gradcam` |
| Cases | `GET /cases`, `POST /cases/{claim_id}/assign`, `POST /cases/{claim_id}/status` |
| Analytics | `GET /analytics/summary`, `/analytics/anomalies`, `/analytics/leaderboard` |
| Post-decision | `POST /analyze` — same core triage as `/claims`, plus `trace`, `rag`, `llm` enrichment fields |
| Inference | `GET /claim/samples`, `POST /inference` |

**Claim input** is strict Pydantic (`ClaimRequest`, `extra="forbid"`). Empty `description` still gets a text embedding from a **JSON snapshot** of the claim for retrieval.

**Claim image input:**

- JSON requests can include `image_base64` (raw base64 string or `data:image/...;base64,...`).
- Multipart requests are supported for browser/UI clients (file field `file` or `image`).

**Image analysis configuration:**

- `ENABLE_IMAGE_ANALYSIS=true|false`
- `IMAGE_MODEL_TYPE=heuristic|cnn`
- `IMAGE_FUSION_WEIGHT` (default `0.2`)
- `CNN_CONF_THRESHOLD` (default `0.7`)
- `IMAGE_CNN_MODEL_PATH` (optional fine-tuned weights; if unavailable the system falls back safely)

## Resilience (MVP)

- **LLM:** timeouts and retries; failures logged with **`claim_id`** when present.
- **Embedding/retrieval failure:** API continues; **vector write skipped** for that request.
- **Fraud LLM failure or bad JSON:** strict JSON parsing (with optional retry when enabled) plus a safe fallback toward **`INVESTIGATE`** / moderate confidence unless policy rules force **`REJECTED`**.

## Repository layout (main)

- `app/main.py` — App factory, routers, `/ui` static mount, eager startup (vector store + LLM warmup + embeddings).
- `app/agents/` — Orchestrator, fraud, policy, decision, **post_decision** (planner/reflection hooks), base agent.
- `app/api/routes/` — health, inference, claims, **analyze**, cases, analytics.
- `app/api/claim_multipart.py` — Shared JSON / multipart claim parsing (covered by `tests/test_request_parsing.py`).
- `app/services/` — LLM/router/providers, embeddings, vector store, HITL, metrics, analytics, retrieval/rerank/context building, samples, **post_decision_service**, **rag_service** (post-decision SBERT store), **case_service** / **feedback_service** (in-memory scaffolding for future workflow hooks).
- `app/core/` — `Settings`, dependency wiring.
- `app/web/` — Dashboard static assets.
- `chroma_db/` (default) — Chroma persistence (`CHROMA_PERSIST_DIR`).

## Positioning

This repository is a **proof-of-concept / reference implementation** for automated micro-claim triage with local AI and a simple case workflow. **Production** use with confidential data would need stronger security, governance, and operations than this MVP.

For a concise top-level overview, see **`docs/PROJECT_OVERVIEW.md`**. For setup, environment variables, and detailed API tables, see the root **`README.md`**.

## Author

Jayendran Subramanian — Consulting AI Data Engineer | AI Architect (in progress)
