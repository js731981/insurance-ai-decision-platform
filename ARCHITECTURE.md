# Architecture — Insurance AI Decision Platform

## Overview

The system is a **production-style AI decision platform** designed to enhance real-time insurance claim processing using:

- Multi-agent architecture
- Retrieval-Augmented Generation (RAG)
- Human-in-the-Loop (HITL)
- Vector-based memory

It augments existing systems **without disrupting core workflows**, acting as an intelligent decision layer.

---

## High-Level Architecture


Client / UI
↓
FastAPI (API Layer) — includes **`POST /analyze`** (core pipeline + post-decision enrichment; see below)
↓
InsurFlow Orchestrator (Async Control Layer)
↓
────────────────────────────────────────────
Decision Pipeline (Multi-Agent System)
────────────────────────────────────────────
├── Embedding Service (Ollama)
│ ↓
│ Vector Store (Chroma)
│ → Retrieve similar claims (weighted + optional filters)
│ → Optional lightweight rerank (product_code boost)
│ → Compact context builder (token-capped RAG context)
│
├── Fraud Agent (LLM-based reasoning)
│    → Strict JSON mode with parse-repair + optional retry when output is invalid
├── Policy Agent (Rule-based validation)
│
└── Decision Agent (Fusion Layer)
     → Optional DL fraud probability fusion (small local head; torch if available)
     → Optional image-signal fusion (heuristic or CNN-driven damage severity)
↓
HITL (Human-in-the-Loop)
↓
────────────────────────────────────────────
Memory Layer (Atomic Write)
────────────────────────────────────────────
└── store_claim()
→ Embedding + Metadata (single upsert)
→ Enables future retrieval & learning


---

## End-to-End Flow

Claim received via API
Optional claim image received (JSON `image_base64` or multipart upload)
Embedding generated from claim description (or JSON snapshot if description is empty)
Similar claims retrieved from vector store (weighted ranking; optional metadata/decision/product filters)
Compact similar-claims context built and injected into the fraud prompt (token-capped)
Fraud Agent evaluates risk using LLM + context (strict structured JSON output)
Policy Agent validates rules and constraints
Optional DL fraud head computes a fraud probability (small local model; deterministic fallback without torch)
Optional image analysis runs (heuristic CV; optional CNN classification when weights are available)
Decision Agent fuses fraud + policy (+ optional DL score + optional image severity + majority reviewed outcome among similar hits) → final decision
HITL triggered if:
decision == INVESTIGATE
calibrated_confidence < threshold
Claim stored (embedding + metadata) in vector DB
Future decisions leverage stored knowledge

---

## Post-decision layer (2026-04-23)

**Purpose:** compatibility and UX for “enhanced” responses **without** changing core triage logic.

**Flow:**

1. Client calls **`POST /analyze`** with a claim-shaped JSON body.
2. **`InsurFlowOrchestrator.process_claim`** runs unchanged (embed → RAG/fraud/policy/decision → HITL semantics → primary Chroma upsert).
3. **`post_decision_service.enhance_after_decision`** merges the core JSON with:
   - **`trace`** — ordered strings from `post_decision_agent.plan_steps` / `reflect` (e.g. core decision, RAG steps, optional RAG retry when fraud score is low).
   - **`rag`** — up to *k* similar rows from a **separate** Chroma collection **`claims_post_decision_sbert`**, embedded with **`sentence-transformers`** (`all-MiniLM-L6-v2`) so dimensions never conflict with the main Ollama embedding index.
   - **`llm`** — narrative text from **`generate_explanation`** in `app.services.llm_service` (uses the **`ollama`** Python client when available; otherwise a deterministic fallback string).
4. Best-effort **`rag_service.store_claim`** upserts a minimal record into the SBERT collection for future post-decision retrieval.

**Explicit non-goals:** this layer must **not** rewrite `decision`, `confidence_score`, or `agent_outputs` from the orchestrator.

---

## Core Components

### 1. API Layer (FastAPI)
- Handles incoming requests
- Exposes endpoints for claims, review, **post-decision analyze**, and inference
- Serves Swagger UI and minimal dashboard
- Supports JSON and multipart claim submission for UI-friendly image uploads
- Exposes image explainability endpoints for stored claims:
  - `GET /claims/{claim_id}/image-preview`
  - `GET /claims/{claim_id}/gradcam` (503 when CNN/weights unavailable)

---

### 2. Orchestrator
- Central control layer
- Executes agents asynchronously
- Ensures **single atomic memory write** to the **primary** claims vector index
- Applies HITL decision logic
- **`POST /analyze`** delegates here first; post-decision storage is a **secondary** best-effort path in `rag_service`

---

### 3. Multi-Agent System

#### 🔹 Fraud Agent (LLM)
- Performs contextual reasoning
- Uses retrieved similar claims (RAG) via a compact, token-capped context
- Outputs:
  - fraud_score
  - explanation
  - entities
  - strict JSON parse health (internal) with retry when enabled

#### 🔹 Policy Agent (Rules)
- Validates business rules (e.g., policy limits)
- Ensures compliance and deterministic checks

#### 🔹 Decision Agent (Fusion Layer)
- Combines Fraud + Policy outputs
- Optionally fuses a lightweight DL fraud probability head (if enabled)
- Calibrates confidence using human-reviewed similar-claim majority when available
- Produces:
  - final decision
  - confidence score
  - explanation

---

### 4. Embedding Service
- Generates embeddings using Ollama
- Model: `nomic-embed-text`
- Converts unstructured claim descriptions into vectors

---

### 5. Vector Store (Chroma - MVP)
- Stores:
  - embeddings
  - structured metadata
- Enables semantic search for similar claims
- **Two logical indexes in MVP:** (1) primary **`claims`** collection — Ollama embeddings, orchestrator RAG + analytics; (2) optional **`claims_post_decision_sbert`** — SBERT embeddings, **post-decision** RAG only.

Note:
- Uses embedded mode (SQLite)
- Suitable for MVP (single process)
- Can be replaced with Qdrant/Pinecone for production

---

### 6. Memory Design (Atomic Write)

All data is stored in a **single upsert operation**:

- claim_id
- fraud_score
- decision
- confidence
- entities
- timestamp
- explanation
- review_status
- has_image / image preview + image signals (best-effort)
- optional CNN diagnostics (label/confidence) when enabled

✔ Ensures consistency  
✔ Avoids partial updates  
✔ Enables reliable retrieval  

---

### 7. HITL (Human-in-the-Loop)

Triggered when:
- Low confidence
- Investigation required

Features:
- Manual approval/rejection
- Feedback stored in memory
- Improves future decisions

---

## Data Model

### Unstructured Data
- claim_description → embedding

### Structured Metadata
- claim_id
- fraud_score
- decision
- confidence
- entities
- timestamp
- explanation
- review_status

---

## Key Design Principles

- **Context-aware decisioning (RAG)**
- **Hybrid AI (LLM + Rules)**
- **Explainability-first design**
- **Feedback-driven learning (HITL)**
- **Atomic memory consistency**
- **Modular & extensible architecture**

---

## Scalability Strategy

| Component        | Scaling Approach |
|----------------|----------------|
| API            | Horizontal scaling (multiple instances) |
| Orchestrator   | Async execution |
| Vector DB      | Replace with Qdrant (distributed) |
| LLM            | GPU / Managed APIs |
| Storage        | External persistent systems |

---

## Production Evolution

MVP → Production:

- Chroma → Qdrant / Pinecone
- Local Ollama → Managed LLM / GPU cluster
- Single process → Kubernetes deployment
- Basic logs → Observability stack (Prometheus, Grafana)

---

## Application lifecycle

- **Startup:** eager initialization of vector store, LLM service (**warmup** for Ollama when configured), and embedding client to reduce first-request latency and surface misconfiguration early.

## Observability

- Structured logging (agent-level)
- Metrics:
  - latency
  - confidence
  - HITL rate
  - embedding/retrieval usage (logged as pipeline perf fields)
- Error tracking and tracing

---

## Why This Architecture?

This design enables:

- Real-time decisioning
- Context-aware reasoning
- Continuous improvement via feedback
- Easy transition from MVP → production

---

## Summary

This is not just an AI model integration.

It is a **modular, context-aware, and feedback-driven AI decision system** designed for real-world, scalable, and explainable operations.