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
FastAPI (API Layer)
↓
InsurFlow Orchestrator (Async Control Layer)
↓
────────────────────────────────────────────
Decision Pipeline (Multi-Agent System)
────────────────────────────────────────────
├── Embedding Service (Ollama)
│ ↓
│ Vector Store (Chroma)
│ → Retrieve similar claims (RAG)
│
├── Fraud Agent (LLM-based reasoning)
├── Policy Agent (Rule-based validation)
│
└── Decision Agent (Fusion Layer)
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
Embedding generated from claim description
Similar claims retrieved from vector store (RAG)
Fraud Agent evaluates risk using LLM + context
Policy Agent validates rules and constraints
Decision Agent combines outputs → final decision
HITL triggered if:
decision == INVESTIGATE
confidence < threshold
Claim stored (embedding + metadata) in vector DB
Future decisions leverage stored knowledge

---

## Core Components

### 1. API Layer (FastAPI)
- Handles incoming requests
- Exposes endpoints for claims, review, and inference
- Serves Swagger UI and minimal dashboard

---

### 2. Orchestrator
- Central control layer
- Executes agents asynchronously
- Ensures **single atomic memory write**
- Applies HITL decision logic

---

### 3. Multi-Agent System

#### 🔹 Fraud Agent (LLM)
- Performs contextual reasoning
- Uses retrieved similar claims (RAG)
- Outputs:
  - fraud_score
  - explanation
  - entities

#### 🔹 Policy Agent (Rules)
- Validates business rules (e.g., policy limits)
- Ensures compliance and deterministic checks

#### 🔹 Decision Agent (Fusion Layer)
- Combines Fraud + Policy outputs
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

## Observability

- Structured logging (agent-level)
- Metrics:
  - latency
  - confidence
  - HITL rate
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