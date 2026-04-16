from __future__ import annotations


def build_production_explanation(current_claim, similar_claims, fraud_score):
    explanation: dict = {}

    # ---------------------------
    # 1. SUMMARY
    # ---------------------------
    explanation["summary"] = {
        "decision": current_claim["decision"],
        "risk_level": "HIGH" if fraud_score > 0.6 else "MEDIUM" if fraud_score > 0.3 else "LOW",
        "fraud_score": round(float(fraud_score), 2),
    }

    # ---------------------------
    # 2. KEY SIGNALS
    # ---------------------------
    signals: list[str] = []

    signals.append(f"Severity detected: {current_claim['severity']}")
    signals.append(
        f"Claim amount: ${current_claim['amount']} vs policy limit ${current_claim['policy_limit']}"
    )

    if float(current_claim["amount"]) > 0.7 * float(current_claim["policy_limit"]):
        signals.append("High claim amount relative to policy limit")

    explanation["signals"] = signals

    # ---------------------------
    # 3. RAG INSIGHTS
    # ---------------------------
    rag = {
        "total": len(similar_claims),
        "approved": 0,
        "risky": 0,
        "examples": [],
    }

    for c in similar_claims:
        if c.get("decision") in ["REJECTED", "INVESTIGATE"]:
            rag["risky"] += 1
        else:
            rag["approved"] += 1

        rag["examples"].append(
            {
                "id": c.get("claim_id", "UNKNOWN"),
                "decision": c.get("decision", "UNKNOWN"),
                "amount": c.get("amount", "N/A"),
            }
        )

    explanation["rag"] = rag

    # ---------------------------
    # 4. INTERPRETATION
    # ---------------------------
    if rag["total"] > 0:
        if rag["risky"] > rag["approved"]:
            interpretation = "Historical patterns indicate elevated fraud risk."
        else:
            interpretation = "Historical patterns align with valid claims."
    else:
        interpretation = "No historical context available."

    explanation["interpretation"] = interpretation

    # ---------------------------
    # 5. FINAL JUSTIFICATION
    # ---------------------------
    explanation["final"] = (
        f"Decision '{current_claim['decision']}' is based on combined signals from "
        f"image analysis, rule validation, and historical claim patterns."
    )

    return explanation


def format_explanation_ui(exp):
    signals = "".join([f"- {s}\n" for s in exp["signals"]])

    return f"""
## Decision Overview

**Decision:** {exp['summary']['decision']}  
**Risk Level:** {exp['summary']['risk_level']}  
**Fraud Score:** {exp['summary']['fraud_score']}

---

## Key Risk Signals
{signals}

---

## Historical Evidence
- Similar claims analyzed: {exp['rag']['total']}
- Risk-aligned: {exp['rag']['risky']}
- Valid: {exp['rag']['approved']}

---

## Interpretation
{exp['interpretation']}

---

## Final Decision Rationale
{exp['final']}
"""

