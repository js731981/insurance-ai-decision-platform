from __future__ import annotations


def build_rag_explanation(current_claim, similar_claims):
    explanation = []

    explanation.append(f"Claim ID: {current_claim['claim_id']}")
    explanation.append(f"Detected severity: {current_claim['severity']}")
    explanation.append(f"Claim amount: ${current_claim['amount']}")
    explanation.append(f"Policy limit: ${current_claim['policy_limit']}")

    if not similar_claims:
        explanation.append("No similar historical claims found. Decision based on rules only.")
        return explanation

    explanation.append("\n🔍 RAG Context Insights:")

    fraud_cases = 0
    approved_cases = 0

    for c in similar_claims:
        if c["decision"] in ["REJECTED", "INVESTIGATE"]:
            fraud_cases += 1
        elif c["decision"] == "APPROVED":
            approved_cases += 1

    total = len(similar_claims)

    explanation.append(f"- Found {total} similar claims")

    if fraud_cases > approved_cases:
        explanation.append(f"- Majority were flagged risky ({fraud_cases}/{total})")
        explanation.append("👉 Current claim follows similar risk pattern")
    else:
        explanation.append(f"- Majority were approved ({approved_cases}/{total})")
        explanation.append("👉 Current claim aligns with valid claims")

    explanation.append("\n📌 Final reasoning:")
    explanation.append(
        f"Decision '{current_claim['decision']}' is influenced by both rules and historical patterns."
    )

    return explanation


def format_explanation(explanation_list):
    return "\n".join([f"- {line}" for line in explanation_list])

