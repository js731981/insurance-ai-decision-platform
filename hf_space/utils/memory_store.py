from typing import List, Dict

# Simple in-memory store
CLAIM_MEMORY: List[Dict] = []


def store_claim(claim: Dict):
    CLAIM_MEMORY.append(claim)


def get_similar_claims(description: str, amount: float, top_k: int = 3):
    results = []

    for c in CLAIM_MEMORY:
        score = 0

        # simple text similarity
        if any(word in c["description"].lower() for word in description.lower().split()):
            score += 0.5

        # amount similarity
        if abs(c["amount"] - amount) < 200:
            score += 0.5

        results.append((score, c))

    results.sort(key=lambda x: x[0], reverse=True)

    return [r[1] for r in results[:top_k]]

