from __future__ import annotations

from typing import Any

FEEDBACK: list[dict[str, Any]] = []


def add_feedback(claim_id: str, label: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"id": str(claim_id), "label": str(label)}
    if extra:
        item["extra"] = dict(extra)
    FEEDBACK.append(item)
    return item


__all__ = ["FEEDBACK", "add_feedback"]

