from __future__ import annotations


def sanitize_output(data):
    if isinstance(data, dict):
        return {k: sanitize_output(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(v) for v in data]
    elif isinstance(data, bool):
        return "Yes" if data else "No"
    else:
        return data

