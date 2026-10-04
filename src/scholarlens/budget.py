"""Shared request-size heuristic; deliberately not a model tokenizer."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence


def estimate_request_tokens(
    messages: Sequence[dict[str, str]],
    model: str,
    *,
    schema: dict[str, object] | None = None,
    estimated_bytes_per_token: int = 6,
    schema_bytes_per_token: int = 32,
    estimate_safety_factor: float = 1.25,
) -> int:
    """Include serialized messages/framing and optional schema with headroom.

    Keep the existing grouped-analysis heuristic unchanged. This does not
    measure the provider's tokenizer or guarantee against server truncation.
    """
    request = {"model": model, "messages": list(messages), "stream": False}
    message_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
    estimate = math.ceil(message_bytes / estimated_bytes_per_token * estimate_safety_factor)
    if schema is not None:
        schema_bytes = len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
        estimate += math.ceil(schema_bytes / schema_bytes_per_token * estimate_safety_factor)
    return estimate
