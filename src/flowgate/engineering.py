from __future__ import annotations

import statistics
from typing import Any, Iterable

from .contracts import validate_request
from .contracts import validate_response
from .io import canonical_json
from .privacy import validate_gpu_export


def approximate_tokens(value: Any) -> int:
    """Conservative tokenizer-free estimate used only for the pilot gate."""

    utf8_bytes = len(canonical_json(value).encode("utf-8"))
    return max(1, (utf8_bytes + 3) // 4)


def engineering_stats(
    requests: Iterable[dict[str, Any]],
    responses: Iterable[dict[str, Any]],
    errors: Iterable[dict[str, Any]],
    *,
    invalid_false_pass_count: int | None = None,
) -> dict[str, Any]:
    request_rows = list(requests)
    response_rows = list(responses)
    error_rows = list(errors)
    request_by_id: dict[str, dict[str, Any]] = {}
    success_count = 0
    token_counts: list[int] = []
    for request in request_rows:
        try:
            validate_request(request)
            validate_gpu_export(request)
            episode_id = str(request["episode_id"])
            if episode_id in request_by_id:
                raise ValueError(f"duplicate request ID: {episode_id}")
            request_by_id[episode_id] = request
            success_count += 1
            token_counts.append(approximate_tokens(request))
        except (KeyError, TypeError, ValueError):
            continue
    response_ids: set[str] = set()
    for response in response_rows:
        episode_id = str(response.get("episode_id"))
        if episode_id in response_ids:
            raise ValueError(f"duplicate response ID: {episode_id}")
        if episode_id not in request_by_id:
            raise ValueError(f"unexpected response ID: {episode_id}")
        allowed_ids = {
            str(event["event_id"])
            for event in request_by_id[episode_id]["observed_events"]
        }
        validate_response(
            response, request_by_id[episode_id], allowed_event_ids=allowed_ids
        )
        response_ids.add(episode_id)
    error_ids: set[str] = set()
    for error in error_rows:
        episode_id = str(error.get("episode_id", error.get("case_id")))
        if episode_id in error_ids:
            raise ValueError(f"duplicate error ID: {episode_id}")
        if episode_id not in request_by_id:
            raise ValueError(f"unexpected error ID: {episode_id}")
        if episode_id in response_ids:
            raise ValueError(f"episode has both response and error: {episode_id}")
        error_ids.add(episode_id)
    if invalid_false_pass_count is None:
        invalid_false_pass_count = audit_fixture_negative_controls()
    return {
        "schema_version": "flowgate-engineering-stats/1.0",
        "token_count_method": "ceil(UTF-8 JSON bytes / 4); tokenizer-free estimate",
        "witness_success_count": success_count,
        "invalid_false_pass_count": int(invalid_false_pass_count),
        "median_witness_tokens": statistics.median(token_counts) if token_counts else 0,
        "parse_success_count": len(response_ids),
        "parse_total_count": len(request_by_id),
        "input_contract_valid_count": success_count,
        "request_count": len(request_rows),
        "unaccounted_output_count": len(
            set(request_by_id) - response_ids - error_ids
        ),
    }


def audit_fixture_negative_controls() -> int:
    """Return false passes across six frozen deny/unknown negative controls."""

    from .authz import evaluate_prefix
    from .fixtures import build_fixture_corpus
    from .models import CapabilityState

    episodes, _ = build_fixture_corpus()
    false_passes = 0
    for index in (2, 10, 18):
        result = evaluate_prefix(episodes[index])
        if result.capability_state is not CapabilityState.VERIFIED or result.capability_gain is not False:
            false_passes += 1
    for index in (6, 14, 22):
        result = evaluate_prefix(episodes[index])
        if result.capability_state is not CapabilityState.UNKNOWN or result.capability_gain is True:
            false_passes += 1
    return false_passes
