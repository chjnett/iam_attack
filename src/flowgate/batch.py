from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    prompt_hash,
    validate_request,
    validate_response,
    validate_run_record,
)
from .inference import Backend
from .privacy import validate_gpu_export


def _id(row: dict[str, Any]) -> str:
    value = row.get("episode_id", row.get("case_id"))
    if value is None:
        raise ValueError("row has no episode_id")
    return str(value)


def _unique_index(
    rows: Iterable[dict[str, Any]], kind: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _id(row)
        if identifier in indexed:
            raise ValueError(f"duplicate {kind} ID: {identifier}")
        indexed[identifier] = row
    return indexed


def _decision_value(row: dict[str, Any]) -> bool:
    """Return a routing decision without Python truthiness coercion."""

    if "route_to_remote" in row:
        value = row["route_to_remote"]
    elif "selected" in row:
        value = row["selected"]
    else:
        raise ValueError(f"routing decision for {_id(row)} has no boolean decision")
    if not isinstance(value, bool):
        raise ValueError(f"routing decision for {_id(row)} must be a JSON boolean")
    return value


def allowed_event_ids(request: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for event in request.get("observed_events", []):
        if isinstance(event, dict) and event.get("event_id") is not None:
            ids.add(str(event["event_id"]))
    return ids


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def run_backend(
    requests: Iterable[dict[str, Any]],
    *,
    backend: Backend,
    prompt: str,
    run_id: str = "manual-run",
    model_role: str = "local",
    prompt_version: str = "local_v1",
    pricing: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a batch, retain first-attempt failures, and create bound run records."""

    if model_role not in {"local", "remote"}:
        raise ValueError("model_role must be local or remote")
    responses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for request in requests:
        episode_id = str(request.get("episode_id", "<missing>"))
        started_at = dt.datetime.now(dt.timezone.utc)
        try:
            validate_request(request)
            validate_gpu_export(request)
            response = backend.predict(request, prompt)
            validate_response(
                response,
                request,
                allowed_event_ids=allowed_event_ids(request),
            )
            finished_at = dt.datetime.now(dt.timezone.utc)
            metadata = dict(getattr(backend, "last_metadata", {}))
            input_tokens = metadata.get("input_tokens")
            output_tokens = metadata.get("output_tokens")
            if pricing and input_tokens is not None and output_tokens is not None:
                amount = (
                    float(input_tokens) * float(pricing["input_per_million_tokens"])
                    + float(output_tokens) * float(pricing["output_per_million_tokens"])
                ) / 1_000_000
                cost = {
                    "amount": amount,
                    "currency": str(pricing["currency"]).upper(),
                    "pricing_snapshot": str(pricing["pricing_snapshot"]),
                }
            else:
                cost = {"amount": None, "currency": None, "pricing_snapshot": None}
            responses.append(response)
            record = {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "model_role": model_role,
                    "model_id": backend.model_id,
                    "prompt_version": prompt_version,
                    "prompt_sha256": prompt_hash(prompt),
                    "witness_digest": request["witness_digest"],
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "latency_ms": metadata.get("latency_ms"),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "validation": {
                        "schema_valid": True,
                        "binding_valid": True,
                        "citation_valid": True,
                        "errors": [],
                    },
                    "output": response,
                }
            validate_run_record(record, request)
            records.append(record)
        except Exception as exc:  # First attempt remains visible; no silent retry.
            raw = getattr(backend, "last_raw_response", None)
            raw_text = str(raw) if raw is not None else ""
            errors.append(
                {
                    "episode_id": episode_id,
                    "run_id": run_id,
                    "model_role": model_role,
                    "model_id": backend.model_id,
                    "prompt_sha256": prompt_hash(prompt),
                    "witness_digest": request.get("witness_digest"),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "raw_response_sha256": (
                        hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                        if raw_text
                        else None
                    ),
                    "raw_response_preview": raw_text[:2000] if raw_text else None,
                    "metadata": dict(getattr(backend, "last_metadata", {})),
                    "started_at": started_at.isoformat(),
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
    return responses, errors, records


def selected_requests(
    requests: Iterable[dict[str, Any]],
    decisions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {
        _id(row)
        for row in decisions
        if _decision_value(row)
    }
    return [row for row in requests if _id(row) in selected]


def merge_predictions(
    requests: Iterable[dict[str, Any]],
    local_responses: Iterable[dict[str, Any]],
    decisions: Iterable[dict[str, Any]],
    remote_responses: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_rows = list(requests)
    request_by_id = _unique_index(request_rows, "request")
    local = _unique_index(local_responses, "local response")
    remote = _unique_index(remote_responses, "remote response")
    decision_by_id = _unique_index(decisions, "routing decision")
    request_ids = set(request_by_id)
    if set(local) != request_ids:
        raise ValueError("local response IDs must match request IDs exactly")
    if not set(remote) <= request_ids:
        raise ValueError(f"unexpected remote response IDs: {sorted(set(remote) - request_ids)}")
    if not set(decision_by_id) <= request_ids:
        raise ValueError(
            f"unexpected routing decision IDs: {sorted(set(decision_by_id) - request_ids)}"
        )
    for episode_id, request in request_by_id.items():
        validate_request(request)
        validate_gpu_export(request)
        validate_response(
            local[episode_id], request, allowed_event_ids=allowed_event_ids(request)
        )
        if episode_id in remote:
            validate_response(
                remote[episode_id], request, allowed_event_ids=allowed_event_ids(request)
            )
    merged: list[dict[str, Any]] = []
    for request in request_rows:
        episode_id = _id(request)
        if episode_id not in local:
            raise ValueError(f"missing local response for {episode_id}")
        decision = decision_by_id.get(
            episode_id,
            {"episode_id": episode_id, "route_to_remote": False, "policy": "never"},
        )
        routed = _decision_value(decision)
        remote_response = remote.get(episode_id)
        if routed and remote_response is None:
            raise ValueError(f"episode {episode_id} was routed but has no remote response")
        final = remote_response if routed else local[episode_id]
        merged.append(
            {
                "episode_id": episode_id,
                "witness_digest": request["witness_digest"],
                "local": local[episode_id],
                "route_to_remote": routed,
                "routing": decision,
                "remote": remote_response,
                "oracle_called": routed,
                "final": final,
            }
        )
    return merged
