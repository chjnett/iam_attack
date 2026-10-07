"""Build prefix-safe, label-blind witnesses from deterministic IAM episodes."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import json
from typing import Any, Mapping

from .authz import evaluate_prefix
from .models import Episode, Witness


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_actions(candidate_path: tuple[str, ...]) -> tuple[str, ...]:
    actions: list[str] = []
    for token in candidate_path:
        if ":" not in token:
            continue
        service, _ = token.split(":", 1)
        if service.endswith("-session") or not service:
            continue
        actions.append(token)
    return tuple(actions)


def _cross_service_count(actions: tuple[str, ...]) -> int:
    services = [action.split(":", 1)[0].lower() for action in actions]
    return sum(left != right for left, right in zip(services, services[1:]))


def build_witness(episode: Episode, prefix_len: int | None = None) -> Witness:
    """Create a witness using no event after ``prefix_len`` and no label input."""

    observed_events = episode.prefix(prefix_len)
    resolved_prefix_len = len(observed_events)
    result = evaluate_prefix(episode, resolved_prefix_len)
    path_actions = _path_actions(result.candidate_path)
    observed_actions = {event.action for event in observed_events}
    observed_path_actions = sum(
        1 for action in path_actions if action in observed_actions
    )
    observed_edge_ratio = (
        observed_path_actions / len(path_actions) if path_actions else 0.0
    )
    completeness = (
        result.satisfied_preconditions / result.required_preconditions
        if result.required_preconditions
        else 0.0
    )
    prefix_fraction = (
        resolved_prefix_len / len(episode.events) if episode.events else 1.0
    )
    feature_summary = tuple(
        sorted(
            {
                "cross_service_count": _cross_service_count(path_actions),
                "observed_edge_ratio": round(observed_edge_ratio, 4),
                "observed_event_count": resolved_prefix_len,
                "path_length": len(result.candidate_path),
                "prefix_fraction": round(prefix_fraction, 4),
                "sensitive_action_observed": result.sensitive_action_observed,
                "unknown_precondition_count": len(result.unknown_preconditions),
                "witness_completeness": round(completeness, 4),
            }.items()
        )
    )
    unsigned = Witness(
        episode_id=episode.episode_id,
        family=episode.family,
        split=episode.split,
        prefix_len=resolved_prefix_len,
        cutoff=episode.cutoff(resolved_prefix_len),
        capability_state=result.capability_state,
        capability_gain=result.capability_gain,
        provenance=result.provenance,
        observed_events=observed_events,
        candidate_path=result.candidate_path,
        unknown_preconditions=result.unknown_preconditions,
        feature_summary=feature_summary,
        witness_digest="",
    )
    payload = unsigned.to_dict()
    payload.pop("witness_digest")
    return replace(unsigned, witness_digest=_canonical_digest(payload))


def verify_witness_digest(witness: Witness | Mapping[str, Any]) -> bool:
    """Verify the embedded lowercase SHA-256 digest without mutating input."""

    serialized = witness.to_dict() if isinstance(witness, Witness) else dict(witness)
    supplied = serialized.pop("witness_digest", None)
    if not isinstance(supplied, str) or len(supplied) != 64:
        return False
    if any(character not in "0123456789abcdef" for character in supplied):
        return False
    expected = _canonical_digest(serialized)
    return hmac.compare_digest(supplied, expected)


__all__ = ["build_witness", "verify_witness_digest"]
