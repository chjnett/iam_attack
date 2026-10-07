from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import validate_request, validate_response


SUPPORTED_POLICIES = {"never", "always", "random", "uncertainty", "flowgate-v0"}


@dataclass(frozen=True)
class RouteDecision:
    episode_id: str
    policy: str
    score: float
    selected: bool
    rank: int
    budget_size: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "policy": self.policy,
            "score": round(self.score, 6),
            "route_to_remote": self.selected,
            "rank": self.rank,
            "budget_size": self.budget_size,
            "reasons": list(self.reasons),
        }


def _stable_random(case_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _unknown_count(witness: dict[str, Any]) -> int:
    explicit = witness.get("unknown_preconditions", [])
    count = len(explicit) if isinstance(explicit, list) else 0
    provenance = witness.get("provenance")
    if isinstance(provenance, dict):
        count += sum(1 for value in provenance.values() if value == "unknown")
    elif isinstance(provenance, list):
        count += sum(
            1
            for value in provenance
            if value == "unknown"
            or (isinstance(value, dict) and value.get("state") == "unknown")
        )
    return count


def _path_length(witness: dict[str, Any]) -> int:
    path = witness.get("candidate_path_edges", witness.get("candidate_path", []))
    return len(path) if isinstance(path, list) else 0


def _bounded_feature(features: dict[str, Any], name: str, default: float) -> float:
    value = features.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return default
    return max(0.0, min(1.0, float(value)))


def _evidence_completeness(request: dict[str, Any]) -> float:
    """Summarize how much of the already-selected witness is observable.

    A second model receives the same witness, so this measures whether model
    reasoning (rather than additional data collection) can plausibly help.
    """

    features = request.get("feature_summary", {})
    if not isinstance(features, dict):
        features = {}
    witness_completeness = _bounded_feature(
        features, "witness_completeness", 0.0
    )
    observed_edge_ratio = _bounded_feature(features, "observed_edge_ratio", 0.0)
    prefix_fraction = _bounded_feature(features, "prefix_fraction", 0.0)
    return (
        0.50 * witness_completeness
        + 0.30 * observed_edge_ratio
        + 0.20 * prefix_fraction
    )


def _path_complexity(request: dict[str, Any]) -> tuple[float, int, int]:
    features = request.get("feature_summary", {})
    if not isinstance(features, dict):
        features = {}
    path_length = _path_length(request)
    cross_service_count = features.get("cross_service_count", 0)
    if not isinstance(cross_service_count, int) or isinstance(
        cross_service_count, bool
    ):
        cross_service_count = 0

    # A two-node path is the simplest direct candidate. Extra hops and service
    # boundaries create semantic work that a stronger model can plausibly fix.
    path_component = min(1.0, max(0, path_length - 2) / 4.0)
    cross_service_component = min(1.0, max(0, cross_service_count) / 2.0)
    complexity = 0.65 * path_component + 0.35 * cross_service_component
    return complexity, path_length, cross_service_count


def score_request(
    request: dict[str, Any],
    local_response: dict[str, Any],
    *,
    policy: str,
    seed: int = 7,
) -> tuple[float, tuple[str, ...]]:
    """Return a label-blind priority score for scarce remote-LLM calls.

    `flowgate-v0` is deliberately an interpretable pilot heuristic, not the
    paper's final learned router.  It only consumes the witness and local-model
    output available at decision time.
    """

    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported routing policy: {policy}")
    episode_id = str(request["episode_id"])
    if policy == "never":
        return -1.0, ("never-route baseline",)
    if policy == "always":
        return 1.0, ("always-route baseline",)
    if policy == "random":
        return _stable_random(episode_id, seed), ("seeded random baseline",)

    if "confidence" in local_response:
        confidence = float(local_response["confidence"])
    else:
        probability = float(local_response.get("malicious_probability", 0.5))
        confidence = abs(probability - 0.5) * 2.0
    uncertainty = max(0.0, min(1.0, 1.0 - confidence))
    abstain = bool(local_response.get("abstain", False)) or (
        local_response.get("verdict") in {"insufficient_evidence", "abstain"}
    )
    if policy == "uncertainty":
        score = uncertainty + (0.35 if abstain else 0.0)
        reasons = [f"local_uncertainty={uncertainty:.3f}"]
        if abstain:
            reasons.append("local_abstention")
        return score, tuple(reasons)

    capability = request.get("capability_state", "unknown")
    capability_resolvability = {
        "verified": 1.0,
        "possible": 0.85,
        "unknown": 0.15,
    }.get(
        str(capability), 0.15
    )
    unknowns = _unknown_count(request)
    unknown_penalty = min(1.0, unknowns / 3.0)
    completeness = _evidence_completeness(request)
    complexity, path_length, cross_service_count = _path_complexity(request)

    # Marginal-rescue estimate. High local uncertainty is useful only when the
    # shared evidence is complete enough for another model to reason over it.
    # Complexity is a smaller positive signal because longer/cross-service
    # paths can exceed a small model's reasoning capacity. Missing capability
    # or precondition state is explicitly penalized: the remote model receives
    # no extra evidence and therefore cannot repair those gaps.
    local_need = min(1.0, 0.80 * uncertainty + 0.20 * float(abstain))
    recoverability = (
        completeness * capability_resolvability * (1.0 - unknown_penalty)
    )
    marginal_rescue = recoverability * (0.78 * local_need + 0.22 * complexity)
    score = (
        marginal_rescue
        - 0.30 * unknown_penalty
        - 0.25 * float(capability == "unknown")
    )
    reasons = [
        f"local_uncertainty={uncertainty:.3f}",
        f"evidence_completeness={completeness:.3f}",
        f"recoverability={recoverability:.3f}",
        f"path_complexity={complexity:.3f}",
        f"capability={capability}",
    ]
    if abstain:
        reasons.append("local_abstention")
    if unknowns:
        reasons.append(f"irreducible_unknowns={unknowns}")
    if path_length:
        reasons.append(f"candidate_path_nodes={path_length}")
    if cross_service_count:
        reasons.append(f"cross_service_count={cross_service_count}")
    return score, tuple(reasons)


def select_remote_calls(
    requests: Iterable[dict[str, Any]],
    local_responses: Iterable[dict[str, Any]],
    *,
    policy: str,
    budget_fraction: float,
    seed: int = 7,
    call_budget: int | None = None,
) -> list[RouteDecision]:
    if not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be between 0 and 1")
    if call_budget is not None and (
        not isinstance(call_budget, int)
        or isinstance(call_budget, bool)
        or call_budget < 0
    ):
        raise ValueError("call_budget must be a non-negative integer")
    request_list = list(requests)
    request_by_id: dict[str, dict[str, Any]] = {}
    for request in request_list:
        validate_request(request)
        episode_id = str(request["episode_id"])
        if episode_id in request_by_id:
            raise ValueError(f"duplicate request ID: {episode_id}")
        request_by_id[episode_id] = request
    local_by_id: dict[str, dict[str, Any]] = {}
    for row in local_responses:
        episode_id = str(row.get("episode_id", row.get("case_id")))
        if episode_id in local_by_id:
            raise ValueError(f"duplicate local response ID: {episode_id}")
        if episode_id not in request_by_id:
            raise ValueError(f"unexpected local response ID: {episode_id}")
        allowed_ids = {
            str(event["event_id"])
            for event in request_by_id[episode_id]["observed_events"]
        }
        validate_response(row, request_by_id[episode_id], allowed_event_ids=allowed_ids)
        local_by_id[episode_id] = row
    if set(local_by_id) != {str(row["episode_id"]) for row in request_list}:
        raise ValueError("local responses must match request case IDs exactly")

    ranked: list[tuple[str, float, tuple[str, ...]]] = []
    for request in request_list:
        case_id = str(request["episode_id"])
        score, reasons = score_request(
            request,
            local_by_id[case_id],
            policy=policy,
            seed=seed,
        )
        ranked.append((case_id, score, reasons))
    ranked.sort(key=lambda item: (-item[1], item[0]))

    if policy == "never":
        budget_size = 0
    elif policy == "always":
        budget_size = len(ranked)
    elif call_budget is not None:
        budget_size = min(len(ranked), call_budget)
    else:
        budget_size = min(len(ranked), math.floor(len(ranked) * budget_fraction))
    selected_ids = {case_id for case_id, _, _ in ranked[:budget_size]}
    return [
        RouteDecision(
            episode_id=case_id,
            policy=policy,
            score=score,
            selected=case_id in selected_ids,
            rank=rank,
            budget_size=budget_size,
            reasons=reasons,
        )
        for rank, (case_id, score, reasons) in enumerate(ranked, start=1)
    ]
