from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

from .contracts import (
    validate_label_row,
    validate_request,
    validate_response,
    validate_run_record,
)
from .evaluate import _truth, _verdict, _identifier, _positive_score
from .metrics import classification_metrics
from .routing import select_remote_calls


ROUTER_POLICIES = ("random", "uncertainty", "flowgate-v0")


def static_witness_only_decision(request: dict[str, Any]) -> str:
    """Apply the transparent non-LLM witness baseline.

    The rule intentionally consumes only facts in the validated worker request.
    Unknown reachability is never coerced to a binary answer.  ``possible`` is
    also not strong enough to raise an attack finding: all three positive facts
    must be explicitly present and verified.
    """

    validate_request(request)
    if request["capability_state"] == "unknown":
        return "abstain"
    if (
        request["capability_state"] == "verified"
        and request["capability_gain"] is True
        and request["feature_summary"]["sensitive_action_observed"] is True
    ):
        return "attack"
    return "benign"


def _index_exact(rows: Iterable[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _identifier(row)
        if identifier in indexed:
            raise ValueError(f"duplicate {kind} ID: {identifier}")
        indexed[identifier] = row
    return indexed


def _finite_nonnegative(value: Any, field: str, episode_id: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{episode_id}: {field} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{episode_id}: {field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{episode_id}: {field} must be finite and non-negative")
    return result


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "median_ms": statistics.median(values) if values else None,
        "p95_ms": _quantile(values, 0.95),
        "total_ms": sum(values),
    }


def _static_witness_only_report(
    labels: dict[str, dict[str, Any]],
    requests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return label-unsealed descriptive metrics for the non-LLM rule.

    This result deliberately has no cost or latency fields and is never passed
    to ``_mark_frontier``.  It is a comparison with static IAM/witness practice,
    not another LLM routing policy.
    """

    truths: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    decision_counts = {"attack": 0, "benign": 0, "abstain": 0}
    for episode_id in sorted(labels):
        decision = static_witness_only_decision(requests[episode_id])
        decision_counts[decision] += 1
        if decision == "abstain":
            continue
        truths.append(_truth(labels[episode_id]))
        prediction = int(decision == "attack")
        predictions.append(prediction)
        scores.append(float(prediction))

    count = len(labels)
    metrics = classification_metrics(truths, predictions, scores)
    return {
        "baseline_id": "static_witness_only",
        "display_name": "static/witness-only",
        "baseline_type": "transparent_non_llm_rule",
        "input_scope": "validated_trusted_request_facts_only",
        "decision_rule": (
            "attack iff capability_state=verified AND capability_gain=true AND "
            "sensitive_action_observed=true; abstain iff capability_state=unknown; "
            "otherwise benign"
        ),
        "decision_counts": decision_counts,
        "coverage": len(predictions) / count if count else 0.0,
        "abstention_count": decision_counts["abstain"],
        "covered_metrics": metrics,
        "included_in_cost_mcc_frontier": False,
        "llm_api_use": "none",
        "runtime_measurement": "not_collected",
    }


def _row_metrics(
    labels: dict[str, dict[str, Any]],
    requests: dict[str, dict[str, Any]],
    local: dict[str, dict[str, Any]],
    remote: dict[str, dict[str, Any]],
    remote_records: dict[str, dict[str, Any]],
    selected_ids: set[str],
    *,
    policy: str,
    budget: int,
    currency: str,
) -> dict[str, Any]:
    ordered_ids = sorted(labels)
    truths: list[int] = []
    final: list[int | None] = []
    final_scores: list[float | None] = []
    attack_count = 0
    attack_rescue_opportunities = 0
    missed_attack_rescues = 0
    selected_cost = 0.0
    selected_remote_latencies: list[float] = []

    for episode_id in ordered_ids:
        truth = _truth(labels[episode_id])
        attack_count += int(truth == 1)
        local_output = local[episode_id]
        remote_output = remote[episode_id]
        local_verdict = _verdict(local_output)
        remote_verdict = _verdict(remote_output)
        use_remote = episode_id in selected_ids
        final_output = remote_output if use_remote else local_output
        final_verdict = remote_verdict if use_remote else local_verdict
        score = _positive_score(final_output, final_verdict)
        truths.append(truth)
        final.append(final_verdict)
        final_scores.append(score)

        if truth == 1 and local_verdict == 0 and remote_verdict == 1:
            attack_rescue_opportunities += 1
            if not use_remote:
                missed_attack_rescues += 1

        if use_remote:
            record = remote_records[episode_id]
            selected_cost += _finite_nonnegative(
                record["cost"]["amount"], "remote cost", episode_id
            )
            selected_remote_latencies.append(_finite_nonnegative(
                record["latency_ms"], "remote latency", episode_id
            ))

    covered_indices = [i for i, verdict in enumerate(final) if verdict is not None]
    covered_truths = [truths[i] for i in covered_indices]
    covered_verdicts = [int(final[i]) for i in covered_indices]
    covered_scores = [final_scores[i] for i in covered_indices]
    scores_arg = (
        [float(score) for score in covered_scores]
        if covered_scores and all(score is not None for score in covered_scores)
        else None
    )
    metrics = classification_metrics(covered_truths, covered_verdicts, scores_arg)
    count = len(ordered_ids)
    remote_all_latency = _latency_summary([
        _finite_nonnegative(remote_records[episode_id]["latency_ms"], "remote latency", episode_id)
        for episode_id in ordered_ids
    ])
    remote_latency = _latency_summary(selected_remote_latencies)
    return {
        "policy": policy,
        "budget": budget,
        "calls": len(selected_ids),
        "call_rate": len(selected_ids) / count if count else 0.0,
        "cost": selected_cost,
        "currency": currency,
        "mcc": metrics["mcc"],
        "fnr": metrics["fnr"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "average_precision": metrics["average_precision"],
        "coverage": len(covered_indices) / count if count else 0.0,
        "abstention_count": count - len(covered_indices),
        "missed_rescue_risk": missed_attack_rescues / attack_count if attack_count else 0.0,
        "attack_rescue_opportunities": attack_rescue_opportunities,
        "missed_attack_rescues": missed_attack_rescues,
        "latency": {
            "local_all": {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "total_ms": None},
            "remote_all_counterfactual": remote_all_latency,
            "remote_selected": remote_latency,
            "sequential_total_mean_ms": None,
        },
    }


def _mark_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other["cost"] <= row["cost"]
                and other["mcc"] >= row["mcc"]
                and other["fnr"] <= row["fnr"]
                and other["coverage"] >= row["coverage"]
                and other["missed_rescue_risk"] <= row["missed_rescue_risk"]
            )
            strictly_better = (
                other["cost"] < row["cost"]
                or other["mcc"] > row["mcc"]
                or other["fnr"] < row["fnr"]
                or other["coverage"] > row["coverage"]
                or other["missed_rescue_risk"] < row["missed_rescue_risk"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return rows


def analyze_pareto(
    requests: Iterable[dict[str, Any]],
    blind_labels: Iterable[dict[str, Any]],
    local_responses: Iterable[dict[str, Any]],
    remote_responses: Iterable[dict[str, Any]],
    remote_run_records: Iterable[dict[str, Any]],
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Sweep exact remote-call budgets and return cost/quality Pareto results.

    Inputs must contain only the blind evaluation split. The remote responses
    and run records are a complete remote-all counterfactual; selecting a small
    call budget in this analysis does not change the cost already incurred to
    obtain those counterfactual outputs.
    """

    request_rows = list(requests)
    request_by_id = _index_exact(request_rows, "request")
    if not request_by_id:
        raise ValueError("Pareto analysis requires at least one blind episode")
    for request in request_rows:
        validate_request(request)

    label_by_id = _index_exact(blind_labels, "label")
    if set(label_by_id) != set(request_by_id):
        raise ValueError("blind labels must match request IDs exactly")
    if any(row.get("split") != "blind" for row in label_by_id.values()):
        raise ValueError("Pareto labels must all have split='blind'")
    for row in label_by_id.values():
        validate_label_row(row)

    local_by_id = _index_exact(local_responses, "local response")
    remote_by_id = _index_exact(remote_responses, "remote response")
    if set(local_by_id) != set(request_by_id):
        raise ValueError("local responses must match request IDs exactly")
    if set(remote_by_id) != set(request_by_id):
        raise ValueError("remote-all responses must match request IDs exactly")
    for episode_id, request in request_by_id.items():
        allowed_events = {str(event["event_id"]) for event in request["observed_events"]}
        validate_response(local_by_id[episode_id], request, allowed_event_ids=allowed_events)
        validate_response(remote_by_id[episode_id], request, allowed_event_ids=allowed_events)

    records_by_id = _index_exact(remote_run_records, "remote run record")
    if set(records_by_id) != set(request_by_id):
        raise ValueError("remote run records must match request IDs exactly")
    currencies: set[str] = set()
    for episode_id, record in records_by_id.items():
        request = request_by_id[episode_id]
        if record.get("model_role") != "remote":
            raise ValueError(f"{episode_id}: run record model_role must be remote")
        validate_run_record(record, request)
        if str(record.get("episode_id")) != episode_id:
            raise ValueError(f"{episode_id}: run record episode binding mismatch")
        if record.get("witness_digest") != request["witness_digest"]:
            raise ValueError(f"{episode_id}: run record witness binding mismatch")
        output = record.get("output")
        if not isinstance(output, dict) or output != remote_by_id[episode_id]:
            raise ValueError(f"{episode_id}: run record output does not match remote response")
        validate_response(
            output,
            request,
            allowed_event_ids={str(event["event_id"]) for event in request["observed_events"]},
        )
        validation = record.get("validation", {})
        if not all(validation.get(key) is True for key in ("schema_valid", "binding_valid", "citation_valid")):
            raise ValueError(f"{episode_id}: remote record validation flags are not all true")
        cost = record.get("cost")
        if not isinstance(cost, dict):
            raise ValueError(f"{episode_id}: missing remote cost record")
        amount = _finite_nonnegative(cost.get("amount"), "remote cost", episode_id)
        currency = cost.get("currency")
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError(f"{episode_id}: remote cost currency is required")
        currencies.add(currency.strip().upper())
        latency = _finite_nonnegative(record.get("latency_ms"), "remote latency", episode_id)
    if len(currencies) != 1:
        raise ValueError(f"remote run records must use one currency, got {sorted(currencies)}")
    currency = next(iter(currencies))

    rows: list[dict[str, Any]] = []
    count = len(request_by_id)
    for policy in ROUTER_POLICIES:
        for budget in range(count + 1):
            decisions = select_remote_calls(
                request_rows,
                local_by_id.values(),
                policy=policy,
                budget_fraction=budget / count,
                seed=seed,
                call_budget=budget,
            )
            selected_ids = {decision.episode_id for decision in decisions if decision.selected}
            if len(selected_ids) != budget:
                raise AssertionError(f"{policy} selected {len(selected_ids)} calls for exact budget {budget}")
            rows.append(_row_metrics(
                label_by_id,
                request_by_id,
                local_by_id,
                remote_by_id,
                records_by_id,
                selected_ids,
                policy=policy,
                budget=budget,
                currency=currency,
            ))

    never_decisions = select_remote_calls(
        request_rows, local_by_id.values(), policy="never", budget_fraction=0.0, seed=seed
    )
    rows.append(_row_metrics(
        label_by_id, request_by_id, local_by_id, remote_by_id, records_by_id,
        {decision.episode_id for decision in never_decisions if decision.selected},
        policy="never", budget=0, currency=currency,
    ))
    always_decisions = select_remote_calls(
        request_rows, local_by_id.values(), policy="always", budget_fraction=1.0, seed=seed
    )
    rows.append(_row_metrics(
        label_by_id, request_by_id, local_by_id, remote_by_id, records_by_id,
        {decision.episode_id for decision in always_decisions if decision.selected},
        policy="always", budget=count, currency=currency,
    ))
    _mark_frontier(rows)
    return {
        "schema_version": "flowgate-pareto/1.1",
        "episode_count": count,
        "currency": currency,
        "remote_all_actual_cost": sum(
            _finite_nonnegative(records_by_id[episode_id]["cost"]["amount"], "remote cost", episode_id)
            for episode_id in request_by_id
        ),
        "seed": seed,
        "remote_counterfactual": "complete remote-all run; cost below is simulated selected-call cost",
        "descriptive_baselines": {
            "static_witness_only": _static_witness_only_report(
                label_by_id, request_by_id
            )
        },
        "rows": rows,
        "frontier": [row for row in rows if row["pareto_optimal"]],
    }


def pareto_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten report rows for CSV writing without a CSV dependency."""

    columns = (
        "policy", "budget", "calls", "call_rate", "cost", "currency", "mcc", "fnr",
        "precision", "recall", "average_precision", "coverage", "abstention_count",
        "missed_rescue_risk", "attack_rescue_opportunities", "missed_attack_rescues",
        "pareto_optimal",
    )
    return [{key: row.get(key) for key in columns} for row in report.get("rows", [])]
