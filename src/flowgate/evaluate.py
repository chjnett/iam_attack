from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .io import read_json, read_jsonl, write_json
from .metrics import (
    classification_metrics,
    complementarity_counts,
    oracle_routing_at_budget,
    pilot_go_no_go,
    routing_metrics,
)


def _identifier(row: dict[str, Any]) -> str:
    for key in ("episode_id", "case_id", "id"):
        if key in row:
            return str(row[key])
    raise ValueError("row is missing episode_id/case_id/id")


def _truth(row: dict[str, Any]) -> int:
    for key in ("malicious", "label", "is_attack", "ground_truth"):
        if key in row:
            value = row[key]
            if isinstance(value, dict):
                value = value.get("malicious")
            return _verdict(value, allow_abstention=False)  # type: ignore[return-value]
    raise ValueError(f"label row {_identifier(row)!r} is missing malicious/label")


def _verdict(value: Any, *, allow_abstention: bool = True) -> int | None:
    if isinstance(value, dict):
        for key in ("verdict", "malicious", "prediction", "label"):
            if key in value:
                return _verdict(value[key], allow_abstention=allow_abstention)
        raise ValueError("prediction object is missing verdict")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"suspicious", "malicious", "attack", "positive", "true", "1"}:
            return 1
        if normalized in {"authorized", "benign", "normal", "negative", "false", "0"}:
            return 0
        if allow_abstention and normalized in {
            "insufficient_evidence",
            "abstain",
            "unknown",
            "unverifiable",
        }:
            return None
    raise ValueError(f"unsupported verdict: {value!r}")


def _model_output(row: dict[str, Any], role: str) -> dict[str, Any] | None:
    value = row.get(role)
    if value is None:
        value = row.get(f"{role}_prediction", row.get(f"{role}_pred"))
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {"verdict": value}


def _positive_score(output: dict[str, Any] | None, prediction: int | None) -> float | None:
    if output is None or prediction is None:
        return None
    raw_score = output.get(
        "malicious_probability",
        output.get("positive_score", output.get("score")),
    )
    if raw_score is not None:
        score = float(raw_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"positive score outside [0, 1]: {score}")
        return score
    if "confidence" not in output:
        return None
    confidence = float(output["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence outside [0, 1]: {confidence}")
    return confidence if prediction == 1 else 1.0 - confidence


def _route_decision(row: dict[str, Any]) -> bool:
    value = row.get("route_to_remote", row.get("routed", row.get("route", False)))
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"remote", "escalate", "true", "1", "yes"}:
            return True
        if normalized in {"local", "accept", "false", "0", "no"}:
            return False
    raise ValueError(f"unsupported route decision: {value!r}")


def _unique_by_id(rows: Iterable[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _identifier(row)
        if identifier in indexed:
            raise ValueError(f"duplicate {kind} id: {identifier}")
        indexed[identifier] = row
    return indexed


def evaluate_rows(
    labels: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    engineering_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_rows = _unique_by_id(labels, "label")
    prediction_rows = _unique_by_id(predictions, "prediction")
    missing = sorted(set(label_rows) - set(prediction_rows))
    extras = sorted(set(prediction_rows) - set(label_rows))
    if missing or extras:
        raise ValueError(f"label/prediction id mismatch; missing={missing}, extras={extras}")

    blind_ids = sorted(
        identifier
        for identifier, row in label_rows.items()
        if row.get("split") == "blind"
    )
    ordered_ids = blind_ids or sorted(label_rows)
    evaluation_split = "blind" if blind_ids else "all"
    truths: list[int] = []
    local_predictions: list[int | None] = []
    remote_predictions: list[int | None] = []
    final_predictions: list[int | None] = []
    local_scores: list[float | None] = []
    remote_scores: list[float | None] = []
    final_scores: list[float | None] = []
    route_decisions: list[bool] = []

    for identifier in ordered_ids:
        label_row = label_rows[identifier]
        prediction_row = prediction_rows[identifier]
        truth = _truth(label_row)
        local_output = _model_output(prediction_row, "local")
        remote_output = _model_output(prediction_row, "remote")
        final_output = _model_output(prediction_row, "final")
        outer_digest = prediction_row.get("witness_digest")
        for role, output in (("local", local_output), ("remote", remote_output)):
            if output is None:
                continue
            if "episode_id" in output and str(output["episode_id"]) != identifier:
                raise ValueError(f"{identifier}: {role} episode binding mismatch")
            if outer_digest is not None and output.get("witness_digest", outer_digest) != outer_digest:
                raise ValueError(f"{identifier}: {role} witness binding mismatch")
        local = _verdict(local_output) if local_output is not None else None
        remote = _verdict(remote_output) if remote_output is not None else None
        route = _route_decision(prediction_row)
        if route:
            if remote_output is None:
                raise ValueError(f"{identifier}: routed episode has no remote output")
            final = remote
            selected_output = remote_output
        else:
            final = local
            selected_output = local_output
        if final_output is not None and final_output != selected_output:
            raise ValueError(f"{identifier}: final output disagrees with routing decision")
        truths.append(truth)
        local_predictions.append(local)
        remote_predictions.append(remote)
        final_predictions.append(final)
        local_scores.append(_positive_score(local_output, local))
        remote_scores.append(_positive_score(remote_output, remote))
        final_scores.append(_positive_score(selected_output, final))
        route_decisions.append(route)

    def covered_metrics(
        predictions_: list[int | None], scores_: list[float | None]
    ) -> dict[str, Any]:
        indices = [index for index, value in enumerate(predictions_) if value is not None]
        covered_truths = [truths[index] for index in indices]
        covered_predictions = [int(predictions_[index]) for index in indices]
        use_scores = bool(indices) and all(scores_[index] is not None for index in indices)
        covered_scores = [float(scores_[index]) for index in indices] if use_scores else None
        metrics = classification_metrics(
            covered_truths, covered_predictions, covered_scores
        )
        metrics["coverage"] = len(indices) / len(truths) if truths else 0.0
        metrics["abstention_count"] = len(truths) - len(indices)
        return metrics

    local_result = covered_metrics(local_predictions, local_scores)
    remote_result = covered_metrics(remote_predictions, remote_scores)
    final_result = covered_metrics(final_predictions, final_scores)

    remote_output_count = sum(
        _model_output(prediction_rows[identifier], "remote") is not None
        for identifier in ordered_ids
    )
    full_remote_coverage = remote_output_count == len(ordered_ids)
    paired_indices = [
        index
        for index, (local, remote) in enumerate(
            zip(local_predictions, remote_predictions, strict=True)
        )
        if local is not None and remote is not None
    ]
    paired_truths = [truths[index] for index in paired_indices]
    paired_locals = [int(local_predictions[index]) for index in paired_indices]
    paired_remotes = [int(remote_predictions[index]) for index in paired_indices]
    paired_routes = [route_decisions[index] for index in paired_indices]
    complementarity = complementarity_counts(
        paired_truths, paired_locals, paired_remotes
    )
    routed = routing_metrics(
        paired_truths, paired_locals, paired_remotes, paired_routes
    )
    routed["paired_remote_call_count"] = routed["remote_call_count"]
    routed["remote_call_count"] = sum(route_decisions)
    routed["remote_call_rate"] = (
        sum(route_decisions) / len(route_decisions) if route_decisions else 0.0
    )
    routed["local_accept_rate"] = 1.0 - float(routed["remote_call_rate"])
    routed["paid_remote_abstention_count"] = sum(
        route and remote is None
        for route, remote in zip(route_decisions, remote_predictions, strict=True)
    )
    oracle_indices = [
        index for index, local in enumerate(local_predictions) if local is not None
    ]
    oracle_truths = [truths[index] for index in oracle_indices]
    oracle_locals = [int(local_predictions[index]) for index in oracle_indices]
    # A remote abstention is a paid no-op, never silently dropped from the
    # counterfactual denominator or treated as a correction.
    oracle_remotes = [
        int(remote_predictions[index])
        if remote_predictions[index] is not None
        else int(local_predictions[index])
        for index in oracle_indices
    ]
    oracle = oracle_routing_at_budget(
        oracle_truths,
        oracle_locals,
        oracle_remotes,
        maximum_call_rate=0.25,
        call_budget=int(len(truths) * 0.25),
        call_rate_denominator=len(truths),
    )

    result: dict[str, Any] = {
        "corpus_episode_count": len(label_rows),
        "episode_count": len(truths),
        "evaluation_split": evaluation_split,
        "paired_local_remote_count": len(paired_indices),
        "remote_output_count": remote_output_count,
        "full_remote_coverage": full_remote_coverage,
        "local_abstention_count": sum(value is None for value in local_predictions),
        "remote_abstention_count": sum(value is None for value in remote_predictions),
        "oracle_local_decision_count": len(oracle_indices),
        "local": local_result,
        "remote": remote_result,
        "final": final_result,
        "complementarity": complementarity,
        "routing": routed,
        "oracle_at_25_percent": oracle,
        "go_no_go": None,
    }
    if engineering_stats is not None:
        required = {
            "witness_success_count",
            "input_contract_valid_count",
            "invalid_false_pass_count",
            "median_witness_tokens",
            "local_parse_success_count",
            "local_parse_total_count",
            "remote_parse_success_count",
            "remote_parse_total_count",
        }
        missing_stats = required - engineering_stats.keys()
        if missing_stats:
            raise ValueError(f"engineering stats missing fields: {sorted(missing_stats)}")
        gate = pilot_go_no_go(
            episode_count=int(engineering_stats.get("request_count", len(label_rows))),
            witness_success_count=int(engineering_stats["witness_success_count"]),
            invalid_false_pass_count=int(engineering_stats["invalid_false_pass_count"]),
            median_witness_tokens=float(engineering_stats["median_witness_tokens"]),
            parse_success_count=int(engineering_stats["local_parse_success_count"]),
            parse_total_count=int(engineering_stats["local_parse_total_count"]),
            local_errors=int(local_result["fp"]) + int(local_result["fn"]),
            rescue_count=int(complementarity["rescue"]),
            harm_count=int(complementarity["harm"]),
            oracle_call_rate=float(oracle["remote_call_rate"]),
            oracle_mcc_gain=float(oracle["mcc_gain_over_local"]),
            input_contract_valid_count=int(
                engineering_stats["input_contract_valid_count"]
            ),
            local_parse_success_count=int(
                engineering_stats["local_parse_success_count"]
            ),
            local_parse_total_count=int(engineering_stats["local_parse_total_count"]),
            remote_parse_success_count=int(
                engineering_stats["remote_parse_success_count"]
            ),
            remote_parse_total_count=int(
                engineering_stats["remote_parse_total_count"]
            ),
        )
        if not full_remote_coverage:
            gate = {
                "decision": "INCONCLUSIVE",
                "passed": False,
                "failed_checks": ["full_remote_coverage"],
                "reason": (
                    "Complementarity and Oracle gates require an independent remote "
                    "output for every evaluated blind episode."
                ),
                "remote_output_count": remote_output_count,
                "required_remote_output_count": len(ordered_ids),
                "provisional_gate": gate,
            }
        result["go_no_go"] = gate
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FlowGate pilot predictions")
    parser.add_argument("--labels", required=True, help="Ground-truth labels JSONL")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL")
    parser.add_argument(
        "--engineering-stats",
        help="Optional JSON with witness/parse engineering counters",
    )
    parser.add_argument("--output", help="Optional report JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engineering_stats = (
        read_json(args.engineering_stats) if args.engineering_stats else None
    )
    report = evaluate_rows(
        read_jsonl(args.labels),
        read_jsonl(args.predictions),
        engineering_stats=engineering_stats,
    )
    if args.output:
        write_json(Path(args.output), report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
