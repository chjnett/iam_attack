from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
from typing import Any

from .io import canonical_json


REQUEST_SCHEMA_VERSION = "1.0"
RESPONSE_SCHEMA_VERSION = "1.0"

ALLOWED_CAPABILITY_STATES = {"verified", "possible", "unknown"}
ALLOWED_FAMILIES = {
    "policy_attachment_abuse",
    "role_trust_abuse",
    "passrole_compute_abuse",
}
ALLOWED_VERDICTS = {"attack", "benign", "abstain"}
ALLOWED_PATH_ASSESSMENTS = {
    "supports_attack",
    "supports_benign_explanation",
    "insufficient",
}
ALLOWED_SIGNALS = {
    "capability_gain",
    "sensitive_action_after_gain",
    "observed_role_assumption",
    "cross_service_chain",
    "temporal_proximity",
    "matched_administrative_pattern",
    "missing_precondition",
    "incomplete_observation",
    "conflicting_evidence",
    "no_attack_signal",
}
ALLOWED_UNCERTAINTY_REASONS = {
    "missing_state",
    "missing_event",
    "unknown_precondition",
    "ambiguous_intent",
    "conflicting_evidence",
    "out_of_scope_pattern",
    "none",
}

WORKER_INPUT_FIELDS = {
    "schema_version",
    "episode_id",
    "family",
    "prefix_len",
    "cutoff",
    "capability_state",
    "capability_gain",
    "provenance",
    "observed_events",
    "candidate_path",
    "unknown_preconditions",
    "feature_summary",
    "witness_digest",
}

MODEL_OUTPUT_FIELDS = {
    "schema_version",
    "episode_id",
    "witness_digest",
    "family",
    "verdict",
    "malicious_probability",
    "candidate_path_assessment",
    "cited_event_ids",
    "signals",
    "uncertainty_reasons",
    "summary",
}

EVENT_FIELDS = {
    "event_id",
    "timestamp",
    "action",
    "actor",
    "target",
    "resource",
    "outcome",
    "details",
}

FEATURE_FIELDS = {
    "path_length",
    "cross_service_count",
    "observed_event_count",
    "prefix_fraction",
    "observed_edge_ratio",
    "witness_completeness",
    "unknown_precondition_count",
    "sensitive_action_observed",
}

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]{1,96}$")
SAFE_ENTITY = re.compile(
    r"^(?:[PRXYC][A-Z0-9]{3,}|(?:role|service)-session:R[A-Z0-9]{3,}|)$"
)
SAFE_ACTION = re.compile(r"^[a-z0-9-]{1,64}:[A-Za-z0-9*]+$")
SAFE_PROVENANCE = re.compile(
    r"^(?:snapshot|event|state|observed|capability_state|baseline_capability):[A-Za-z0-9._:-]+$"
)
SAFE_UNKNOWN = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
OPERATIONAL_DETAIL_KEYS = {
    "change_ticket_present",
    "approved_actor_match",
    "maintenance_window_match",
    "rollback_observed",
    "break_glass",
}
REFERENCE_DETAIL_PATTERNS = {
    "policy_id": re.compile(r"^Y[A-Z0-9]{3,}$"),
    "added_principal": re.compile(r"^P[A-Z0-9]{3,}$"),
    "execution_role": re.compile(r"^R[A-Z0-9]{3,}$"),
}
ALLOWED_DETAIL_KEYS = (
    OPERATIONAL_DETAIL_KEYS
    | set(REFERENCE_DETAIL_PATTERNS)
    | {"trust_condition", "code_controlled"}
)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def witness_digest(worker_input_without_digest: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(worker_input_without_digest).encode("utf-8")
    ).hexdigest()


def _parse_datetime(value: Any, field_name: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an RFC3339 date-time string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def make_request(
    *,
    case_id: str,
    cutoff_time: str,
    witness: dict[str, Any],
) -> dict[str, Any]:
    """Create the exact label-blind object allowed to leave the laptop.

    The historical function name is retained for CLI stability. The returned
    object is the direct worker input, not a transport envelope.
    """

    public = dict(witness)
    for private_key in (
        "split",
        "ground_truth",
        "label",
        "malicious",
        "expected_verdict",
        "witness_digest",
    ):
        public.pop(private_key, None)
    public.setdefault("schema_version", REQUEST_SCHEMA_VERSION)
    public.setdefault("episode_id", case_id)
    public.setdefault("cutoff", cutoff_time)
    if str(public["episode_id"]) != str(case_id):
        raise ValueError("case_id does not match witness episode_id")
    if str(public["cutoff"]) != str(cutoff_time):
        raise ValueError("cutoff_time does not match witness cutoff")
    public["witness_digest"] = witness_digest(public)
    return public


def validate_request(request: dict[str, Any]) -> None:
    missing = WORKER_INPUT_FIELDS - request.keys()
    extras = request.keys() - WORKER_INPUT_FIELDS
    if missing:
        raise ValueError(f"worker input missing fields: {sorted(missing)}")
    if extras:
        raise ValueError(f"worker input has unexpected fields: {sorted(extras)}")
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported worker-input schema")
    if not SAFE_EPISODE_ID.fullmatch(str(request["episode_id"])):
        raise ValueError("invalid episode_id")
    if request["family"] not in ALLOWED_FAMILIES:
        raise ValueError("invalid family")
    if not _is_integer(request["prefix_len"]) or request["prefix_len"] < 1:
        raise ValueError("prefix_len must be a positive integer")
    cutoff = _parse_datetime(request["cutoff"], "cutoff")
    if request["capability_state"] not in ALLOWED_CAPABILITY_STATES:
        raise ValueError("invalid capability state")
    if request["capability_gain"] is not None and not isinstance(
        request["capability_gain"], bool
    ):
        raise ValueError("capability_gain must be boolean or null")
    if request["capability_state"] == "unknown" and request["capability_gain"] is not None:
        raise ValueError("unknown capability state requires null capability_gain")
    if request["capability_state"] != "unknown" and request["capability_gain"] is None:
        raise ValueError("known capability state requires boolean capability_gain")
    provenance = request["provenance"]
    if (
        not isinstance(provenance, list)
        or len(provenance) > 128
        or any(
            not isinstance(item, str) or not SAFE_PROVENANCE.fullmatch(item)
            for item in provenance
        )
        or len(provenance) != len(set(provenance))
    ):
        raise ValueError("invalid provenance")
    if not isinstance(request["observed_events"], list) or not request["observed_events"]:
        raise ValueError("observed_events must be a non-empty list")
    if len(request["observed_events"]) > 128:
        raise ValueError("too many observed events")
    if request["prefix_len"] != len(request["observed_events"]):
        raise ValueError("prefix_len must equal observed event count")
    seen_event_ids: set[str] = set()
    previous_event_time: dt.datetime | None = None
    for event in request["observed_events"]:
        if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
            raise ValueError("observed event does not match the allowlist")
        if not SAFE_ID.fullmatch(str(event["event_id"])):
            raise ValueError("invalid event_id")
        if str(event["event_id"]) in seen_event_ids:
            raise ValueError("duplicate event_id")
        seen_event_ids.add(str(event["event_id"]))
        event_time = _parse_datetime(event["timestamp"], "event timestamp")
        if event_time > cutoff:
            raise ValueError("event falls after the declared cutoff")
        if previous_event_time is not None and event_time < previous_event_time:
            raise ValueError("observed events must be chronological")
        previous_event_time = event_time
        if not isinstance(event["action"], str) or not SAFE_ACTION.fullmatch(event["action"]):
            raise ValueError("invalid event action")
        if event["outcome"] not in {"success", "failure", "unknown"}:
            raise ValueError("invalid event outcome")
        for field_name in ("actor", "target", "resource"):
            if not isinstance(event[field_name], str) or not SAFE_ENTITY.fullmatch(
                event[field_name]
            ):
                raise ValueError(f"event {field_name} is not an opaque entity ID")
        details = event["details"]
        if not isinstance(details, dict) or len(details) > 24:
            raise ValueError("invalid event details")
        for key, value in details.items():
            if key not in ALLOWED_DETAIL_KEYS:
                raise ValueError(f"event detail key is not allowlisted: {key}")
            if key in OPERATIONAL_DETAIL_KEYS and not isinstance(value, bool):
                raise ValueError(f"event detail {key} must be boolean")
            if key in REFERENCE_DETAIL_PATTERNS and (
                not isinstance(value, str)
                or not REFERENCE_DETAIL_PATTERNS[key].fullmatch(value)
            ):
                raise ValueError(f"event detail {key} must be an opaque ID")
            if key == "trust_condition" and value not in {"true", "false", "unknown"}:
                raise ValueError("invalid trust_condition")
            if key == "code_controlled" and value is not None and not isinstance(value, bool):
                raise ValueError("code_controlled must be boolean or null")
    if previous_event_time != cutoff:
        raise ValueError("cutoff must equal the final observed event timestamp")
    if not isinstance(request["candidate_path"], list):
        raise ValueError("candidate_path must be a list")
    if len(request["candidate_path"]) > 128 or any(
        not isinstance(item, str)
        or not (SAFE_ENTITY.fullmatch(item) or SAFE_ACTION.fullmatch(item))
        for item in request["candidate_path"]
    ):
        raise ValueError("invalid candidate_path")
    if not isinstance(request["unknown_preconditions"], list):
        raise ValueError("unknown_preconditions must be a list")
    if len(request["unknown_preconditions"]) > 64 or len(
        request["unknown_preconditions"]
    ) != len(set(request["unknown_preconditions"])):
        raise ValueError("invalid unknown_preconditions")
    if any(
        not isinstance(item, str) or not SAFE_UNKNOWN.fullmatch(item)
        for item in request["unknown_preconditions"]
    ):
        raise ValueError("unknown_preconditions must contain safe symbolic identifiers")
    features = request["feature_summary"]
    if not isinstance(features, dict) or set(features) != FEATURE_FIELDS:
        raise ValueError("feature_summary does not match the allowlist")
    if features["observed_event_count"] != len(request["observed_events"]):
        raise ValueError("feature/event count mismatch")
    if features["path_length"] != len(request["candidate_path"]):
        raise ValueError("feature/path length mismatch")
    if features["unknown_precondition_count"] != len(
        request["unknown_preconditions"]
    ):
        raise ValueError("feature/unknown-precondition count mismatch")
    for key in ("path_length", "cross_service_count", "observed_event_count", "unknown_precondition_count"):
        if not _is_integer(features[key]) or features[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if not isinstance(features["sensitive_action_observed"], bool):
        raise ValueError("sensitive_action_observed must be boolean")
    for key in ("prefix_fraction", "observed_edge_ratio", "witness_completeness"):
        value = features[key]
        if not _is_number(value) or not 0 <= float(value) <= 1:
            raise ValueError(f"{key} must be between zero and one")
    unsigned = dict(request)
    supplied_digest = str(unsigned.pop("witness_digest"))
    if supplied_digest != witness_digest(unsigned):
        raise ValueError("witness_digest mismatch")
    serialized = canonical_json(request).lower()
    for forbidden in (
        "ground_truth",
        "malicious_label",
        "attack_label",
        "expected_verdict",
    ):
        if forbidden in serialized:
            raise ValueError(f"worker input contains forbidden label field: {forbidden}")


def validate_response(
    response: dict[str, Any],
    request: dict[str, Any],
    *,
    allowed_event_ids: set[str] | None = None,
) -> None:
    missing = MODEL_OUTPUT_FIELDS - response.keys()
    extras = response.keys() - MODEL_OUTPUT_FIELDS
    if missing:
        raise ValueError(f"model output missing fields: {sorted(missing)}")
    if extras:
        raise ValueError(f"model output has unexpected fields: {sorted(extras)}")
    if response["schema_version"] != RESPONSE_SCHEMA_VERSION:
        raise ValueError("unsupported model-output schema")
    if response["episode_id"] != request["episode_id"]:
        raise ValueError("episode_id mismatch")
    if response["witness_digest"] != request["witness_digest"]:
        raise ValueError("witness_digest mismatch")
    if response["family"] != request["family"]:
        raise ValueError("family mismatch")
    verdict = response["verdict"]
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError("invalid verdict")
    probability = response["malicious_probability"]
    if not _is_number(probability) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("malicious_probability must be between 0 and 1")
    assessment = response["candidate_path_assessment"]
    if assessment not in ALLOWED_PATH_ASSESSMENTS:
        raise ValueError("invalid candidate_path_assessment")
    citations = response["cited_event_ids"]
    if (
        not isinstance(citations, list)
        or len(citations) > 32
        or any(not isinstance(item, str) or not SAFE_ID.fullmatch(item) for item in citations)
        or len(citations) != len(set(citations))
    ):
        raise ValueError("cited_event_ids must be a list")
    if allowed_event_ids is not None:
        extra_citations = set(map(str, citations)) - allowed_event_ids
        if extra_citations:
            raise ValueError(f"response cites unknown events: {sorted(extra_citations)}")
    signals = response["signals"]
    uncertainty = response["uncertainty_reasons"]
    if (
        not isinstance(signals, list)
        or len(signals) > 10
        or any(not isinstance(item, str) for item in signals)
        or len(signals) != len(set(signals))
        or not set(signals) <= ALLOWED_SIGNALS
    ):
        raise ValueError("invalid signals")
    if (
        not isinstance(uncertainty, list)
        or len(uncertainty) > 8
        or any(not isinstance(item, str) for item in uncertainty)
        or len(uncertainty) != len(set(uncertainty))
        or not set(uncertainty) <= ALLOWED_UNCERTAINTY_REASONS
    ):
        raise ValueError("invalid uncertainty_reasons")
    if not isinstance(response["summary"], str) or not 1 <= len(response["summary"]) <= 600:
        raise ValueError("summary must contain 1-600 characters")
    if verdict == "attack" and (assessment != "supports_attack" or not citations):
        raise ValueError("attack output requires supporting path and citation")
    if verdict == "attack" and float(probability) < 0.5:
        raise ValueError("attack output requires malicious_probability >= 0.5")
    if verdict == "benign" and float(probability) > 0.5:
        raise ValueError("benign output requires malicious_probability <= 0.5")
    if "none" in uncertainty and uncertainty != ["none"]:
        raise ValueError("uncertainty reason 'none' must appear alone")
    if verdict == "abstain" and (
        assessment != "insufficient" or not uncertainty or uncertainty == ["none"]
    ):
        raise ValueError("abstain output requires an uncertainty reason")


def validate_label_row(row: dict[str, Any]) -> None:
    if set(row) != {"episode_id", "malicious", "family", "split"}:
        raise ValueError("trusted label row has unexpected fields")
    if not isinstance(row["episode_id"], str) or not SAFE_EPISODE_ID.fullmatch(
        row["episode_id"]
    ):
        raise ValueError("invalid trusted-label episode_id")
    if not isinstance(row["malicious"], bool):
        raise ValueError("trusted-label malicious must be boolean")
    if row["family"] not in ALLOWED_FAMILIES:
        raise ValueError("invalid trusted-label family")
    if row["split"] not in {"prompt_dev", "blind"}:
        raise ValueError("invalid trusted-label split")


def validate_private_witness(witness: dict[str, Any]) -> None:
    from .witness import verify_witness_digest

    if set(witness) != WORKER_INPUT_FIELDS | {"split"}:
        raise ValueError("private witness has unexpected fields")
    if witness.get("split") not in {"prompt_dev", "blind"}:
        raise ValueError("private witness has invalid split")
    if not verify_witness_digest(witness):
        raise ValueError("private witness digest mismatch")
    request = make_request(
        case_id=str(witness["episode_id"]),
        cutoff_time=str(witness["cutoff"]),
        witness=witness,
    )
    validate_request(request)


def validate_run_record(record: dict[str, Any], request: dict[str, Any]) -> None:
    fields = {
        "schema_version",
        "run_id",
        "episode_id",
        "model_role",
        "model_id",
        "prompt_version",
        "prompt_sha256",
        "witness_digest",
        "started_at",
        "finished_at",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost",
        "validation",
        "output",
    }
    if set(record) != fields:
        raise ValueError("run record has unexpected fields")
    if record["schema_version"] != "1.0":
        raise ValueError("unsupported run-record schema")
    for key in ("run_id", "model_id"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"run-record {key} must be non-empty")
    if record["episode_id"] != request["episode_id"]:
        raise ValueError("run-record episode binding mismatch")
    if record["witness_digest"] != request["witness_digest"]:
        raise ValueError("run-record witness binding mismatch")
    if record["model_role"] not in {"local", "remote"}:
        raise ValueError("invalid run-record model role")
    if record["prompt_version"] not in {"local_v1", "remote_v1", "remote_v2"}:
        raise ValueError("invalid run-record prompt version")
    if not isinstance(record["prompt_sha256"], str) or not re.fullmatch(
        r"[a-f0-9]{64}", record["prompt_sha256"]
    ):
        raise ValueError("invalid run-record prompt hash")
    started = _parse_datetime(record["started_at"], "run start")
    finished = _parse_datetime(record["finished_at"], "run finish")
    if finished < started:
        raise ValueError("run record finishes before it starts")
    for key in ("latency_ms", "input_tokens", "output_tokens"):
        value = record[key]
        if value is not None and (not _is_integer(value) or value < 0):
            raise ValueError(f"run-record {key} must be a non-negative integer or null")
    cost = record["cost"]
    if not isinstance(cost, dict) or set(cost) != {
        "amount",
        "currency",
        "pricing_snapshot",
    }:
        raise ValueError("invalid run-record cost object")
    amount = cost["amount"]
    if amount is None:
        if cost["currency"] is not None or cost["pricing_snapshot"] is not None:
            raise ValueError("unknown cost requires null currency and pricing snapshot")
    else:
        if not _is_number(amount) or float(amount) < 0:
            raise ValueError("run-record cost must be finite and non-negative")
        if not isinstance(cost["currency"], str) or not re.fullmatch(
            r"[A-Z]{3}", cost["currency"]
        ):
            raise ValueError("invalid run-record currency")
        if not isinstance(cost["pricing_snapshot"], str) or not cost["pricing_snapshot"]:
            raise ValueError("missing run-record pricing snapshot")
    validation = record["validation"]
    if not isinstance(validation, dict) or set(validation) != {
        "schema_valid",
        "binding_valid",
        "citation_valid",
        "errors",
    }:
        raise ValueError("invalid run-record validation object")
    if any(not isinstance(validation[key], bool) for key in (
        "schema_valid",
        "binding_valid",
        "citation_valid",
    )):
        raise ValueError("run-record validation flags must be boolean")
    if not isinstance(validation["errors"], list) or any(
        not isinstance(item, str) for item in validation["errors"]
    ):
        raise ValueError("run-record validation errors must be strings")
    validate_response(
        record["output"],
        request,
        allowed_event_ids={str(event["event_id"]) for event in request["observed_events"]},
    )


def make_run_manifest(
    *,
    run_id: str,
    model_id: str,
    prompt: str,
    decoding: dict[str, Any],
    input_sha256: str,
    code_version: str,
    backend: str,
) -> dict[str, Any]:
    return {
        "schema_version": "flowgate-run-manifest/1.0",
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_sha256": prompt_hash(prompt),
        "decoding": decoding,
        "input_sha256": input_sha256,
        "code_version": code_version,
        "backend": backend,
    }
