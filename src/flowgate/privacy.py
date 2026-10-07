from __future__ import annotations

import re
from typing import Any

from .contracts import validate_request
from .io import canonical_json


FORBIDDEN_KEYS = {
    "account_id",
    "principal_arn",
    "resource_arn",
    "user_name",
    "username",
    "source_ip",
    "raw_event",
    "raw_cloudtrail",
    "policy_document",
    "ground_truth",
    "label",
    "malicious",
    "intent_label",
    "split",
    "split_name",
    "raw_arn",
    "raw_principal_name",
    "raw_resource_name",
    "credential",
    "secret_value",
    "expected_evidence",
    "expected_path",
    "expected_event_ids",
    "is_attack",
    "attack",
    "expected_verdict",
}

SENSITIVE_PATTERNS = {
    "aws_arn": re.compile(r"arn:(?:aws|aws-cn|aws-us-gov):", re.IGNORECASE),
    "aws_account_id": re.compile(r"(?<!\d)\d{12}(?!\d)"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "ipv4_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "email_address": re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
}


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_keys(nested))
    return keys


def validate_gpu_export(request: dict[str, Any]) -> None:
    """Fail closed when a GPU-bound request contains obvious private material."""

    validate_request(request)
    present = set(_walk_keys(request))
    bad_keys = sorted(present & FORBIDDEN_KEYS)
    if bad_keys:
        raise ValueError(f"GPU export contains forbidden keys: {bad_keys}")
    pattern_payload = dict(request)
    # A SHA-256 digest can contain a 12-digit run by chance; it is not an AWS
    # account identifier and must not create a false privacy failure.
    pattern_payload.pop("witness_digest", None)
    serialized = canonical_json(pattern_payload)
    matches = [name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(serialized)]
    if matches:
        raise ValueError(f"GPU export contains sensitive patterns: {sorted(matches)}")
