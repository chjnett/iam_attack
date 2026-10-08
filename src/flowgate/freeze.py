from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path
from typing import Any

from .io import file_sha256, read_json, read_jsonl, write_json


FREEZE_SCHEMA_VERSION = "flowgate-protocol-freeze/1.0"
PROTOCOL_ID = "flowgate-pilot-24-v1"
VLLM_RUNTIME_VERSION = "0.31.0"
VLLM_RUNTIME_ENVIRONMENT = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
FROZEN_EXPLICIT_FILES = (
    "configs/pilot.json",
    "scripts/capture_runtime.py",
    "prompts/local_v1.txt",
    "prompts/remote_v1.txt",
    "prompts/remote_v2.txt",
    "schemas/witness.schema.json",
    "schemas/worker-input.schema.json",
    "schemas/model-output.schema.json",
    "schemas/run-record.schema.json",
    "schemas/trusted-labels.schema.json",
    "data/generated/manifest.json",
    "data/generated/private/episodes.jsonl",
    "data/generated/private/witnesses.jsonl",
    "data/generated/private/prompt_dev_labels.jsonl",
    "data/generated/private/blind_labels.jsonl",
    "data/generated/gpu/requests.jsonl",
)


def _frozen_files(project_root: Path) -> list[str]:
    code = [
        str(path.relative_to(project_root))
        for path in sorted((project_root / "src" / "flowgate").glob("*.py"))
    ]
    return [*FROZEN_EXPLICIT_FILES, *code]


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def validate_local_runtime_metadata(
    metadata: Any,
    *,
    expected_vllm_version: str = VLLM_RUNTIME_VERSION,
    expected_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("local blind run is missing runtime metadata")
    packages = metadata.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("local runtime metadata is missing packages")
    if packages.get("vllm") != expected_vllm_version:
        raise ValueError(
            "local vLLM runtime differs from protocol freeze: "
            f"expected {expected_vllm_version}, got {packages.get('vllm')!r}"
        )
    required_environment = (
        VLLM_RUNTIME_ENVIRONMENT
        if expected_environment is None
        else expected_environment
    )
    environment = metadata.get("environment")
    if not isinstance(environment, dict) or any(
        environment.get(name) != value
        for name, value in required_environment.items()
    ):
        raise ValueError(
            "local runtime environment differs from protocol freeze: "
            f"expected {required_environment}, got {environment!r}"
        )
    return metadata


def _validate_pricing(
    *,
    input_price_per_million: Any,
    output_price_per_million: Any,
    currency: Any,
    pricing_snapshot: Any,
) -> dict[str, Any]:
    input_price = float(input_price_per_million)
    output_price = float(output_price_per_million)
    if not math.isfinite(input_price) or input_price < 0:
        raise ValueError("input price must be finite and non-negative")
    if not math.isfinite(output_price) or output_price < 0:
        raise ValueError("output price must be finite and non-negative")
    normalized_currency = _required_text(currency, "currency").upper()
    if not re.fullmatch(r"[A-Z]{3}", normalized_currency):
        raise ValueError("currency must be a three-letter uppercase code")
    return {
        "currency": normalized_currency,
        "input_per_million_tokens": input_price,
        "output_per_million_tokens": output_price,
        "snapshot": _required_text(pricing_snapshot, "pricing snapshot"),
    }


def _validated_split_ids(root: Path) -> tuple[list[str], list[str]]:
    prompt_rows = list(
        read_jsonl(root / "data/generated/private/prompt_dev_labels.jsonl")
    )
    blind_rows = list(read_jsonl(root / "data/generated/private/blind_labels.jsonl"))
    if len(prompt_rows) != 6 or any(row.get("split") != "prompt_dev" for row in prompt_rows):
        raise ValueError("prompt-development labels must contain exactly six prompt_dev rows")
    if len(blind_rows) != 18 or any(row.get("split") != "blind" for row in blind_rows):
        raise ValueError("blind labels must contain exactly eighteen blind rows")
    prompt_ids = [str(row.get("episode_id")) for row in prompt_rows]
    blind_ids = [str(row.get("episode_id")) for row in blind_rows]
    if len(set(prompt_ids + blind_ids)) != 24 or set(prompt_ids) & set(blind_ids):
        raise ValueError("label splits must contain 24 unique disjoint episode IDs")
    request_ids = {
        str(row.get("episode_id"))
        for row in read_jsonl(root / "data/generated/gpu/requests.jsonl")
    }
    if request_ids != set(prompt_ids + blind_ids):
        raise ValueError("frozen request IDs do not match the two label splits")
    return sorted(prompt_ids), sorted(blind_ids)


def create_freeze_manifest(
    *,
    project_root: str | Path,
    output_path: str | Path,
    local_model_id: str,
    local_model_revision: str,
    remote_model_id: str,
    remote_model_revision: str,
    remote_provider: str,
    provider_retention: str,
    pricing_snapshot: str,
    input_price_per_million: float,
    output_price_per_million: float,
    currency: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    files = _frozen_files(root)
    missing = [relative for relative in files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze; missing files: {missing}")
    prompt_ids, blind_ids = _validated_split_ids(root)
    pricing = _validate_pricing(
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        currency=currency,
        pricing_snapshot=pricing_snapshot,
    )
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_blind_run",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "models": {
            "local": {
                "model_id": _required_text(local_model_id, "local model ID"),
                "model_revision": _required_text(local_model_revision, "local model revision"),
                "role": "local",
                "quantization": "awq",
                "runtime": {
                    "vllm": VLLM_RUNTIME_VERSION,
                    "environment": VLLM_RUNTIME_ENVIRONMENT,
                },
                "serving": {
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.85,
                },
            },
            "remote": {
                "model_id": _required_text(remote_model_id, "remote model ID"),
                "model_revision": _required_text(remote_model_revision, "remote model revision"),
                "provider": _required_text(remote_provider, "remote provider"),
                "retention_setting": _required_text(
                    provider_retention, "provider retention setting"
                ),
                "role": "remote",
            },
        },
        "decoding": {"temperature": 0, "top_p": 1, "max_output_tokens": 700},
        "routing": {
            "policy": "flowgate-v0",
            "budget_fraction": 0.25,
            "seed": 7,
            "evaluation_split": "blind",
            "call_budget": 4,
        },
        "privacy": {
            "sanitizer_version": "privacy-v1-closed-details",
            "worker_schema": "schemas/worker-input.schema.json",
        },
        "pricing": pricing,
        "prompt_dev_episode_ids": prompt_ids,
        "blind_episode_ids": blind_ids,
        "file_sha256": {relative: file_sha256(root / relative) for relative in files},
        "blind_label_commitment": file_sha256(
            root / "data/generated/private/blind_labels.jsonl"
        ),
        "note": (
            "This manifest is an audit commitment, not cryptographic access control. "
            "Do not inspect blind_labels.jsonl until blind outputs are sealed."
        ),
    }
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"freeze manifest already exists: {target}")
    write_json(target, manifest)
    return manifest


def verify_freeze_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("freeze manifest must be an object")
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("unsupported freeze manifest schema")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol ID")
    if manifest.get("status") != "frozen_before_blind_run":
        raise ValueError("freeze manifest is not in frozen_before_blind_run state")
    expected_files = _frozen_files(root)
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(expected_files):
        raise ValueError("freeze manifest does not contain the exact required file set")
    mismatches = [
        relative
        for relative in expected_files
        if not (root / relative).is_file()
        or file_sha256(root / relative) != hashes.get(relative)
    ]
    if mismatches:
        raise ValueError(f"frozen files changed after commitment: {mismatches}")
    prompt_ids, blind_ids = _validated_split_ids(root)
    if manifest.get("prompt_dev_episode_ids") != prompt_ids:
        raise ValueError("prompt-development ID commitment mismatch")
    if manifest.get("blind_episode_ids") != blind_ids:
        raise ValueError("blind ID commitment mismatch")
    blind_hash = file_sha256(root / "data/generated/private/blind_labels.jsonl")
    if manifest.get("blind_label_commitment") != blind_hash:
        raise ValueError("blind label commitment mismatch")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("freeze manifest is missing model commitments")
    for role in ("local", "remote"):
        model = models.get(role)
        if not isinstance(model, dict) or model.get("role") != role:
            raise ValueError(f"freeze manifest is missing the {role} model commitment")
        _required_text(model.get("model_id"), f"{role} model ID")
        _required_text(model.get("model_revision"), f"{role} model revision")
    if models["local"].get("quantization") != "awq" or models["local"].get(
        "serving"
    ) != {"max_model_len": 8192, "gpu_memory_utilization": 0.85}:
        raise ValueError("local quantization/serving commitment differs from protocol")
    if models["local"].get("runtime") != {
        "vllm": VLLM_RUNTIME_VERSION,
        "environment": VLLM_RUNTIME_ENVIRONMENT,
    }:
        raise ValueError("local vLLM runtime commitment differs from protocol")
    remote = models["remote"]
    _required_text(remote.get("provider"), "remote provider")
    _required_text(remote.get("retention_setting"), "remote retention setting")
    expected_routing = {
        "policy": "flowgate-v0",
        "budget_fraction": 0.25,
        "seed": 7,
        "evaluation_split": "blind",
        "call_budget": 4,
    }
    if manifest.get("routing") != expected_routing:
        raise ValueError("routing commitment differs from the frozen protocol")
    if manifest.get("decoding") != {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 700,
    }:
        raise ValueError("decoding commitment differs from the frozen protocol")
    pricing = manifest.get("pricing")
    if not isinstance(pricing, dict):
        raise ValueError("freeze manifest is missing pricing")
    validated_pricing = _validate_pricing(
        input_price_per_million=pricing.get("input_per_million_tokens"),
        output_price_per_million=pricing.get("output_per_million_tokens"),
        currency=pricing.get("currency"),
        pricing_snapshot=pricing.get("snapshot"),
    )
    if pricing != validated_pricing:
        raise ValueError("pricing commitment is not canonical")
    return manifest
