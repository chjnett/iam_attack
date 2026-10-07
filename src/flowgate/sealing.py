from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .batch import merge_predictions
from .contracts import prompt_hash, validate_request, validate_response, validate_run_record
from .freeze import validate_local_runtime_metadata, verify_freeze_manifest
from .io import canonical_json, file_sha256, read_json, read_jsonl, write_json
from .privacy import validate_gpu_export


OUTPUT_SEAL_SCHEMA_VERSION = "flowgate-output-seal/1.0"
REQUIRED_ARTIFACTS = {
    "blind_requests",
    "local_responses",
    "local_errors",
    "local_records",
    "local_manifest",
    "routing_decisions",
    "remote_responses",
    "remote_errors",
    "remote_records",
    "remote_manifest",
    "predictions",
}


def _id(row: dict[str, Any]) -> str:
    value = row.get("episode_id")
    if not isinstance(value, str) or not value:
        raise ValueError("sealed row is missing episode_id")
    return value


def _index(rows: Iterable[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _id(row)
        if identifier in result:
            raise ValueError(f"duplicate {kind} ID: {identifier}")
        result[identifier] = row
    return result


def _relative(root: Path, path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"sealed artifact must be inside the project: {resolved}") from exc


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    role: str,
    freeze: dict[str, Any],
    freeze_hash: str,
    request_hash: str,
    episode_count: int,
    project_root: Path,
) -> None:
    model = freeze["models"][role]
    prompt_file = f"prompts/{role}_v1.txt"
    if manifest.get("schema_version") != "flowgate-run-manifest/1.0":
        raise ValueError(f"{role} run manifest has an unsupported schema")
    from .bundle import SAFE_WORKER_FILES

    code_digest = hashlib.sha256()
    for relative in sorted(
        item for item in SAFE_WORKER_FILES if item.startswith("src/") and item.endswith(".py")
    ):
        code_digest.update(relative.encode("utf-8"))
        code_digest.update((project_root / relative).read_bytes())
    expected = {
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "input_sha256": request_hash,
        "prompt_sha256": prompt_hash(
            (project_root / prompt_file).read_text(encoding="utf-8").strip()
        ),
        "phase": "blind",
        "protocol_freeze_sha256": freeze_hash,
        "request_count": episode_count,
        "valid_response_count": episode_count,
        "error_count": 0,
        "run_record_count": episode_count,
        "code_version": "sha256:" + code_digest.hexdigest(),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{role} run manifest mismatch: {mismatches}")
    if manifest.get("decoding") != freeze["decoding"]:
        raise ValueError(f"{role} run decoding differs from protocol freeze")
    if role == "local":
        if manifest.get("quantization") != model["quantization"]:
            raise ValueError("local quantization differs from protocol freeze")
        if manifest.get("serving_config") != model["serving"]:
            raise ValueError("local serving configuration differs from protocol freeze")
        validate_local_runtime_metadata(
            manifest.get("runtime_metadata"),
            expected_vllm_version=model["runtime"]["vllm"],
        )
    if role == "remote":
        expected_pricing = {
            "input_per_million_tokens": freeze["pricing"]["input_per_million_tokens"],
            "output_per_million_tokens": freeze["pricing"]["output_per_million_tokens"],
            "currency": freeze["pricing"]["currency"],
            "pricing_snapshot": freeze["pricing"]["snapshot"],
        }
        if manifest.get("pricing") != expected_pricing:
            raise ValueError("remote run pricing differs from protocol freeze")


def create_output_seal(
    *,
    project_root: str | Path,
    protocol_freeze_path: str | Path,
    output_path: str | Path,
    artifact_paths: dict[str, str | Path],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    freeze = verify_freeze_manifest(protocol_freeze_path, project_root=root)
    if set(artifact_paths) != REQUIRED_ARTIFACTS:
        raise ValueError("output seal requires the exact artifact set")
    resolved = {name: root / _relative(root, path) for name, path in artifact_paths.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot seal missing artifacts: {missing}")

    blind_ids = set(freeze["blind_episode_ids"])
    requests = list(read_jsonl(resolved["blind_requests"]))
    request_by_id = _index(requests, "blind request")
    if set(request_by_id) != blind_ids:
        raise ValueError("blind request IDs differ from protocol freeze")
    canonical_requests = _index(
        read_jsonl(root / "data/generated/gpu/requests.jsonl"), "canonical request"
    )
    for identifier, request in request_by_id.items():
        validate_request(request)
        validate_gpu_export(request)
        if canonical_json(request) != canonical_json(canonical_requests[identifier]):
            raise ValueError(f"blind request differs from frozen request: {identifier}")

    local = _index(read_jsonl(resolved["local_responses"]), "local response")
    remote = _index(read_jsonl(resolved["remote_responses"]), "remote response")
    local_errors = list(read_jsonl(resolved["local_errors"]))
    remote_errors = list(read_jsonl(resolved["remote_errors"]))
    if local_errors or remote_errors:
        raise ValueError(
            "Pareto/complementarity output seal requires zero first-attempt errors; "
            "record an INCONCLUSIVE run instead"
        )
    if set(local) != blind_ids or set(remote) != blind_ids:
        raise ValueError("local and remote-all responses must cover every blind episode")
    for identifier in sorted(blind_ids):
        request = request_by_id[identifier]
        allowed = {str(event["event_id"]) for event in request["observed_events"]}
        validate_response(local[identifier], request, allowed_event_ids=allowed)
        validate_response(remote[identifier], request, allowed_event_ids=allowed)

    for role, response_by_id in (("local", local), ("remote", remote)):
        records = _index(read_jsonl(resolved[f"{role}_records"]), f"{role} record")
        if set(records) != blind_ids:
            raise ValueError(f"{role} run records must cover every blind episode")
        for identifier, record in records.items():
            if record.get("model_role") != role:
                raise ValueError(f"{role} run record has the wrong role")
            validate_run_record(record, request_by_id[identifier])
            if canonical_json(record["output"]) != canonical_json(response_by_id[identifier]):
                raise ValueError(f"{role} record/output mismatch: {identifier}")

    decisions = list(read_jsonl(resolved["routing_decisions"]))
    decision_by_id = _index(decisions, "routing decision")
    if set(decision_by_id) != blind_ids:
        raise ValueError("routing decisions must cover every blind episode")
    selected = 0
    for row in decisions:
        if row.get("policy") != freeze["routing"]["policy"]:
            raise ValueError("routing decision policy differs from protocol freeze")
        value = row.get("route_to_remote")
        if not isinstance(value, bool):
            raise ValueError("routing decision must contain a JSON boolean")
        selected += int(value)
    if selected != freeze["routing"]["call_budget"]:
        raise ValueError("routing decisions do not use the exact frozen call budget")

    predictions = list(read_jsonl(resolved["predictions"]))
    prediction_by_id = _index(predictions, "prediction")
    if set(prediction_by_id) != blind_ids:
        raise ValueError("predictions must cover every blind episode")
    derived = merge_predictions(requests, local.values(), decisions, remote.values())
    if {canonical_json(row) for row in predictions} != {canonical_json(row) for row in derived}:
        raise ValueError("predictions are not the deterministic merge of sealed inputs")

    freeze_hash = file_sha256(protocol_freeze_path)
    request_hash = file_sha256(resolved["blind_requests"])
    for role in ("local", "remote"):
        _validate_manifest(
            read_json(resolved[f"{role}_manifest"]),
            role=role,
            freeze=freeze,
            freeze_hash=freeze_hash,
            request_hash=request_hash,
            episode_count=len(blind_ids),
            project_root=root,
        )

    artifacts = {
        name: {
            "path": _relative(root, path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(resolved.items())
    }
    seal = {
        "schema_version": OUTPUT_SEAL_SCHEMA_VERSION,
        "status": "sealed_before_label_unseal",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": freeze["protocol_id"],
        "protocol_freeze_path": _relative(root, protocol_freeze_path),
        "protocol_freeze_sha256": freeze_hash,
        "blind_episode_count": len(blind_ids),
        "blind_episode_ids": sorted(blind_ids),
        "remote_all_counterfactual": True,
        "artifacts": artifacts,
    }
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"output seal already exists: {target}")
    write_json(target, seal)
    return seal


def verify_output_seal(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    seal = read_json(path)
    if not isinstance(seal, dict) or seal.get("schema_version") != OUTPUT_SEAL_SCHEMA_VERSION:
        raise ValueError("unsupported output seal schema")
    if seal.get("status") != "sealed_before_label_unseal":
        raise ValueError("outputs were not sealed before label unseal")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("output seal does not contain the exact artifact set")
    freeze_path = root / str(seal.get("protocol_freeze_path", ""))
    freeze = verify_freeze_manifest(freeze_path, project_root=root)
    if file_sha256(freeze_path) != seal.get("protocol_freeze_sha256"):
        raise ValueError("protocol freeze hash differs from output seal")
    if seal.get("protocol_id") != freeze.get("protocol_id"):
        raise ValueError("output seal protocol ID mismatch")
    if seal.get("blind_episode_ids") != freeze.get("blind_episode_ids"):
        raise ValueError("output seal blind ID commitment mismatch")
    mismatches: list[str] = []
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            mismatches.append(name)
            continue
        artifact = root / str(metadata.get("path", ""))
        if (
            not artifact.is_file()
            or file_sha256(artifact) != metadata.get("sha256")
            or artifact.stat().st_size != metadata.get("size_bytes")
        ):
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"sealed outputs changed after commitment: {mismatches}")
    return seal
