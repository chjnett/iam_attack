from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .batch import merge_predictions, read_prompt, run_backend, selected_requests
from .contracts import (
    make_request,
    make_run_manifest,
    prompt_hash,
    validate_label_row,
    validate_private_witness,
    validate_request,
)
from .inference import MockBackend, OpenAICompatibleBackend
from .io import canonical_json, file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .privacy import validate_gpu_export
from .routing import SUPPORTED_POLICIES, select_remote_calls


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted
    raise TypeError(f"expected dict-like object, got {type(value).__name__}")


def _case_id(value: dict[str, Any]) -> str:
    result = value.get("episode_id", value.get("case_id", value.get("id")))
    if result is None:
        raise ValueError("fixture has no episode_id/case_id/id")
    return str(result)


def _code_fingerprint(project_root: Path) -> str:
    from .bundle import SAFE_WORKER_FILES

    digest = hashlib.sha256()
    for relative in sorted(
        item for item in SAFE_WORKER_FILES if item.startswith("src/") and item.endswith(".py")
    ):
        path = project_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _ensure_new(paths: Sequence[str | Path], *, force: bool = False) -> None:
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite first-attempt artifacts: {existing}")


def _validate_blind_requests(
    rows: list[dict[str, Any]],
    *,
    freeze: dict[str, Any],
    project_root: Path,
    require_canonical_bytes: bool,
) -> None:
    expected_ids = set(map(str, freeze.get("blind_episode_ids", [])))
    actual_ids = {_case_id(row) for row in rows}
    if len(rows) != len(actual_ids) or actual_ids != expected_ids:
        raise ValueError("blind request IDs differ from the protocol commitment")
    for row in rows:
        validate_request(row)
        validate_gpu_export(row)
    if require_canonical_bytes:
        canonical = {
            _case_id(row): row
            for row in read_jsonl(project_root / "data/generated/gpu/requests.jsonl")
        }
        for row in rows:
            if canonical_json(row) != canonical_json(canonical[_case_id(row)]):
                raise ValueError(f"blind request differs from frozen input: {_case_id(row)}")


def _worker_freeze(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("freeze manifest must be an object")
    if manifest.get("schema_version") != "flowgate-protocol-freeze/1.0":
        raise ValueError("unsupported freeze manifest schema")
    if manifest.get("protocol_id") != "flowgate-pilot-24-v1":
        raise ValueError("unexpected protocol ID")
    if manifest.get("status") != "frozen_before_blind_run":
        raise ValueError("protocol is not frozen for a blind run")
    if len(manifest.get("blind_episode_ids", [])) != 18:
        raise ValueError("freeze manifest does not commit to 18 blind episodes")
    if manifest.get("decoding") != {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 700,
    }:
        raise ValueError("unexpected blind decoding commitment")
    from .freeze import VLLM_RUNTIME_ENVIRONMENT, VLLM_RUNTIME_VERSION

    local_model = manifest.get("models", {}).get("local", {})
    if local_model.get("runtime") != {
        "vllm": VLLM_RUNTIME_VERSION,
        "environment": VLLM_RUNTIME_ENVIRONMENT,
    }:
        raise ValueError("unexpected local vLLM runtime commitment")
    return manifest


def generate_corpus(out_dir: Path) -> dict[str, Any]:
    from .fixtures import build_fixture_corpus
    from .witness import build_witness

    episodes_raw, labels_raw = build_fixture_corpus()
    episodes = [_dict(item) for item in episodes_raw]
    labels = [_dict(item) for item in labels_raw]
    for label in labels:
        validate_label_row(label)
    witnesses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    from .witness import verify_witness_digest

    for raw_episode, episode in zip(episodes_raw, episodes, strict=True):
        witness = _dict(build_witness(raw_episode))
        validate_private_witness(witness)
        if not verify_witness_digest(witness):
            raise ValueError(f"canonical witness digest failed for {_case_id(episode)}")
        witnesses.append(witness)
        case_id = _case_id(episode)
        cutoff = witness.get(
            "cutoff",
            witness.get(
                "event_prefix_cutoff",
                episode.get("cutoff", episode.get("cutoff_time", "1970-01-01T00:00:00Z")),
            ),
        )
        request = make_request(case_id=case_id, cutoff_time=str(cutoff), witness=witness)
        validate_request(request)
        validate_gpu_export(request)
        requests.append(request)

    label_ids = {_case_id(row) for row in labels}
    request_ids = {str(row["episode_id"]) for row in requests}
    if len(episodes) != 24 or len(labels) != 24 or len(requests) != 24:
        raise ValueError("pilot corpus must contain exactly 24 episodes")
    if label_ids != request_ids:
        raise ValueError("label/request case IDs differ")

    private_dir = out_dir / "private"
    gpu_dir = out_dir / "gpu"
    write_jsonl(private_dir / "episodes.jsonl", episodes)
    write_jsonl(private_dir / "witnesses.jsonl", witnesses)
    write_jsonl(
        private_dir / "prompt_dev_labels.jsonl",
        [row for row in labels if row.get("split") == "prompt_dev"],
    )
    write_jsonl(
        private_dir / "blind_labels.jsonl",
        [row for row in labels if row.get("split") == "blind"],
    )
    write_jsonl(gpu_dir / "requests.jsonl", requests)

    families: dict[str, int] = {}
    splits: dict[str, int] = {}
    attacks = 0
    for label in labels:
        family = str(label.get("family", "unknown"))
        split = str(label.get("split", "unknown"))
        families[family] = families.get(family, 0) + 1
        splits[split] = splits.get(split, 0) + 1
        attacks += int(bool(label.get("malicious", label.get("label", False))))
    from .engineering import approximate_tokens, audit_fixture_negative_controls

    token_counts = sorted(approximate_tokens(request) for request in requests)
    median_tokens = (
        (token_counts[11] + token_counts[12]) / 2 if len(token_counts) == 24 else 0
    )
    manifest = {
        "schema_version": "flowgate-corpus-manifest/1.0",
        "purpose": "synthetic feasibility pilot; not a benchmark claim",
        "episodes": len(episodes),
        "attacks": attacks,
        "benign": len(labels) - attacks,
        "families": families,
        "splits": splits,
        "gpu_export_is_label_blind": True,
        "input_contract_valid_count": len(requests),
        "witness_success_count": len(witnesses),
        "invalid_false_pass_count": audit_fixture_negative_controls(),
        "median_witness_tokens": median_tokens,
        "token_count_method": "ceil(UTF-8 canonical JSON bytes / 4)",
        "paths": {
            "private_episodes": "private/episodes.jsonl",
            "private_witnesses": "private/witnesses.jsonl",
            "prompt_dev_labels": "private/prompt_dev_labels.jsonl",
            "blind_labels": "private/blind_labels.jsonl",
            "gpu_requests": "gpu/requests.jsonl",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def cmd_generate(args: argparse.Namespace) -> int:
    _ensure_new([Path(args.out_dir) / "manifest.json"], force=args.force)
    manifest = generate_corpus(Path(args.out_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_bundle_gpu(args: argparse.Namespace) -> int:
    from .bundle import create_gpu_bundle

    project_root = Path(__file__).resolve().parents[2]
    include_ids = None
    freeze_path = None
    if args.split == "blind":
        if not args.freeze_manifest:
            raise ValueError("blind GPU export requires --freeze-manifest")
        from .freeze import verify_freeze_manifest

        freeze = verify_freeze_manifest(args.freeze_manifest, project_root=project_root)
        include_ids = set(freeze["blind_episode_ids"])
        source_rows = list(read_jsonl(args.requests))
        expected_hash = freeze["file_sha256"]["data/generated/gpu/requests.jsonl"]
        if file_sha256(args.requests) != expected_hash:
            raise ValueError("blind bundle requests are not the frozen canonical file")
        _validate_blind_requests(
            [row for row in source_rows if _case_id(row) in include_ids],
            freeze=freeze,
            project_root=project_root,
            require_canonical_bytes=True,
        )
        freeze_path = args.freeze_manifest
    elif args.split != "all":
        label_rows = [row for path in args.labels for row in read_jsonl(path)]
        include_ids = {
            _case_id(row) for row in label_rows if row.get("split") == args.split
        }
        if not include_ids:
            raise ValueError(f"no label rows found for split {args.split}")
    manifest = create_gpu_bundle(
        project_root=project_root,
        requests_path=args.requests,
        output_path=args.out,
        include_ids=include_ids,
        protocol_freeze_path=freeze_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    from .freeze import create_freeze_manifest

    project_root = Path(__file__).resolve().parents[2]
    manifest = create_freeze_manifest(
        project_root=project_root,
        output_path=args.out,
        local_model_id=args.local_model_id,
        remote_model_id=args.remote_model_id,
        remote_provider=args.remote_provider,
        provider_retention=args.provider_retention,
        pricing_snapshot=args.pricing_snapshot,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        currency=args.currency,
        local_model_revision=args.local_model_revision,
        remote_model_revision=args.remote_model_revision,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _backend_from_args(args: argparse.Namespace):
    if args.backend == "mock":
        return MockBackend(model_id=args.model_id or "mock-smoke-only", remote=args.remote)
    if not args.base_url or not args.model_id or not args.api_key_env:
        raise ValueError("openai backend requires --base-url, --model-id, and --api-key-env")
    parsed = urlparse(args.base_url)
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if args.remote and parsed.scheme != "https":
        raise ValueError("remote model endpoints must use HTTPS")
    if parsed.scheme == "http" and not loopback:
        raise ValueError("plaintext HTTP is permitted only for a loopback local model")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("model endpoint must use HTTP(S)")
    return OpenAICompatibleBackend(
        base_url=args.base_url,
        model_id=args.model_id,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        temperature=0,
        top_p=1,
        max_output_tokens=700,
    )


def cmd_run_batch(args: argparse.Namespace) -> int:
    requests = list(read_jsonl(args.requests))
    prompt = read_prompt(args.prompt)
    backend = _backend_from_args(args)
    output_paths = [
        Path(args.out),
        Path(args.errors or (str(args.out) + ".errors.jsonl")),
        Path(args.records or (str(args.out) + ".records.jsonl")),
        Path(str(args.out) + ".manifest.json"),
    ]
    _ensure_new(output_paths, force=args.force)
    project_root = Path(__file__).resolve().parents[2]
    model_role = "remote" if args.remote else "local"
    prompt_version = Path(args.prompt).stem
    allowed_prompt_versions = (
        {"remote_v1", "remote_v2"} if args.remote else {"local_v1"}
    )
    if prompt_version not in allowed_prompt_versions:
        raise ValueError(
            f"unsupported {model_role} prompt version: {prompt_version}"
        )
    canonical_prompt = project_root / "prompts" / f"{prompt_version}.txt"
    if not canonical_prompt.is_file() or prompt_hash(prompt) != prompt_hash(
        read_prompt(canonical_prompt)
    ):
        raise ValueError("run prompt must match the checked-in role-specific prompt")
    protocol_freeze_hash = None
    runtime_metadata = read_json(args.runtime_metadata) if args.runtime_metadata else None
    if args.phase == "blind":
        if not args.freeze_manifest:
            raise ValueError("blind run requires --freeze-manifest")
        freeze = _worker_freeze(args.freeze_manifest)
        _validate_blind_requests(
            requests,
            freeze=freeze,
            project_root=project_root,
            require_canonical_bytes=(
                project_root / "data/generated/gpu/requests.jsonl"
            ).is_file(),
        )
        model_commitment = freeze["models"][model_role]
        if args.model_id != model_commitment.get("model_id"):
            raise ValueError("model ID differs from protocol freeze")
        if args.model_revision != model_commitment.get("model_revision"):
            raise ValueError("model revision differs from protocol freeze")
        if model_role == "local":
            if str(args.quantization or "").lower() != model_commitment.get("quantization"):
                raise ValueError("local quantization differs from protocol freeze")
            serving = {
                "max_model_len": args.server_max_model_len,
                "gpu_memory_utilization": args.server_gpu_memory_utilization,
            }
            if serving != model_commitment.get("serving"):
                raise ValueError("local serving configuration differs from protocol freeze")
            if not args.runtime_metadata:
                raise ValueError("blind local run requires --runtime-metadata")
            from .freeze import validate_local_runtime_metadata

            runtime_metadata = validate_local_runtime_metadata(
                runtime_metadata,
                expected_vllm_version=model_commitment["runtime"]["vllm"],
                expected_environment=model_commitment["runtime"]["environment"],
            )
        expected_prompt_file_hash = freeze["file_sha256"][
            f"prompts/{prompt_version}.txt"
        ]
        if file_sha256(canonical_prompt) != expected_prompt_file_hash:
            raise ValueError("prompt differs from protocol freeze")
        protocol_freeze_hash = freeze.get(
            "source_protocol_freeze_sha256", file_sha256(args.freeze_manifest)
        )
    pricing = None
    if args.remote and args.backend != "mock":
        required_pricing = (
            args.input_price_per_million,
            args.output_price_per_million,
            args.currency,
            args.pricing_snapshot,
        )
        if any(value is None for value in required_pricing):
            raise ValueError("remote run requires frozen pricing arguments")
        from .freeze import _validate_pricing

        checked_pricing = _validate_pricing(
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            currency=args.currency,
            pricing_snapshot=args.pricing_snapshot,
        )
        pricing = {
            "input_per_million_tokens": checked_pricing["input_per_million_tokens"],
            "output_per_million_tokens": checked_pricing["output_per_million_tokens"],
            "currency": checked_pricing["currency"],
            "pricing_snapshot": checked_pricing["snapshot"],
        }
        if args.phase == "blind":
            expected_pricing = freeze["pricing"]
            actual_pricing = {
                "currency": str(args.currency).upper(),
                "input_per_million_tokens": args.input_price_per_million,
                "output_per_million_tokens": args.output_price_per_million,
                "snapshot": args.pricing_snapshot,
            }
            if actual_pricing != expected_pricing:
                raise ValueError("remote pricing differs from protocol freeze")
    responses, errors, records = run_backend(
        requests,
        backend=backend,
        prompt=prompt,
        run_id=args.run_id,
        model_role=model_role,
        prompt_version=prompt_version,
        pricing=pricing,
    )
    write_jsonl(args.out, responses)
    error_path = args.errors or (str(args.out) + ".errors.jsonl")
    write_jsonl(error_path, errors)
    record_path = args.records or (str(args.out) + ".records.jsonl")
    write_jsonl(record_path, records)
    manifest = make_run_manifest(
        run_id=args.run_id,
        model_id=backend.model_id,
        prompt=prompt,
        decoding={"temperature": 0, "top_p": 1, "max_output_tokens": 700},
        input_sha256=file_sha256(args.requests),
        code_version=_code_fingerprint(project_root),
        backend=args.backend,
    )
    manifest.update(
        {
            "request_count": len(requests),
            "valid_response_count": len(responses),
            "error_count": len(errors),
            "run_record_count": len(records),
            "smoke_only": args.backend == "mock",
            "endpoint_origin": (
                f"{urlparse(args.base_url).scheme}://{urlparse(args.base_url).netloc}"
                if args.base_url
                else None
            ),
            "pricing": pricing,
            "model_revision": args.model_revision,
            "quantization": args.quantization,
            "runtime_metadata": runtime_metadata,
            "serving_config": (
                {
                    "max_model_len": args.server_max_model_len,
                    "gpu_memory_utilization": args.server_gpu_memory_utilization,
                }
                if model_role == "local"
                else None
            ),
            "phase": args.phase,
            "protocol_freeze_sha256": protocol_freeze_hash,
        }
    )
    write_json(str(args.out) + ".manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def cmd_export_split(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    source = list(read_jsonl(args.requests))
    if args.split == "blind":
        if not args.freeze_manifest:
            raise ValueError("blind export requires --freeze-manifest")
        from .freeze import verify_freeze_manifest

        freeze = verify_freeze_manifest(args.freeze_manifest, project_root=project_root)
        expected_source = freeze["file_sha256"]["data/generated/gpu/requests.jsonl"]
        if file_sha256(args.requests) != expected_source:
            raise ValueError("split source is not the frozen canonical request file")
        include_ids = set(freeze["blind_episode_ids"])
    else:
        label_rows = [row for path in args.labels for row in read_jsonl(path)]
        include_ids = {
            _case_id(row) for row in label_rows if row.get("split") == args.split
        }
    rows = [row for row in source if _case_id(row) in include_ids]
    expected_count = 18 if args.split == "blind" else 6
    if len(rows) != expected_count or len({_case_id(row) for row in rows}) != expected_count:
        raise ValueError(f"{args.split} export must contain exactly {expected_count} episodes")
    for row in rows:
        validate_request(row)
        validate_gpu_export(row)
    manifest_path = args.manifest_out or (str(args.out) + ".manifest.json")
    _ensure_new([args.out, manifest_path], force=args.force)
    write_jsonl(args.out, rows)
    manifest = {
        "schema_version": "flowgate-split-export/1.0",
        "split": args.split,
        "episode_count": len(rows),
        "episode_ids": sorted(_case_id(row) for row in rows),
        "source_sha256": file_sha256(args.requests),
        "output_sha256": file_sha256(args.out),
        "protocol_freeze_sha256": (
            file_sha256(args.freeze_manifest) if args.freeze_manifest else None
        ),
        "label_blind": True,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_seal_outputs(args: argparse.Namespace) -> int:
    from .sealing import create_output_seal

    project_root = Path(__file__).resolve().parents[2]
    artifacts = {
        "blind_requests": args.requests,
        "local_responses": args.local_responses,
        "local_errors": args.local_errors,
        "local_records": args.local_records,
        "local_manifest": args.local_manifest,
        "routing_decisions": args.decisions,
        "remote_responses": args.remote_responses,
        "remote_errors": args.remote_errors,
        "remote_records": args.remote_records,
        "remote_manifest": args.remote_manifest,
        "predictions": args.predictions,
    }
    seal = create_output_seal(
        project_root=project_root,
        protocol_freeze_path=args.freeze_manifest,
        output_path=args.out,
        artifact_paths=artifacts,
    )
    print(json.dumps(seal, ensure_ascii=False, indent=2))
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    _ensure_new([args.decisions_out, args.selected_out], force=args.force)
    requests = list(read_jsonl(args.requests))
    local = [row for path in args.local_responses for row in read_jsonl(path)]
    if args.split and not args.labels:
        raise ValueError("--split requires --labels on the trusted laptop")
    if args.labels:
        label_rows = [row for path in args.labels for row in read_jsonl(path)]
        allowed_ids = {
            _case_id(row)
            for row in label_rows
            if args.split is None or str(row.get("split")) == args.split
        }
        requests = [row for row in requests if _case_id(row) in allowed_ids]
        local = [row for row in local if _case_id(row) in allowed_ids]
    decisions = select_remote_calls(
        requests,
        local,
        policy=args.policy,
        budget_fraction=args.budget_fraction,
        seed=args.seed,
        call_budget=args.call_budget,
    )
    decision_rows = [row.to_dict() for row in decisions]
    write_jsonl(args.decisions_out, decision_rows)
    write_jsonl(args.selected_out, selected_requests(requests, decision_rows))
    selected_count = sum(int(row.selected) for row in decisions)
    print(
        json.dumps(
            {
                "policy": args.policy,
                "cases": len(requests),
                "split": args.split,
                "selected": selected_count,
                "actual_call_rate": selected_count / len(requests) if requests else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    _ensure_new([args.out], force=args.force)
    requests = list(read_jsonl(args.requests))
    local = [row for path in args.local_responses for row in read_jsonl(path)]
    decisions = list(read_jsonl(args.decisions))
    remote = [row for path in args.remote_responses for row in read_jsonl(path)]
    merged = merge_predictions(requests, local, decisions, remote)
    write_jsonl(args.out, merged)
    print(json.dumps({"predictions": len(merged), "out": str(args.out)}, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .engineering import engineering_stats

    _ensure_new([args.out], force=args.force)
    report = engineering_stats(
        read_jsonl(args.requests),
        [row for path in args.responses for row in read_jsonl(path)],
        [row for path in args.errors for row in read_jsonl(path)],
    )
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_rows
    from .freeze import verify_freeze_manifest
    from .sealing import verify_output_seal

    project_root = Path(__file__).resolve().parents[2]
    freeze = verify_freeze_manifest(args.freeze_manifest, project_root=project_root)
    seal = verify_output_seal(args.output_seal, project_root=project_root)
    if seal["protocol_freeze_sha256"] != file_sha256(args.freeze_manifest):
        raise ValueError("output seal is not bound to the supplied protocol freeze")
    if len(args.labels) != 1 or file_sha256(args.labels[0]) != freeze["blind_label_commitment"]:
        raise ValueError("evaluation labels do not match the frozen blind-label commitment")
    sealed_prediction_hash = seal["artifacts"]["predictions"]["sha256"]
    if file_sha256(args.predictions) != sealed_prediction_hash:
        raise ValueError("evaluation predictions do not match the sealed predictions")
    _ensure_new([args.out], force=args.force)
    engineering = read_json(args.engineering_stats) if args.engineering_stats else None
    split_stats_args = (args.corpus_manifest, args.local_stats, args.remote_stats)
    if any(split_stats_args):
        if not all(split_stats_args) or engineering is not None:
            raise ValueError(
                "use either --engineering-stats or all of --corpus-manifest, "
                "--local-stats, and --remote-stats"
            )
        corpus = read_json(args.corpus_manifest)
        local_stats = read_json(args.local_stats)
        remote_stats = read_json(args.remote_stats)
        engineering = {
            "request_count": int(corpus["episodes"]),
            "input_contract_valid_count": int(corpus["input_contract_valid_count"]),
            "witness_success_count": int(corpus["witness_success_count"]),
            "invalid_false_pass_count": int(corpus["invalid_false_pass_count"]),
            "median_witness_tokens": float(corpus["median_witness_tokens"]),
            "local_parse_success_count": int(local_stats["parse_success_count"]),
            "local_parse_total_count": int(local_stats["parse_total_count"]),
            "remote_parse_success_count": int(remote_stats["parse_success_count"]),
            "remote_parse_total_count": int(remote_stats["parse_total_count"]),
        }
    label_rows = [row for path in args.labels for row in read_jsonl(path)]
    report = evaluate_rows(
        label_rows,
        read_jsonl(args.predictions),
        engineering_stats=engineering,
    )
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .freeze import verify_freeze_manifest
    from .pareto import analyze_pareto, pareto_csv_rows
    from .sealing import verify_output_seal

    project_root = Path(__file__).resolve().parents[2]
    freeze = verify_freeze_manifest(args.freeze_manifest, project_root=project_root)
    seal = verify_output_seal(args.output_seal, project_root=project_root)
    if seal["protocol_freeze_sha256"] != file_sha256(args.freeze_manifest):
        raise ValueError("output seal is not bound to the supplied protocol freeze")
    if file_sha256(args.labels) != freeze["blind_label_commitment"]:
        raise ValueError("Pareto labels do not match the blind-label commitment")
    paths = {
        name: project_root / metadata["path"]
        for name, metadata in seal["artifacts"].items()
    }
    _ensure_new([args.out_json, args.out_csv], force=args.force)
    report = analyze_pareto(
        read_jsonl(paths["blind_requests"]),
        read_jsonl(args.labels),
        read_jsonl(paths["local_responses"]),
        read_jsonl(paths["remote_responses"]),
        read_jsonl(paths["remote_records"]),
        seed=int(freeze["routing"]["seed"]),
    )
    report["protocol_freeze_sha256"] = file_sha256(args.freeze_manifest)
    report["output_seal_sha256"] = file_sha256(args.output_seal)
    write_json(args.out_json, report)
    csv_rows = pareto_csv_rows(report)
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_rows:
        raise ValueError("Pareto report has no rows")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        json.dumps(
            {
                "rows": len(csv_rows),
                "frontier_rows": len(report["frontier"]),
                "json": str(args.out_json),
                "csv": str(args.out_csv),
                "remote_all_actual_cost": report["remote_all_actual_cost"],
                "currency": report["currency"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    root = Path(args.out_dir)
    corpus_dir = root / "corpus"
    manifest = generate_corpus(corpus_dir)
    requests = list(read_jsonl(corpus_dir / "gpu" / "requests.jsonl"))
    prompt = "Smoke-test only. Return the required JSON object."
    local, local_errors, local_records = run_backend(
        requests,
        backend=MockBackend(model_id="mock-local-smoke-only", remote=False),
        prompt=prompt,
        run_id="smoke-local",
        model_role="local",
        prompt_version="local_v1",
    )
    label_rows = [
        *read_jsonl(corpus_dir / "private" / "prompt_dev_labels.jsonl"),
        *read_jsonl(corpus_dir / "private" / "blind_labels.jsonl"),
    ]
    blind_ids = {_case_id(row) for row in label_rows if row.get("split") == "blind"}
    blind_requests = [row for row in requests if _case_id(row) in blind_ids]
    blind_local = [row for row in local if _case_id(row) in blind_ids]
    decisions = select_remote_calls(
        blind_requests,
        blind_local,
        policy="flowgate-v0",
        budget_fraction=0.25,
        seed=7,
    )
    decision_rows = [row.to_dict() for row in decisions]
    routed_requests = selected_requests(requests, decision_rows)
    remote, remote_errors, remote_records = run_backend(
        routed_requests,
        backend=MockBackend(model_id="mock-remote-smoke-only", remote=True),
        prompt=prompt,
        run_id="smoke-remote",
        model_role="remote",
        prompt_version="remote_v1",
    )
    merged = merge_predictions(requests, local, decision_rows, remote)
    write_jsonl(root / "local_responses.jsonl", local)
    write_jsonl(root / "routing_decisions.jsonl", decision_rows)
    write_jsonl(root / "selected_remote_requests.jsonl", routed_requests)
    write_jsonl(root / "remote_responses.jsonl", remote)
    write_jsonl(root / "run_records.jsonl", local_records + remote_records)
    write_jsonl(root / "predictions.jsonl", merged)
    write_jsonl(root / "errors.jsonl", local_errors + remote_errors)
    report = {
        "status": "PASS" if not local_errors and not remote_errors else "FAIL",
        "smoke_only": True,
        "warning": "Mock outputs validate plumbing only and must not be reported as research results.",
        "corpus": manifest,
        "local_responses": len(local),
        "remote_calls": len(remote),
        "blind_cases": len(blind_requests),
        "merged_predictions": len(merged),
        "errors": len(local_errors) + len(remote_errors),
    }
    write_json(root / "SMOKE_ONLY.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowgate-pilot",
        description="CPU-first, label-blind feasibility pilot for FlowGate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="build the fixed 24-case corpus")
    generate.add_argument("--out-dir", default="data/generated")
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=cmd_generate)

    bundle = subparsers.add_parser(
        "bundle-gpu", help="create a label-free archive for the separate GPU machine"
    )
    bundle.add_argument("--requests", default="data/generated/gpu/requests.jsonl")
    bundle.add_argument(
        "--labels",
        nargs="+",
        default=[
            "data/generated/private/prompt_dev_labels.jsonl",
            "data/generated/private/blind_labels.jsonl",
        ],
    )
    bundle.add_argument("--split", choices=("prompt_dev", "blind", "all"), default="prompt_dev")
    bundle.add_argument("--freeze-manifest")
    bundle.add_argument("--out", default="dist/flowgate-gpu-bundle.tar.gz")
    bundle.set_defaults(func=cmd_bundle_gpu)

    freeze = subparsers.add_parser(
        "freeze", help="hash prompts, code, models, routing, labels, and pricing before blind run"
    )
    freeze.add_argument("--out", default="results/protocol-freeze.json")
    freeze.add_argument("--local-model-id", required=True)
    freeze.add_argument("--local-model-revision", required=True)
    freeze.add_argument("--remote-model-id", required=True)
    freeze.add_argument("--remote-model-revision", required=True)
    freeze.add_argument("--remote-provider", required=True)
    freeze.add_argument("--provider-retention", required=True)
    freeze.add_argument("--pricing-snapshot", required=True)
    freeze.add_argument("--input-price-per-million", type=float, required=True)
    freeze.add_argument("--output-price-per-million", type=float, required=True)
    freeze.add_argument("--currency", default="USD")
    freeze.set_defaults(func=cmd_freeze)

    run = subparsers.add_parser("run-batch", help="run a local or remote model batch")
    run.add_argument("--requests", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--errors")
    run.add_argument("--records")
    run.add_argument("--run-id", default="manual-run")
    run.add_argument("--backend", choices=("mock", "openai"), default="openai")
    run.add_argument("--base-url")
    run.add_argument("--model-id")
    run.add_argument("--api-key-env")
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--remote", action="store_true")
    run.add_argument("--phase", choices=("prompt_dev", "blind"), default="prompt_dev")
    run.add_argument("--freeze-manifest")
    run.add_argument("--model-revision", required=False)
    run.add_argument("--quantization")
    run.add_argument("--runtime-metadata", help="JSON captured from Python/GPU/runtime version commands")
    run.add_argument("--server-max-model-len", type=int)
    run.add_argument("--server-gpu-memory-utilization", type=float)
    run.add_argument("--input-price-per-million", type=float)
    run.add_argument("--output-price-per-million", type=float)
    run.add_argument("--currency")
    run.add_argument("--pricing-snapshot")
    run.add_argument("--force", action="store_true", help="explicitly overwrite prior artifacts")
    run.set_defaults(func=cmd_run_batch)

    export = subparsers.add_parser(
        "export-split", help="export a sanitized prompt-development or frozen blind request file"
    )
    export.add_argument("--requests", default="data/generated/gpu/requests.jsonl")
    export.add_argument(
        "--labels",
        nargs="+",
        default=["data/generated/private/prompt_dev_labels.jsonl"],
    )
    export.add_argument("--split", choices=("prompt_dev", "blind"), required=True)
    export.add_argument("--freeze-manifest")
    export.add_argument("--out", required=True)
    export.add_argument("--manifest-out")
    export.add_argument("--force", action="store_true")
    export.set_defaults(func=cmd_export_split)

    select = subparsers.add_parser("select", help="rank cases under a remote-call budget")
    select.add_argument("--requests", required=True)
    select.add_argument("--local-responses", nargs="+", required=True)
    select.add_argument(
        "--labels",
        nargs="+",
        help="trusted laptop-only label JSONL files used only to select a split",
    )
    select.add_argument("--split", choices=("prompt_dev", "blind"))
    select.add_argument("--decisions-out", required=True)
    select.add_argument("--selected-out", required=True)
    select.add_argument("--policy", choices=sorted(SUPPORTED_POLICIES), default="flowgate-v0")
    select.add_argument("--budget-fraction", type=float, default=0.25)
    select.add_argument("--call-budget", type=int)
    select.add_argument("--seed", type=int, default=7)
    select.add_argument("--force", action="store_true")
    select.set_defaults(func=cmd_select)

    merge = subparsers.add_parser("merge", help="merge local and selected remote outputs")
    merge.add_argument("--requests", required=True)
    merge.add_argument("--local-responses", nargs="+", required=True)
    merge.add_argument("--decisions", required=True)
    merge.add_argument("--remote-responses", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(func=cmd_merge)

    stats = subparsers.add_parser("stats", help="build the frozen engineering counters")
    stats.add_argument("--requests", required=True)
    stats.add_argument("--responses", nargs="+", required=True)
    stats.add_argument("--errors", nargs="+", required=True)
    stats.add_argument("--out", required=True)
    stats.add_argument("--force", action="store_true")
    stats.set_defaults(func=cmd_stats)

    evaluate = subparsers.add_parser("evaluate", help="compute metrics and Go/No-Go")
    evaluate.add_argument("--labels", nargs="+", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--engineering-stats")
    evaluate.add_argument("--corpus-manifest")
    evaluate.add_argument("--local-stats")
    evaluate.add_argument("--remote-stats")
    evaluate.add_argument("--freeze-manifest", required=True)
    evaluate.add_argument("--output-seal", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--force", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser(
        "compare", help="replay fixed blind outputs across routing budgets and build a Pareto table"
    )
    compare.add_argument("--freeze-manifest", required=True)
    compare.add_argument("--output-seal", required=True)
    compare.add_argument("--labels", required=True)
    compare.add_argument("--out-json", required=True)
    compare.add_argument("--out-csv", required=True)
    compare.add_argument("--force", action="store_true")
    compare.set_defaults(func=cmd_compare)

    seal = subparsers.add_parser(
        "seal-outputs", help="commit blind model artifacts before unsealing labels"
    )
    seal.add_argument("--freeze-manifest", required=True)
    seal.add_argument("--requests", required=True)
    seal.add_argument("--local-responses", required=True)
    seal.add_argument("--local-errors", required=True)
    seal.add_argument("--local-records", required=True)
    seal.add_argument("--local-manifest", required=True)
    seal.add_argument("--decisions", required=True)
    seal.add_argument("--remote-responses", required=True)
    seal.add_argument("--remote-errors", required=True)
    seal.add_argument("--remote-records", required=True)
    seal.add_argument("--remote-manifest", required=True)
    seal.add_argument("--predictions", required=True)
    seal.add_argument("--out", default="results/blind-output-seal.json")
    seal.set_defaults(func=cmd_seal_outputs)

    smoke = subparsers.add_parser("smoke", help="validate plumbing with mock models only")
    smoke.add_argument("--out-dir", default="results/smoke")
    smoke.set_defaults(func=cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
