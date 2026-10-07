from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from flowgate.batch import merge_predictions, read_prompt, run_backend
from flowgate.cli import _code_fingerprint
from flowgate.contracts import make_run_manifest
from flowgate.freeze import (
    VLLM_RUNTIME_ENVIRONMENT,
    VLLM_RUNTIME_VERSION,
    create_freeze_manifest,
)
from flowgate.inference import MockBackend
from flowgate.io import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from flowgate.routing import select_remote_calls
from flowgate.sealing import create_output_seal, verify_output_seal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"
LOCAL_MODEL_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
REMOTE_MODEL_ID = "remote-test"
REMOTE_MODEL_REVISION = "remote-test-2026-10-07"
DECODING = {"temperature": 0, "top_p": 1, "max_output_tokens": 700}


class OutputSealEndToEndTests(unittest.TestCase):
    def _write_run_manifest(
        self,
        *,
        path: Path,
        role: str,
        prompt: str,
        request_path: Path,
        freeze_path: Path,
        response_count: int,
    ) -> None:
        model_id = LOCAL_MODEL_ID if role == "local" else REMOTE_MODEL_ID
        model_revision = (
            LOCAL_MODEL_REVISION if role == "local" else REMOTE_MODEL_REVISION
        )
        manifest = make_run_manifest(
            run_id=f"blind-{role}-mock",
            model_id=model_id,
            prompt=prompt,
            decoding=DECODING,
            input_sha256=file_sha256(request_path),
            code_version=_code_fingerprint(PROJECT_ROOT),
            backend="mock",
        )
        manifest.update(
            {
                "request_count": response_count,
                "valid_response_count": response_count,
                "error_count": 0,
                "run_record_count": response_count,
                "smoke_only": True,
                "endpoint_origin": None,
                "model_revision": model_revision,
                "phase": "blind",
                "protocol_freeze_sha256": file_sha256(freeze_path),
                "quantization": "awq" if role == "local" else None,
                "serving_config": (
                    {"max_model_len": 8192, "gpu_memory_utilization": 0.85}
                    if role == "local"
                    else None
                ),
                "runtime_metadata": (
                    {
                        "capture": "unit-test",
                        "packages": {"vllm": VLLM_RUNTIME_VERSION},
                        "environment": dict(VLLM_RUNTIME_ENVIRONMENT),
                        "gpu": {"name": "NVIDIA GeForce RTX 3090"},
                    }
                    if role == "local"
                    else None
                ),
                "pricing": (
                    {
                        "input_per_million_tokens": 1.0,
                        "output_per_million_tokens": 2.0,
                        "currency": "USD",
                        "pricing_snapshot": "2026-10-07",
                    }
                    if role == "remote"
                    else None
                ),
            }
        )
        write_json(path, manifest)

    def test_freeze_blind_runs_route_merge_and_output_seal(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            work = Path(directory)
            freeze_path = work / "protocol-freeze.json"
            freeze = create_freeze_manifest(
                project_root=PROJECT_ROOT,
                output_path=freeze_path,
                local_model_id=LOCAL_MODEL_ID,
                local_model_revision=LOCAL_MODEL_REVISION,
                remote_model_id=REMOTE_MODEL_ID,
                remote_model_revision=REMOTE_MODEL_REVISION,
                remote_provider="unit-test-provider",
                provider_retention="disabled",
                pricing_snapshot="2026-10-07",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                currency="USD",
            )

            blind_ids = set(freeze["blind_episode_ids"])
            requests = [
                row
                for row in read_jsonl(
                    PROJECT_ROOT / "data/generated/gpu/requests.jsonl"
                )
                if row["episode_id"] in blind_ids
            ]
            self.assertEqual(len(requests), 18)
            request_path = work / "blind-requests.jsonl"
            write_jsonl(request_path, requests)

            local_prompt = read_prompt(PROJECT_ROOT / "prompts/local_v1.txt")
            remote_prompt = read_prompt(PROJECT_ROOT / "prompts/remote_v1.txt")
            local, local_errors, local_records = run_backend(
                requests,
                backend=MockBackend(model_id=LOCAL_MODEL_ID),
                prompt=local_prompt,
                run_id="blind-local-mock",
                model_role="local",
                prompt_version="local_v1",
            )
            remote, remote_errors, remote_records = run_backend(
                requests,
                backend=MockBackend(model_id=REMOTE_MODEL_ID, remote=True),
                prompt=remote_prompt,
                run_id="blind-remote-mock",
                model_role="remote",
                prompt_version="remote_v1",
                pricing={
                    "input_per_million_tokens": 1.0,
                    "output_per_million_tokens": 2.0,
                    "currency": "USD",
                    "pricing_snapshot": "2026-10-07",
                },
            )
            self.assertEqual((len(local), len(remote)), (18, 18))
            self.assertEqual((local_errors, remote_errors), ([], []))

            decisions = [
                decision.to_dict()
                for decision in select_remote_calls(
                    requests,
                    local,
                    policy="flowgate-v0",
                    budget_fraction=0.25,
                    seed=7,
                    call_budget=4,
                )
            ]
            self.assertEqual(sum(row["route_to_remote"] for row in decisions), 4)
            predictions = merge_predictions(requests, local, decisions, remote)

            paths = {
                "blind_requests": request_path,
                "local_responses": work / "local-responses.jsonl",
                "local_errors": work / "local-errors.jsonl",
                "local_records": work / "local-records.jsonl",
                "local_manifest": work / "local-manifest.json",
                "routing_decisions": work / "routing-decisions.jsonl",
                "remote_responses": work / "remote-responses.jsonl",
                "remote_errors": work / "remote-errors.jsonl",
                "remote_records": work / "remote-records.jsonl",
                "remote_manifest": work / "remote-manifest.json",
                "predictions": work / "predictions.jsonl",
            }
            write_jsonl(paths["local_responses"], local)
            write_jsonl(paths["local_errors"], local_errors)
            write_jsonl(paths["local_records"], local_records)
            write_jsonl(paths["routing_decisions"], decisions)
            write_jsonl(paths["remote_responses"], remote)
            write_jsonl(paths["remote_errors"], remote_errors)
            write_jsonl(paths["remote_records"], remote_records)
            write_jsonl(paths["predictions"], predictions)
            self._write_run_manifest(
                path=paths["local_manifest"],
                role="local",
                prompt=local_prompt,
                request_path=request_path,
                freeze_path=freeze_path,
                response_count=18,
            )
            self._write_run_manifest(
                path=paths["remote_manifest"],
                role="remote",
                prompt=remote_prompt,
                request_path=request_path,
                freeze_path=freeze_path,
                response_count=18,
            )

            wrong_runtime = read_json(paths["local_manifest"])
            wrong_runtime["runtime_metadata"]["packages"]["vllm"] = "0.30.0"
            write_json(paths["local_manifest"], wrong_runtime)
            with self.assertRaisesRegex(ValueError, "vLLM runtime differs"):
                create_output_seal(
                    project_root=PROJECT_ROOT,
                    protocol_freeze_path=freeze_path,
                    output_path=work / "bad-output-seal.json",
                    artifact_paths=paths,
                )
            self._write_run_manifest(
                path=paths["local_manifest"],
                role="local",
                prompt=local_prompt,
                request_path=request_path,
                freeze_path=freeze_path,
                response_count=18,
            )

            seal_path = work / "output-seal.json"
            seal = create_output_seal(
                project_root=PROJECT_ROOT,
                protocol_freeze_path=freeze_path,
                output_path=seal_path,
                artifact_paths=paths,
            )
            self.assertEqual(seal["blind_episode_count"], 18)
            self.assertTrue(seal["remote_all_counterfactual"])
            verified = verify_output_seal(seal_path, project_root=PROJECT_ROOT)
            self.assertEqual(verified["blind_episode_ids"], sorted(blind_ids))

            with paths["predictions"].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "changed after commitment"):
                verify_output_seal(seal_path, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
