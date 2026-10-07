from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from flowgate.bundle import create_gpu_bundle
from flowgate.cli import generate_corpus
from flowgate.freeze import (
    VLLM_RUNTIME_ENVIRONMENT,
    VLLM_RUNTIME_VERSION,
    create_freeze_manifest,
)
from flowgate.io import write_json


class BundleTests(unittest.TestCase):
    def test_bundle_contains_worker_inputs_but_no_private_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            corpus = temp / "corpus"
            generate_corpus(corpus)
            archive_path = temp / "worker.tar.gz"
            manifest = create_gpu_bundle(
                project_root=project_root,
                requests_path=corpus / "gpu" / "requests.jsonl",
                output_path=archive_path,
            )
            self.assertEqual(manifest["request_count"], 24)
            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
                self.assertIn("data/requests.jsonl", names)
                self.assertIn("GPU_README.md", names)
                # cli._code_fingerprint imports SAFE_WORKER_FILES at the end of
                # every run, so the bundle module is part of the runtime too.
                self.assertIn("src/flowgate/bundle.py", names)
                # Blind worker validation imports the pinned runtime contract
                # lazily from this module; omitting it breaks only after transfer.
                self.assertIn("src/flowgate/freeze.py", names)
                self.assertFalse(any("private" in name or "label" in name for name in names))
                payload = archive.extractfile("data/requests.jsonl")
                self.assertIsNotNone(payload)
                rows = [json.loads(line) for line in payload.read().decode("utf-8").splitlines()]
                self.assertEqual(len(rows), 24)
                self.assertTrue(all("split" not in row and "malicious" not in row for row in rows))

    def test_extracted_blind_bundle_can_execute_the_worker_cli(self) -> None:
        """Catch archive-only import failures before transferring to the GPU."""

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            freeze_path = temp / "protocol-freeze-source.json"
            freeze = create_freeze_manifest(
                project_root=project_root,
                output_path=freeze_path,
                local_model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
                local_model_revision="b25037543e9394b818fdfca67ab2a00ecc7dd641",
                remote_model_id="remote-test-model",
                remote_model_revision="remote-test-revision",
                remote_provider="test-provider",
                provider_retention="test-only",
                pricing_snapshot="test-only",
                input_price_per_million=0.0,
                output_price_per_million=0.0,
                currency="USD",
            )
            archive_path = temp / "blind-worker.tar.gz"
            create_gpu_bundle(
                project_root=project_root,
                requests_path=project_root / "data/generated/gpu/requests.jsonl",
                output_path=archive_path,
                include_ids=set(freeze["blind_episode_ids"]),
                protocol_freeze_path=freeze_path,
            )

            extracted = temp / "extracted"
            extracted.mkdir()
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extracted, filter="data")
            runtime_path = extracted / "runtime.json"
            write_json(runtime_path, {
                "packages": {"vllm": VLLM_RUNTIME_VERSION},
                "environment": dict(VLLM_RUNTIME_ENVIRONMENT),
            })

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flowgate.cli",
                    "run-batch",
                    "--requests",
                    "data/requests.jsonl",
                    "--prompt",
                    "prompts/local_v1.txt",
                    "--out",
                    "results/local_responses.jsonl",
                    "--records",
                    "results/local_run_records.jsonl",
                    "--errors",
                    "results/local_errors.jsonl",
                    "--run-id",
                    "blind-bundle-mock",
                    "--backend",
                    "mock",
                    "--model-id",
                    "Qwen/Qwen2.5-7B-Instruct-AWQ",
                    "--model-revision",
                    "b25037543e9394b818fdfca67ab2a00ecc7dd641",
                    "--quantization",
                    "awq",
                    "--phase",
                    "blind",
                    "--freeze-manifest",
                    "protocol-freeze.json",
                    "--runtime-metadata",
                    "runtime.json",
                    "--server-max-model-len",
                    "8192",
                    "--server-gpu-memory-utilization",
                    "0.85",
                ],
                cwd=extracted,
                env={**os.environ, "PYTHONPATH": str(extracted / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            responses = (extracted / "results/local_responses.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(responses), 18)
            self.assertTrue(
                (extracted / "results/local_responses.jsonl.manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
