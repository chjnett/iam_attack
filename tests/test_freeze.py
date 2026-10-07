from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from flowgate.freeze import (
    VLLM_RUNTIME_VERSION,
    VLLM_RUNTIME_ENVIRONMENT,
    create_freeze_manifest,
    validate_local_runtime_metadata,
    verify_freeze_manifest,
)
from flowgate.io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FreezeTests(unittest.TestCase):
    def test_complete_manifest_verifies_and_empty_hash_map_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "freeze.json"
            manifest = create_freeze_manifest(
                project_root=PROJECT_ROOT,
                output_path=target,
                local_model_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
                local_model_revision="b25037543e9394b818fdfca67ab2a00ecc7dd641",
                remote_model_id="remote-test",
                remote_model_revision="remote-revision",
                remote_provider="test-provider",
                provider_retention="disabled",
                pricing_snapshot="2026-10-07",
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                currency="USD",
            )
            verified = verify_freeze_manifest(target, project_root=PROJECT_ROOT)
            self.assertEqual(len(verified["blind_episode_ids"]), 18)
            self.assertEqual(
                verified["models"]["local"]["runtime"]["vllm"],
                VLLM_RUNTIME_VERSION,
            )

            wrong_runtime = Path(directory) / "wrong-runtime.json"
            manifest["models"]["local"]["runtime"]["vllm"] = "0.30.0"
            write_json(wrong_runtime, manifest)
            with self.assertRaisesRegex(ValueError, "vLLM runtime commitment"):
                verify_freeze_manifest(wrong_runtime, project_root=PROJECT_ROOT)

            fake = Path(directory) / "fake.json"
            write_json(
                fake,
                {
                    "schema_version": manifest["schema_version"],
                    "protocol_id": manifest["protocol_id"],
                    "status": "frozen_before_blind_run",
                    "file_sha256": {},
                },
            )
            with self.assertRaisesRegex(ValueError, "exact required file set"):
                verify_freeze_manifest(fake, project_root=PROJECT_ROOT)

    def test_negative_price_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                create_freeze_manifest(
                    project_root=PROJECT_ROOT,
                    output_path=Path(directory) / "freeze.json",
                    local_model_id="local",
                    local_model_revision="revision",
                    remote_model_id="remote",
                    remote_model_revision="revision",
                    remote_provider="provider",
                    provider_retention="disabled",
                    pricing_snapshot="2026-10-07",
                    input_price_per_million=-1,
                    output_price_per_million=2,
                    currency="USD",
                )

    def test_local_runtime_metadata_must_match_vllm_commitment(self) -> None:
        metadata = {
            "packages": {"vllm": VLLM_RUNTIME_VERSION},
            "environment": dict(VLLM_RUNTIME_ENVIRONMENT),
            "gpus": [{"name": "NVIDIA GeForce RTX 3090"}],
        }
        self.assertIs(validate_local_runtime_metadata(metadata), metadata)
        with self.assertRaisesRegex(ValueError, "vLLM runtime differs"):
            validate_local_runtime_metadata({
                "packages": {"vllm": "0.30.0"},
                "environment": dict(VLLM_RUNTIME_ENVIRONMENT),
            })
        with self.assertRaisesRegex(ValueError, "runtime environment differs"):
            validate_local_runtime_metadata({
                "packages": {"vllm": VLLM_RUNTIME_VERSION},
                "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "1"},
            })


if __name__ == "__main__":
    unittest.main()
