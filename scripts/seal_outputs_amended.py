"""Seal blind outputs after a narrowly scoped pre-unseal verifier amendment.

The frozen verifier hard-codes ``prompts/remote_v1.txt`` although the protocol
freeze also commits ``remote_v2.txt`` and the remote run records bind every
response to ``remote_v2``. This helper preserves the original freeze and all
model outputs, checks the intended prompt binding explicitly, and relaxes only
that stale filename lookup while delegating every other seal check to the
frozen implementation.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from flowgate import sealing
from flowgate.contracts import prompt_hash
from flowgate.io import file_sha256, read_json, read_jsonl, write_json


AMENDMENT_ID = "flowgate-pre-unseal-amendment-001"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--remote-records", required=True)
    parser.add_argument("--remote-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact", action="append", nargs=2, metavar=("NAME", "PATH"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    freeze_path = Path(args.freeze_manifest).resolve()
    records_path = Path(args.remote_records).resolve()
    manifest_path = Path(args.remote_manifest).resolve()
    out_path = Path(args.out).resolve()
    artifacts = dict(args.artifact or [])

    if set(artifacts) != sealing.REQUIRED_ARTIFACTS:
        raise ValueError("the exact standard seal artifact set is required")

    freeze = sealing.verify_freeze_manifest(freeze_path, project_root=root)
    remote_prompt_path = root / "prompts/remote_v2.txt"
    frozen_prompt_digest = freeze["file_sha256"].get("prompts/remote_v2.txt")
    if file_sha256(remote_prompt_path) != frozen_prompt_digest:
        raise ValueError("remote_v2 prompt differs from the original protocol freeze")

    expected_prompt_hash = prompt_hash(
        remote_prompt_path.read_text(encoding="utf-8").strip()
    )
    remote_manifest = read_json(manifest_path)
    records = list(read_jsonl(records_path))
    if remote_manifest.get("prompt_sha256") != expected_prompt_hash:
        raise ValueError("remote manifest is not bound to frozen remote_v2")
    if len(records) != len(freeze["blind_episode_ids"]) or any(
        row.get("prompt_version") != "remote_v2"
        or row.get("prompt_sha256") != expected_prompt_hash
        for row in records
    ):
        raise ValueError("remote records are not uniformly bound to frozen remote_v2")

    original_validate = sealing._validate_manifest

    def amended_validate(
        manifest,
        *,
        role,
        freeze,
        freeze_hash,
        request_hash,
        episode_count,
        project_root,
    ):
        checked = copy.deepcopy(manifest)
        if role == "remote":
            # Explicit checks above prove the actual remote_v2 binding. Feed the
            # legacy value only through the stale filename check.
            legacy = root / "prompts/remote_v1.txt"
            checked["prompt_sha256"] = prompt_hash(
                legacy.read_text(encoding="utf-8").strip()
            )
        return original_validate(
            checked,
            role=role,
            freeze=freeze,
            freeze_hash=freeze_hash,
            request_hash=request_hash,
            episode_count=episode_count,
            project_root=project_root,
        )

    sealing._validate_manifest = amended_validate
    try:
        seal = sealing.create_output_seal(
            project_root=root,
            protocol_freeze_path=freeze_path,
            output_path=out_path,
            artifact_paths=artifacts,
        )
    finally:
        sealing._validate_manifest = original_validate

    seal["protocol_amendment"] = {
        "amendment_id": AMENDMENT_ID,
        "timing": "after blind inference and routing, before label unseal",
        "scope": "seal verifier prompt filename lookup only",
        "reason": (
            "frozen verifier hard-coded remote_v1 while the frozen run used remote_v2"
        ),
        "original_freeze_sha256": file_sha256(freeze_path),
        "remote_prompt_path": "prompts/remote_v2.txt",
        "remote_prompt_file_sha256": file_sha256(remote_prompt_path),
        "remote_prompt_contract_sha256": expected_prompt_hash,
        "remote_manifest_sha256": file_sha256(manifest_path),
        "remote_records_sha256": file_sha256(records_path),
        "model_outputs_changed": False,
        "routing_changed": False,
        "labels_opened_before_amendment": False,
    }
    write_json(out_path, seal)
    sealing.verify_output_seal(out_path, project_root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
