from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from .io import canonical_json, file_sha256, read_json, read_jsonl
from .privacy import validate_gpu_export


SAFE_WORKER_FILES = (
    "pyproject.toml",
    "scripts/capture_runtime.py",
    "src/flowgate/__init__.py",
    "src/flowgate/anonymize.py",
    "src/flowgate/batch.py",
    "src/flowgate/bundle.py",
    "src/flowgate/cli.py",
    "src/flowgate/contracts.py",
    "src/flowgate/engineering.py",
    "src/flowgate/evaluate.py",
    "src/flowgate/freeze.py",
    "src/flowgate/inference.py",
    "src/flowgate/io.py",
    "src/flowgate/metrics.py",
    "src/flowgate/privacy.py",
    "src/flowgate/routing.py",
    "prompts/local_v1.txt",
    "schemas/worker-input.schema.json",
    "schemas/model-output.schema.json",
    "schemas/run-record.schema.json",
)

GPU_README = """# FlowGate sanitized RTX 3090 worker

This machine is an execution-only, label-blind worker. This archive intentionally
contains no labels, split membership, raw episodes, pseudonym map, remote API key,
remote-model prompt, or remote-model output. Never add them here, and never call
the remote model API from this machine.

The frozen local checkpoint is the official Apache-2.0
`Qwen/Qwen2.5-7B-Instruct-AWQ` 4-bit AWQ model. The revision below is an immutable
Hugging Face commit. vLLM detects AWQ from the checkpoint configuration; do not
add an online quantization step.

Official references:

- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ

On Linux (or WSL2), create a clean environment and start the localhost server.
The 8,192-token cap and 0.85 memory fraction leave practical headroom on a
24 GB RTX 3090 for CUDA graphs, KV cache, and display use:

```bash
uv venv --python 3.12 --seed .venv-vllm
source .venv-vllm/bin/activate
uv pip install 'vllm==0.31.0' --torch-backend=auto

export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve "$FLOWGATE_LOCAL_MODEL" \\
  --revision "$FLOWGATE_LOCAL_REVISION" \\
  --host 127.0.0.1 \\
  --port 8000 \\
  --dtype auto \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.85 \\
  --generation-config vllm \\
  --api-key "$FLOWGATE_LOCAL_KEY"
```

In a second shell, activate the same environment. A six-request development
archive must run with `--phase prompt_dev`. An 18-request blind archive contains
`protocol-freeze.json` and must run with both `--phase blind` and
`--freeze-manifest protocol-freeze.json`. The branch below enforces that choice.

```bash
source .venv-vllm/bin/activate
export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key
export VLLM_USE_FLASHINFER_SAMPLER=0
if test -f protocol-freeze.json; then
  export FLOWGATE_RUN_ID=blind-local-v1
  python3 scripts/capture_runtime.py > runtime.json
  PHASE_ARGS="--phase blind --freeze-manifest protocol-freeze.json --runtime-metadata runtime.json --server-max-model-len 8192 --server-gpu-memory-utilization 0.85"
else
  export FLOWGATE_RUN_ID=prompt-dev-local-v1
  PHASE_ARGS="--phase prompt_dev"
fi

mkdir -p results
PYTHONPATH=src python3 -m flowgate.cli run-batch \\
  --requests data/requests.jsonl \\
  --prompt prompts/local_v1.txt \\
  --out results/local_responses.jsonl \\
  --records results/local_run_records.jsonl \\
  --errors results/local_errors.jsonl \\
  --run-id "$FLOWGATE_RUN_ID" \\
  --backend openai \\
  --base-url http://127.0.0.1:8000/v1 \\
  --model-id "$FLOWGATE_LOCAL_MODEL" \\
  --model-revision "$FLOWGATE_LOCAL_REVISION" \\
  --quantization awq \\
  --api-key-env FLOWGATE_LOCAL_KEY \\
  $PHASE_ARGS
```

For a blind run, `runtime.json` records the secret-free Python, package, driver,
and GPU snapshot and the two `--server-*` arguments bind the actual localhost
server settings to the frozen 8,192/0.85 commitment. `run-batch` automatically
writes the fourth artifact,
`results/local_responses.jsonl.manifest.json`. Return exactly these four files to
the trusted laptop: the responses, run records, errors, and automatic run
manifest; `runtime.json` is embedded in that manifest and is not a fifth return
file. The trusted laptop alone retains labels and remote credentials, calls
the remote model independently on all 18 blind requests exactly once, stores
remote responses, selects the four `flowgate-v0` routes, seals outputs, and only
then evaluates or compares against blind labels.
"""


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o600
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def create_gpu_bundle(
    *,
    project_root: str | Path,
    requests_path: str | Path,
    output_path: str | Path,
    include_ids: set[str] | None = None,
    protocol_freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    request_file = Path(requests_path).resolve()
    output = Path(output_path)
    rows = list(read_jsonl(request_file))
    if include_ids is not None:
        available = {str(row.get("episode_id")) for row in rows}
        missing_ids = include_ids - available
        if missing_ids:
            raise ValueError(f"bundle filter references unknown episodes: {sorted(missing_ids)}")
        rows = [row for row in rows if str(row.get("episode_id")) in include_ids]
    if not rows:
        raise ValueError("GPU bundle cannot contain an empty request set")
    for row in rows:
        validate_gpu_export(row)

    missing = [relative for relative in SAFE_WORKER_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing worker files: {missing}")
    request_payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    freeze_payload = None
    source_freeze_sha256 = None
    if protocol_freeze_path is not None:
        source_freeze = read_json(protocol_freeze_path)
        source_freeze_sha256 = file_sha256(protocol_freeze_path)
        # Export only the commitments required by the local worker. In
        # particular, omit blind-label hashes, prompt-development IDs, remote
        # provider/pricing details, and every private-file hash.
        worker_freeze = {
            "schema_version": source_freeze["schema_version"],
            "status": source_freeze["status"],
            "protocol_id": source_freeze["protocol_id"],
            "source_protocol_freeze_sha256": source_freeze_sha256,
            "models": {"local": source_freeze["models"]["local"]},
            "decoding": source_freeze["decoding"],
            "blind_episode_ids": source_freeze["blind_episode_ids"],
            "file_sha256": {
                "prompts/local_v1.txt": source_freeze["file_sha256"][
                    "prompts/local_v1.txt"
                ]
            },
        }
        freeze_payload = (
            json.dumps(worker_freeze, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
    included_files = [*SAFE_WORKER_FILES, "data/requests.jsonl", "GPU_README.md"]
    if freeze_payload is not None:
        included_files.append("protocol-freeze.json")
    manifest = {
        "schema_version": "flowgate-gpu-bundle/1.0",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request_count": len(rows),
        "requests_sha256": "sha256:" + hashlib.sha256(request_payload).hexdigest(),
        "label_blind": True,
        "included_files": included_files,
        "protocol_freeze_sha256": (
            "sha256:" + hashlib.sha256(freeze_payload).hexdigest()
            if freeze_payload is not None
            else None
        ),
        "source_protocol_freeze_sha256": source_freeze_sha256,
        "explicitly_omitted": [
            "data/generated/private",
            "fixtures.py",
            "trusted labels",
            "raw episodes",
            "pseudonym map",
            "remote prompt and API credentials",
        ],
        "file_sha256": {
            relative: file_sha256(root / relative) for relative in SAFE_WORKER_FILES
        },
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GPU bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative in SAFE_WORKER_FILES:
            archive.add(root / relative, arcname=relative, recursive=False)
        _add_bytes(archive, "data/requests.jsonl", request_payload)
        _add_bytes(archive, "GPU_README.md", GPU_README.encode("utf-8"))
        if freeze_payload is not None:
            _add_bytes(archive, "protocol-freeze.json", freeze_payload)
        _add_bytes(
            archive,
            "bundle-manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    return manifest
