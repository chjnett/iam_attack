#!/usr/bin/env python3
"""Print a secret-free JSON snapshot of the local inference runtime."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from typing import Any


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia() -> list[dict[str, Any]] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "name": parts[0],
                    "uuid": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mib": int(parts[3]),
                }
            )
    return rows


def main() -> int:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "vllm": _version("vllm"),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
        "environment": {
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
                "VLLM_USE_FLASHINFER_SAMPLER"
            ),
        },
        "gpus": _nvidia(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
