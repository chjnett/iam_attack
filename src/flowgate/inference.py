from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import RESPONSE_SCHEMA_VERSION, validate_request


class Backend(Protocol):
    model_id: str
    last_metadata: dict[str, Any]
    last_raw_response: str | None

    def predict(self, request: dict[str, Any], prompt: str) -> dict[str, Any]:
        ...


@dataclass
class MockBackend:
    """Deterministic plumbing backend; never valid as a research baseline."""

    model_id: str = "mock-smoke-only"
    remote: bool = False
    last_metadata: dict[str, Any] = field(default_factory=dict, init=False)
    last_raw_response: str | None = field(default=None, init=False)

    def predict(self, request: dict[str, Any], prompt: str) -> dict[str, Any]:
        validate_request(request)
        self.last_raw_response = None
        self.last_metadata = {"latency_ms": 1, "input_tokens": 0, "output_tokens": 0}
        unknowns = request.get("unknown_preconditions", [])
        features = request.get("feature_summary", {})
        event_ids = [
            str(item["event_id"])
            for item in request.get("observed_events", [])
            if isinstance(item, dict) and "event_id" in item
        ]
        if unknowns or request["capability_state"] == "unknown":
            verdict = "abstain"
            probability = 0.5
            assessment = "insufficient"
            citations: list[str] = []
            signals = ["incomplete_observation", "missing_precondition"]
            uncertainty = ["unknown_precondition"]
        elif bool(request.get("capability_gain")) and bool(
            features.get("sensitive_action_observed", False)
        ):
            verdict = "attack"
            probability = 0.82 if self.remote else 0.68
            assessment = "supports_attack"
            citations = event_ids[:3]
            signals = ["capability_gain", "sensitive_action_after_gain"]
            uncertainty = ["none"]
        else:
            verdict = "benign"
            probability = 0.18 if self.remote else 0.32
            assessment = "supports_benign_explanation"
            citations = event_ids[:3]
            signals = ["matched_administrative_pattern"]
            uncertainty = ["none"]
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "episode_id": request["episode_id"],
            "witness_digest": request["witness_digest"],
            "family": request["family"],
            "verdict": verdict,
            "malicious_probability": probability,
            "candidate_path_assessment": assessment,
            "cited_event_ids": citations,
            "signals": signals,
            "uncertainty_reasons": uncertainty,
            "summary": "Mock output for transport and contract validation only.",
        }


@dataclass
class OpenAICompatibleBackend:
    """Minimal client for local vLLM or a remote OpenAI-compatible endpoint."""

    base_url: str
    model_id: str
    api_key_env: str
    timeout_seconds: int = 120
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 700
    last_metadata: dict[str, Any] = field(default_factory=dict, init=False)
    last_raw_response: str | None = field(default=None, init=False)

    def predict(self, request: dict[str, Any], prompt: str) -> dict[str, Any]:
        validate_request(request)
        self.last_metadata = {}
        self.last_raw_response = None
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing environment variable: {self.api_key_env}")
        system_prompt = prompt.replace("{{WORKER_INPUT_JSON}}", "")
        body = {
            "model": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False, sort_keys=True),
                },
            ],
        }
        wire = json.dumps(body).encode("utf-8")
        http_request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=wire,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw_http = response.read().decode("utf-8")
                self.last_raw_response = raw_http
                payload = json.loads(raw_http)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_raw_response = detail
            raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail[:500]}") from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        content = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        self.last_metadata = {
            "latency_ms": elapsed_ms,
            "input_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
            "output_tokens": (
                int(completion_tokens) if completion_tokens is not None else None
            ),
            "token_count_method": "provider_usage" if isinstance(usage, dict) else "unavailable",
        }
        self.last_raw_response = str(content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("model response must be a JSON object")
        return parsed
