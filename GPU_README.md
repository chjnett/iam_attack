# FlowGate sanitized RTX 3090 worker

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

vllm serve "$FLOWGATE_LOCAL_MODEL" \
  --revision "$FLOWGATE_LOCAL_REVISION" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
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
if test -f protocol-freeze.json; then
  export FLOWGATE_RUN_ID=blind-local-v1
  python3 scripts/capture_runtime.py > runtime.json
  PHASE_ARGS="--phase blind --freeze-manifest protocol-freeze.json --runtime-metadata runtime.json --server-max-model-len 8192 --server-gpu-memory-utilization 0.85"
else
  export FLOWGATE_RUN_ID=prompt-dev-local-v1
  PHASE_ARGS="--phase prompt_dev"
fi

mkdir -p results
PYTHONPATH=src python3 -m flowgate.cli run-batch \
  --requests data/requests.jsonl \
  --prompt prompts/local_v1.txt \
  --out results/local_responses.jsonl \
  --records results/local_run_records.jsonl \
  --errors results/local_errors.jsonl \
  --run-id "$FLOWGATE_RUN_ID" \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id "$FLOWGATE_LOCAL_MODEL" \
  --model-revision "$FLOWGATE_LOCAL_REVISION" \
  --quantization awq \
  --api-key-env FLOWGATE_LOCAL_KEY \
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
