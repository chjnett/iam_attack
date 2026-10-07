# FlowGate 24-case dry run

Korean first-run checklist: [`START_HERE_KO.md`](START_HERE_KO.md)

Step-by-step Korean execution guide: [`NEXT_STEPS_KO.md`](NEXT_STEPS_KO.md)

FlowGate is a proposed, verifier-grounded deferment system for AWS IAM privilege-change episode triage. It converts an episode into a compact temporal permission witness and asks whether a remote LLM is likely to correct a local model's error rather than merely reconsider an uncertain answer.

This directory is a **research protocol scaffold, not a results package**. No accuracy, cost saving, risk guarantee, or SOTA claim has been established. With only 24 cases, the dry run can validate contracts, privacy boundaries, evidence reconstruction, and local/remote complementarity; it cannot establish publication-grade performance.

## Frozen dry-run scope

| Item | Protocol |
|---|---|
| Cases | 24 executable or deterministic fixture episodes |
| Families | `policy_attachment_abuse`, `role_trust_abuse`, `passrole_compute_abuse` |
| Intent labels | 12 `malicious: true` (attack), 12 `malicious: false` (matched benign) |
| Prompt-development split | 6 cases: 2 per family, one attack and one benign |
| Blind split | 18 cases: 6 per family, three attack and three benign |
| Local inference | One fixed 7–8B-class model on the sanitized GPU worker |
| Remote inference | One fixed remote model, independently given the same sanitized worker input |
| Primary purpose | Decide whether a larger, preregistered pilot is justified |

The matched benign cases intentionally use the same family and similar API sequence as their attack counterpart. A family name is therefore **not** an intent label.

## One-sentence research claim to test

> FlowGate converts AWS privilege-change episodes into verifiable temporal permission witnesses and uses them to defer only cases in which a remote LLM is expected to rescue the local model, minimizing API use subject to a local-accept miss-risk target.

The 24-case dry run tests whether this claim is plausible enough to warrant a larger study. It does not test the miss-risk bound itself.

## Trust boundary

```text
TRUSTED LAPTOP
raw CloudTrail / IAM state / pseudonym map / labels / API credentials
        |
        | deterministic witness construction + allowlist sanitization
        v
SANITIZED WORKER INPUT
no label, no split, no raw account ID, ARN, secret, policy document, or credential
        |                              |
        v                              v
GPU LOCAL MODEL                 REMOTE MODEL API
        |                              |
        +---------- JSON only --------+
                       |
                       v
TRUSTED LAPTOP EVALUATOR
schema validation, digest binding, blind-label unsealing, rescue/harm analysis
```

The remote model must not see the local answer. Independent judgments are required to measure rescue and harm without anchoring.

## Quick Start

This is a two-stage protocol, not a single 24-case run: develop only on six
`prompt_dev` cases, freeze the complete protocol, and then run the 18 blind cases
once. Keep the physical boundary explicit:

- The **trusted laptop** alone holds `data/generated/private/`, all labels, raw episodes, the pseudonym map, `prompts/remote_v1.txt`, remote provider settings and credentials, remote outputs, seals, and evaluation results.
- The **GPU worker** receives only a sanitized `bundle-gpu` archive. It runs the local model and never receives a label, remote prompt, remote credential, or remote output.
- Return exactly four GPU artifacts: `local_responses.jsonl`, `local_run_records.jsonl`, `local_errors.jsonl`, and the automatically created `local_responses.jsonl.manifest.json`.

Run setup from this project directory on the trusted laptop:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mkdir -p work/prompt_dev results/prompt_dev results/blind dist
flowgate-pilot generate --out-dir data/generated
```

Define remote metadata only on the trusted laptop. The recommended reproducible
default below is the immutable GPT-5.4 Mini snapshot. Its official model page
lists Chat Completions and structured-output support, with standard text-token
prices of USD 0.75/M input and USD 4.50/M output as checked on 2026-10-07. The
retention value is deliberately left blank because it must describe the controls
actually enabled for your API organization/project, not a generic provider
promise. Recheck the linked official page immediately before freezing and never
copy these variables to the GPU worker:

```bash
export FLOWGATE_REMOTE_BASE_URL='https://api.openai.com/v1'
export FLOWGATE_REMOTE_MODEL='gpt-5.4-mini-2026-03-17'
export FLOWGATE_REMOTE_REVISION='gpt-5.4-mini-2026-03-17'
export FLOWGATE_REMOTE_PROVIDER='OpenAI'
export FLOWGATE_PROVIDER_RETENTION='<record-the-actual-project-retention-control>'
export FLOWGATE_PRICING_SNAPSHOT='https://developers.openai.com/api/docs/models/gpt-5.4-mini accessed 2026-10-07'
export FLOWGATE_INPUT_PRICE_PER_MILLION='0.75'
export FLOWGATE_OUTPUT_PRICE_PER_MILLION='4.50'
export FLOWGATE_CURRENCY='USD'
export FLOWGATE_REMOTE_KEY='<secret-kept-only-on-this-laptop>'
```

Official source: [GPT-5.4 Mini model and pricing](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
At the frozen 700-token output cap, 18 uncached calls can consume at most USD
0.0567 in output tokens; input and any regional-processing uplift are additional.
This is a planning bound, not a bill prediction—the run records use provider
usage and the Pareto report separates the one-time remote-all measurement cost
from simulated deployment cost at each routing budget.

### Stage 1 — six `prompt_dev` cases

Export the six sanitized development requests and build their GPU archive on the
trusted laptop:

```bash
flowgate-pilot export-split \
  --requests data/generated/gpu/requests.jsonl \
  --labels data/generated/private/prompt_dev_labels.jsonl \
  --split prompt_dev \
  --out work/prompt_dev/requests.jsonl \
  --manifest-out work/prompt_dev/requests.manifest.json

flowgate-pilot bundle-gpu \
  --requests data/generated/gpu/requests.jsonl \
  --labels data/generated/private/prompt_dev_labels.jsonl \
  --split prompt_dev \
  --out dist/flowgate-prompt-dev-gpu-v2.tar.gz
```

Copy only the archive to the RTX 3090 worker. The worker's `GPU_README.md`
contains the same commands; the local server is the official
[Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ)
checkpoint through the current [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/):

```bash
uv venv --python 3.12 --seed .venv-vllm
source .venv-vllm/bin/activate
uv pip install 'vllm==0.31.0' --torch-backend=auto

export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key
export VLLM_USE_FLASHINFER_SAMPLER=0

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

In a second GPU shell, run the development batch with its required phase:

```bash
source .venv-vllm/bin/activate
export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p results

PYTHONPATH=src python3 -m flowgate.cli run-batch \
  --requests data/requests.jsonl \
  --prompt prompts/local_v1.txt \
  --out results/local_responses.jsonl \
  --records results/local_run_records.jsonl \
  --errors results/local_errors.jsonl \
  --run-id prompt-dev-local-v1 \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id "$FLOWGATE_LOCAL_MODEL" \
  --model-revision "$FLOWGATE_LOCAL_REVISION" \
  --quantization awq \
  --api-key-env FLOWGATE_LOCAL_KEY \
  --phase prompt_dev
```

Return the four local artifacts to `results/prompt_dev/` on the trusted laptop.
If the remote prompt also needs development, call it only from the trusted
laptop on `work/prompt_dev/requests.jsonl`, with `--phase prompt_dev`. The six
development labels may be opened there. Use new versioned paths for every prompt
iteration; never overwrite a first-attempt artifact, and never include these six
cases in blind metrics.

### Freeze the protocol

After Stage 1, stop editing prompts, schemas, code, model IDs/revisions, decoding,
routing, retention, and pricing. Create the commitment on the trusted laptop:

```bash
export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641

flowgate-pilot freeze \
  --out results/protocol-freeze.json \
  --local-model-id "$FLOWGATE_LOCAL_MODEL" \
  --local-model-revision "$FLOWGATE_LOCAL_REVISION" \
  --remote-model-id "$FLOWGATE_REMOTE_MODEL" \
  --remote-model-revision "$FLOWGATE_REMOTE_REVISION" \
  --remote-provider "$FLOWGATE_REMOTE_PROVIDER" \
  --provider-retention "$FLOWGATE_PROVIDER_RETENTION" \
  --pricing-snapshot "$FLOWGATE_PRICING_SNAPSHOT" \
  --input-price-per-million "$FLOWGATE_INPUT_PRICE_PER_MILLION" \
  --output-price-per-million "$FLOWGATE_OUTPUT_PRICE_PER_MILLION" \
  --currency "$FLOWGATE_CURRENCY"
```

The freeze hashes the blind-label file but does not authorize opening it.

### Stage 2 — export and run all 18 blind cases once

Only after the freeze succeeds, export the committed blind requests and build the
GPU bundle. The bundle automatically embeds `protocol-freeze.json`; no blind
label file is used or copied in either command:

```bash
flowgate-pilot export-split \
  --requests data/generated/gpu/requests.jsonl \
  --split blind \
  --freeze-manifest results/protocol-freeze.json \
  --out results/blind/requests.jsonl \
  --manifest-out results/blind/requests.manifest.json

flowgate-pilot bundle-gpu \
  --requests data/generated/gpu/requests.jsonl \
  --split blind \
  --freeze-manifest results/protocol-freeze.json \
  --out dist/flowgate-blind-gpu.tar.gz
```

Copy only the blind archive to the GPU worker. Inside its extracted directory,
capture the runtime and run all 18 local requests with the embedded freeze and
the exact frozen server settings:

```bash
source .venv-vllm/bin/activate
export FLOWGATE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
export FLOWGATE_LOCAL_REVISION=b25037543e9394b818fdfca67ab2a00ecc7dd641
export FLOWGATE_LOCAL_KEY=local-dev-key
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p results

python3 scripts/capture_runtime.py > runtime.json

PYTHONPATH=src python3 -m flowgate.cli run-batch \
  --requests data/requests.jsonl \
  --prompt prompts/local_v1.txt \
  --out results/local_responses.jsonl \
  --records results/local_run_records.jsonl \
  --errors results/local_errors.jsonl \
  --run-id blind-local-v1 \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id "$FLOWGATE_LOCAL_MODEL" \
  --model-revision "$FLOWGATE_LOCAL_REVISION" \
  --quantization awq \
  --api-key-env FLOWGATE_LOCAL_KEY \
  --phase blind \
  --freeze-manifest protocol-freeze.json \
  --runtime-metadata runtime.json \
  --server-max-model-len 8192 \
  --server-gpu-memory-utilization 0.85
```

The automatic local run manifest embeds `runtime.json`; do not treat the source
snapshot as a fifth return artifact. Return the four local artifacts and place
them under `results/blind/` without renaming their basenames.

Next, still without opening blind labels, invoke the frozen remote model from the
trusted laptop on the **same complete 18-request file**. This is one independent
counterfactual call per case, exactly once; it is not a call on the later
four-request selection and it receives no local answer:

```bash
flowgate-pilot run-batch \
  --requests results/blind/requests.jsonl \
  --prompt prompts/remote_v1.txt \
  --out results/blind/remote_responses.jsonl \
  --records results/blind/remote_run_records.jsonl \
  --errors results/blind/remote_errors.jsonl \
  --run-id blind-remote-all-v1 \
  --backend openai \
  --base-url "$FLOWGATE_REMOTE_BASE_URL" \
  --model-id "$FLOWGATE_REMOTE_MODEL" \
  --model-revision "$FLOWGATE_REMOTE_REVISION" \
  --api-key-env FLOWGATE_REMOTE_KEY \
  --remote \
  --phase blind \
  --freeze-manifest results/protocol-freeze.json \
  --input-price-per-million "$FLOWGATE_INPUT_PRICE_PER_MILLION" \
  --output-price-per-million "$FLOWGATE_OUTPUT_PRICE_PER_MILLION" \
  --currency "$FLOWGATE_CURRENCY" \
  --pricing-snapshot "$FLOWGATE_PRICING_SNAPSHOT"
```

Do not retry a failed blind case or use `--force`; retain the error and mark the
run inconclusive. Build local and remote engineering statistics separately:

```bash
flowgate-pilot stats \
  --requests results/blind/requests.jsonl \
  --responses results/blind/local_responses.jsonl \
  --errors results/blind/local_errors.jsonl \
  --out results/blind/local_stats.json

flowgate-pilot stats \
  --requests results/blind/requests.jsonl \
  --responses results/blind/remote_responses.jsonl \
  --errors results/blind/remote_errors.jsonl \
  --out results/blind/remote_stats.json
```

Derive the frozen `flowgate-v0` decision for exactly four cases from local output
only. `selected_remote_requests.jsonl` is an audit artifact; do not use it to make
another remote call. Merge those decisions with the already captured remote-all
counterfactual:

```bash
flowgate-pilot select \
  --requests results/blind/requests.jsonl \
  --local-responses results/blind/local_responses.jsonl \
  --decisions-out results/blind/routing_decisions.jsonl \
  --selected-out results/blind/selected_remote_requests.jsonl \
  --policy flowgate-v0 \
  --budget-fraction 0.25 \
  --call-budget 4 \
  --seed 7

flowgate-pilot merge \
  --requests results/blind/requests.jsonl \
  --local-responses results/blind/local_responses.jsonl \
  --decisions results/blind/routing_decisions.jsonl \
  --remote-responses results/blind/remote_responses.jsonl \
  --out results/blind/predictions.jsonl
```

Seal every blind artifact before any blind label is opened:

```bash
flowgate-pilot seal-outputs \
  --freeze-manifest results/protocol-freeze.json \
  --requests results/blind/requests.jsonl \
  --local-responses results/blind/local_responses.jsonl \
  --local-errors results/blind/local_errors.jsonl \
  --local-records results/blind/local_run_records.jsonl \
  --local-manifest results/blind/local_responses.jsonl.manifest.json \
  --decisions results/blind/routing_decisions.jsonl \
  --remote-responses results/blind/remote_responses.jsonl \
  --remote-errors results/blind/remote_errors.jsonl \
  --remote-records results/blind/remote_run_records.jsonl \
  --remote-manifest results/blind/remote_responses.jsonl.manifest.json \
  --predictions results/blind/predictions.jsonl \
  --out results/blind/output-seal.json
```

Only after `seal-outputs` succeeds may the trusted laptop open
`data/generated/private/blind_labels.jsonl`. Evaluation consumes the separate
local/remote stats, and comparison replays the sealed outputs across budgets:

```bash
flowgate-pilot evaluate \
  --labels data/generated/private/blind_labels.jsonl \
  --predictions results/blind/predictions.jsonl \
  --corpus-manifest data/generated/manifest.json \
  --local-stats results/blind/local_stats.json \
  --remote-stats results/blind/remote_stats.json \
  --freeze-manifest results/protocol-freeze.json \
  --output-seal results/blind/output-seal.json \
  --out results/blind/evaluation.json

flowgate-pilot compare \
  --freeze-manifest results/protocol-freeze.json \
  --output-seal results/blind/output-seal.json \
  --labels data/generated/private/blind_labels.jsonl \
  --out-json results/blind/pareto.json \
  --out-csv results/blind/pareto.csv
```

Any blind-label access before the seal, any retry, or any post-freeze change
invalidates this blind run and requires a new protocol version and blind set.

## Artifact layout

```text
configs/pilot.json                 frozen case allocation, controls, and engineering gates
schemas/witness.schema.json        canonical label-blind witness retained on the laptop
schemas/worker-input.schema.json   exact sanitized payload allowed off the laptop
schemas/model-output.schema.json   exact JSON contract for both models
schemas/run-record.schema.json     reproducibility record binding prompt, witness, and output
schemas/trusted-labels.schema.json laptop-only minimal ground-truth contract
prompts/local_v1.txt               frozen local-model prompt
prompts/remote_v1.txt              frozen independent remote-model prompt
docs/threat-model.md               assets, trust boundaries, threats, and residual risks
```

## Canonical data flow

1. The laptop creates 24 episodes and stores intent labels separately.
2. `build_witness(...)` produces a label-blind evidence object with trusted bookkeeping. Its private digest covers that exact trusted object except `witness_digest`.
3. The sanitizer removes `split`, rejects forbidden fields and raw identifiers, then computes a new lowercase SHA-256 over the exact worker input excluding `witness_digest`. Only this worker-bound digest leaves the laptop.
4. The local and remote prompts receive the same worker input independently.
5. Their responses must validate under `model-output.schema.json`. Markdown, prose outside the JSON object, invented event IDs, and digest mismatches are invalid outputs.
6. A trusted run record binds the model ID, prompt hash, witness digest, token counts, cost metadata, and validated output.
7. Blind labels are unsealed only after prompt files, routing rules, model IDs, and the run manifest have been frozen and hashed.

## Exact model decision semantics

- `attack`: the observed sequence and supplied candidate path support malicious privilege escalation or abuse.
- `benign`: the same type of capability change is better explained by the matched administrative workflow.
- `abstain`: the supplied evidence is insufficient or contradictory. Abstention is measured as uncovered analyst workload, not silently counted as correct.
- `malicious_probability`: the model's estimate in `[0, 1]`; it is not treated as calibrated without a separate calibration study.
- `candidate_path_assessment`: whether the supplied path supports attack, supports a benign explanation, or is insufficient.
- `cited_event_ids`: only IDs present in `observed_events` are legal.

Models do not invent or recompute IAM reachability. The witness generator owns capability facts; the model judges operational context.

## Leakage controls

The following controls are mandatory:

1. Labels, expected verdicts, and expected evidence never appear in a witness, prompt, worker log, or remote request.
2. The six `prompt_dev` cases are the only cases available for prompt editing. They are excluded from blind metrics.
3. The 18 blind labels remain sealed on the laptop until prompts, model IDs, decoding parameters, routing rules, and manifest hashes are frozen.
4. No blind result may trigger prompt, feature, threshold, family, or case edits. Any such edit creates a new protocol version and requires a new blind set.
5. Both models receive the same prefix and cutoff. Future events are prohibited.
6. Remote output is never exposed to the local model, and local output is never exposed to the remote model.
7. The family allocation is balanced in both splits, but the `attack|benign` label is never transmitted.
8. All outputs and failures are retained; malformed or timed-out outputs cannot be silently rerun with a changed prompt.

## Dry-run measurements

Report counts before aggregate scores:

- valid witnesses and schema-valid worker payloads;
- local and remote valid-output rates;
- local-correct/remote-correct contingency table;
- `rescue = local wrong AND remote correct`;
- `harm = local correct AND remote wrong`;
- abstention count for each model;
- local-only, remote-only, and Oracle-at-4-call correct counts on the 18 blind cases;
- MCC and AUPRC with exact episode counts, marked descriptive;
- raw-to-worker serialized size ratio and, when a tokenizer is fixed, token ratio;
- witness/path disagreements found by the trusted verifier.

Also report a transparent, non-LLM **static/witness-only** baseline. It predicts
`attack` only when the trusted witness simultaneously says
`capability_state=verified`, `capability_gain=true`, and
`sensitive_action_observed=true`; it abstains when capability state is unknown
and otherwise predicts `benign`. Its coverage and covered-case classification
metrics are descriptive and appear in `pareto.json` under
`descriptive_baselines.static_witness_only`. It is not a router, has no LLM API
cost or LLM latency, and is deliberately excluded from the cost–MCC frontier
and `pareto.csv` router rows.

Router Pareto dominance jointly uses lower API cost, higher MCC, lower FNR,
higher decision coverage, and lower missed-rescue risk. Thus a policy cannot
look optimal merely by abstaining on hard cases and scoring only an easy subset.
Latency is reported separately because the dry run does not measure a comparable
end-to-end deployment latency for every replayed policy.

Opaque commercial CSPM/CNAPP/ITDR products are contextual related work, not
experimental baselines in this dry run: their exact rules, model versions, and
replay interfaces are not reproducible here. Do not describe their exclusion as
evidence that FlowGate outperforms commercial systems.

`Oracle@4` is the floor of the frozen 25% budget on 18 blind cases. It may use labels only during offline analysis and is an unattainable ceiling, never a deployable baseline.

## Go / No-Go engineering gates

These are **engineering stop rules**, not statistical significance thresholds or claimed results.

### Gate 1 — contracts and privacy (hard stop)

- exactly 24 unique episodes, three specified families, 6/18 split, and 12/12 intent balance;
- every witness, worker input, model output, label file, and run record validates against its schema;
- at least 95% of first-attempt model outputs parse and validate;
- no forbidden key or raw identifier appears in a worker or remote payload;
- all model outputs bind to the input `episode_id` and `witness_digest`.
- deliberately invalid authorization paths produce zero false passes.

Failure means fix the pipeline before interpreting any model result.

### Gate 2 — evidence reconstruction

- at least 20 of 24 cases produce a valid, prefix-safe witness;
- median worker input is at most 700 tokens (record the tokenizer, or clearly mark the byte-based estimate);
- unknown or partial evidence is surfaced as such rather than coerced into a verified path.

Failure means narrow the supported templates or pivot to an evidence-reconstruction study. Do not proceed to a learned router.

### Gate 3 — model complementarity

On the 18 blind cases:

- the local model must make at least four non-abstaining errors; otherwise routing headroom is inconclusive rather than proven absent;
- there must be at least two rescue cases;
- rescue count must exceed harm count;
- an offline Oracle may use at most four remote calls (25% of 18, rounded down);
- that Oracle must improve MCC over local-only by at least 0.05.

If the local model makes fewer than four errors, do not weaken it artificially. Record the result as inconclusive and design a separately versioned hard-case extension. If the other conditions fail, kill the rescue-routing branch and consider a witness/verifier pivot.

### Gate 4 — decision

- **Continue to an expanded pilot:** Gates 1–3 pass without looking at blind labels during development.
- **Pivot:** witness reconstruction passes but complementarity fails; study compact/verifiable context or symbolic verification instead.
- **Kill:** contract/privacy cannot be enforced, or even Oracle routing lacks useful correction headroom.
- **Inconclusive:** too few local errors or blind outputs to evaluate complementarity; do not claim success and do not begin full implementation.

No risk-controlled or SOTA claim is allowed until a larger, separately split train/calibration/test corpus is evaluated under held-out families or accounts with uncertainty intervals.

## Scope exclusions

This dry run does not implement or claim:

- production streaming, automatic blocking, or incident response;
- complete AWS IAM coverage, cross-account reasoning, or multi-cloud support;
- equivalence to GuardDuty, Wiz, Orca, or other opaque managed products;
- a trained GNN router or formal distribution-shift guarantee;
- real-world prevalence, analyst-time savings, or operational false-positive rates.
