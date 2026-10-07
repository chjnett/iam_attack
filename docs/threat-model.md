# FlowGate pilot threat model

## Scope

This threat model covers the 24-case dry run identified by `flowgate-pilot-24-v1`. It governs how raw AWS-like episode data, trusted labels, compact temporal permission witnesses, model prompts, and model outputs move between the researcher's laptop, a GPU worker, and a remote model API.

It does not claim that the pilot protects a production AWS environment or that the models can safely trigger an automated response.

## Security objectives

1. **Confidentiality:** raw account identifiers, ARNs, principal and resource names, policy documents, credentials, secrets, intent labels, and expected paths do not leave the trusted laptop.
2. **Evaluation integrity:** blind labels and future events cannot influence prompts, model inputs, routing decisions, or output repair.
3. **Evidence integrity:** every inference result is bound to an exact episode and witness digest, and every cited event exists in the submitted prefix.
4. **Reproducibility:** prompts, models, decoding parameters, sanitizer, witness builder, and pricing metadata are versioned before blind-label unsealing.
5. **Fail-safe behavior:** missing state, missing events, contradictory evidence, and unsupported patterns remain unknown or abstained rather than being silently asserted as verified.
6. **Outcome sealing:** all 18 local outputs, all 18 independent remote outputs, failures, run records, routing decisions, merged predictions, and run manifests are hash-committed before a blind label may be opened.

## Trust zones

### Trusted laptop

The laptop is the only trusted plane. It may hold:

- raw CloudTrail-like events and IAM state;
- pseudonymization maps;
- the canonical `malicious` boolean label and separately held evaluator expectations;
- prompt-development and blind split membership;
- remote API credentials;
- prompt and manifest hashes;
- the evaluator and blind-label unsealing logic.

The laptop constructs the canonical witness, applies allowlist sanitization, validates the worker payload, verifies output bindings, and performs evaluation.

### Sanitized GPU worker

The GPU worker is trusted to execute the selected local model but is **not trusted with raw episode data or labels**. It may receive only an object valid under `schemas/worker-input.schema.json`.

The worker must not receive:

- split membership;
- labels or expected verdicts;
- raw account IDs, ARNs, names, policies, credentials, or secrets;
- events after the declared cutoff;
- the remote model's output.

### Remote model API

The remote provider is treated as an external, honest-but-curious processor. It receives the same sanitized worker input independently and never receives the local answer, trusted labels, pseudonym map, or raw evidence.

The selected provider, model snapshot, retention setting, region when available, and pricing snapshot must be recorded before execution. Provider-side retention and training controls are residual external dependencies and must not be assumed from memory.

## Protected assets

| Asset | Primary risk | Required control |
|---|---|---|
| Raw event/state data | disclosure | laptop only; no raw export |
| Pseudonym map | re-identification | laptop only; separate storage |
| Intent labels | evaluation leakage | sealed until frozen blind run |
| Expected paths/events | evaluation leakage | trusted evaluator only |
| API credential | account compromise | environment/secret store; never committed |
| Prompt and model configuration | undisclosed tuning | content hash and immutable run manifest |
| Witness | re-identification or hidden label | allowlist schema, pseudonymization, forbidden-key scan |
| Model output | fabricated evidence | strict schema, digest binding, citation validation |

## Adversaries and failure sources

### A1. Embedded prompt injection

An actor, resource name, user-agent string, or event detail may contain text such as “ignore prior instructions” or fake JSON fields. Models must treat every input value as evidence, not an instruction. The sanitizer permits structured scalar/list values only; prompts explicitly reject embedded instructions.

### A2. Honest-but-curious compute provider

The GPU host or remote API operator may retain or inspect requests. Only pseudonymized, label-blind, secret-free worker input may be transmitted. Sanitization is mandatory even for synthetic fixtures so the boundary remains testable.

### A3. Accidental researcher leakage

The researcher may tune a prompt or threshold after viewing blind outcomes. Blind labels remain in a separate laptop-only artifact. Prompt hashes, model IDs, decoding settings, sanitizer version, routing rule, and episode manifest are frozen before unsealing. Any post-unseal change requires a new protocol version and new blind set.

### A4. Future-event leakage

An episode builder may accidentally include events after the decision point. `prefix_len` and `cutoff` bind the visible prefix; model inputs must contain only `observed_events` at or before the cutoff. Prefix consistency is checked on the laptop.

### A5. Label leakage through features

Fields such as `malicious`, `label`, `ground_truth`, `intent`, `expected_verdict`, `expected_path`, or split membership can trivially reveal the answer. They are absent from the worker schema. Family is retained because each family contains matched attack and benign cases in both splits; it must never be used as an intent label.

### A6. Hallucinated evidence

A model may invent an event or path. Outputs can cite only event IDs present in `observed_events`; the witness contains one candidate path produced by the trusted evidence layer. Digest, episode, family, and citation bindings are checked after inference. Invalid output is recorded as invalid rather than silently repaired into a correct answer.

### A7. Incomplete or stale state

CloudTrail-like events and snapshots may omit a precondition. `capability_state`, nullable `capability_gain`, `unknown_preconditions`, and witness completeness make missingness explicit. Unknown evidence cannot be upgraded to verified by a model. The appropriate terminal action is abstention or analyst review

### A8. Output retry bias

Repeatedly calling a model until it emits a convenient answer biases results and spends budget selectively. The protocol retains first attempts, parsing failures, timeouts, and retries. A retry may repair transport or syntax only under a predeclared policy; it cannot use labels or altered instructions.

## Sanitization requirements

Before an object can validate as worker input, the laptop must:

1. replace account, principal, role, session, resource, and policy names with stable episode-local pseudonyms;
2. remove raw ARNs, account numbers, credential material, policy documents, free-form request bodies, labels, split, and expected evidence;
3. retain only allowlisted event fields and scalar/list `details` values;
4. normalize timestamps while preserving order and cutoff semantics;
5. remove trusted-only fields, then compute a worker-input digest over the exact exported evidence;
6. reject rather than redact silently if a forbidden field or secret pattern survives;
7. validate the resulting payload against `worker-input.schema.json`.

The canonical witness includes `split` for trusted bookkeeping. The worker view removes it, discards the private digest, and computes a new digest over the exact exported object (excluding the digest field itself). Model outputs bind to this worker digest.

## Evaluation-integrity rules

- Only six `prompt_dev` cases may be inspected during prompt development.
- Blind outputs are collected once with frozen prompts and model settings.
- The remote model is run independently of the local result exactly once on all
  18 blind cases. This remote-all pass is the fixed counterfactual used to replay
  routing policies without making label-dependent API calls.
- A successful output seal must commit both complete model passes, their first-attempt
  errors and records, the exact four-case `flowgate-v0` decision, and deterministic
  merged predictions before blind labels are opened. If complete first-attempt
  outputs cannot be sealed, the complementarity/Pareto result is inconclusive.
- Blind labels are joined only in the trusted evaluator.
- Abstentions, invalid JSON, timeouts, and schema failures remain visible.
- Oracle routing is calculated after unsealing and is explicitly non-deployable.
- The 24 balanced cases do not estimate production prevalence or operational false-positive rate.

## Residual risks

- Pseudonyms and action sequences may still reveal that the data concerns a particular AWS workflow.
- Provider retention and training behavior remain outside the local enforcement boundary.
- A valid schema cannot prove that a witness is semantically correct.
- Matched synthetic or deterministic fixtures may be easier than real organizational logs.
- Six prompt-development cases and 18 blind cases are too small for a statistical risk guarantee.
- Model self-reported probabilities may be uncalibrated.
- Unknown state can cause true attacks to be abstained or missed if abstentions are not reviewed.

These risks must be reported as limitations. Passing the dry-run gates authorizes only an expanded pilot, not deployment or a SOTA claim.
