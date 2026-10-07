# FlowGate prompt_dev v2 분석

- Source branch: `results/prompt-dev-v2`
- Source commit: `535d5e36a018bf0b07c373c9763ca0f994749e51`
- Local model: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Model revision: `b25037543e9394b818fdfca67ab2a00ecc7dd641`
- vLLM: `0.31.0`
- Phase: `prompt_dev` only

## Contract result

- Requests: 6
- Valid responses: 6
- Run records: 6
- Errors: 0
- Parse/contract success: 100%
- Median worker-input estimate: 420 tokens
- Invalid authorization false passes: 0

## Label-aware development result

| Family | Attack case | Benign case | Decisive coverage |
|---|---|---|---:|
| policy attachment | correct `attack` | correct `benign` | 2/2 |
| role trust | `abstain` | `abstain` | 0/2 |
| PassRole/compute | `abstain` | `abstain` | 0/2 |

- Verdict distribution: attack 1, benign 1, abstain 4
- Decision coverage: 2/6 (33.3%)
- Covered-case accuracy: 2/2 (100%)
- Abstentions: two attack labels and two benign labels; no one-sided label skew

The four abstentions are not equivalent. The two role-trust cases have verified,
substantially complete evidence and are potentially recoverable by a stronger
model. The two PassRole cases contain an unknown initial state and therefore
cannot be repaired by sending the same evidence to a larger model.

## Label-blind routing sanity check

At an illustrative two-call development budget, `flowgate-v0` selects exactly
the two verified role-trust abstentions (score 0.875 each) and rejects the two
irreducibly unknown PassRole abstentions (score -0.270001 each). This is the
intended marginal-rescue behavior and supports proceeding to protocol freeze.

## Decision

**Proceed to freeze after pinning the observed runtime environment.** Do not
change the prompt or routing rule based on blind data. Record
`VLLM_USE_FLASHINFER_SAMPLER=0` as part of the local serving/runtime commitment
before exporting the blind GPU bundle.

This six-case result validates engineering behavior only. It is not a model
accuracy, cost-saving, or SOTA result.
