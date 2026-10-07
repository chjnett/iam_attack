from __future__ import annotations

import copy
import unittest

from flowgate.contracts import make_request
from flowgate.inference import MockBackend
from flowgate.pareto import (
    _mark_frontier,
    analyze_pareto,
    pareto_csv_rows,
    static_witness_only_decision,
)
from tests.test_contracts import witness


def set_verdict(output: dict, verdict: str) -> dict:
    value = copy.deepcopy(output)
    value["verdict"] = verdict
    if verdict == "attack":
        value.update(
            malicious_probability=0.9,
            candidate_path_assessment="supports_attack",
            cited_event_ids=["E001"],
            signals=["capability_gain", "sensitive_action_after_gain"],
            uncertainty_reasons=["none"],
        )
    elif verdict == "benign":
        value.update(
            malicious_probability=0.1,
            candidate_path_assessment="supports_benign_explanation",
            cited_event_ids=["E001"],
            signals=["matched_administrative_pattern"],
            uncertainty_reasons=["none"],
        )
    else:
        value.update(
            malicious_probability=0.5,
            candidate_path_assessment="insufficient",
            cited_event_ids=[],
            signals=["incomplete_observation", "missing_precondition"],
            uncertainty_reasons=["unknown_precondition"],
        )
    return value


class ParetoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = []
        self.labels = []
        self.local = []
        self.remote = []
        self.records = []
        truth = [1, 0, 1, 0]
        local_verdicts = ["benign", "benign", "attack", "attack"]
        remote_verdicts = ["attack", "benign", "attack", "benign"]
        for i in range(4):
            item = witness()
            if i == 3:
                item["capability_state"] = "unknown"
                item["capability_gain"] = None
                item["unknown_preconditions"] = ["missing_snapshot"]
                item["feature_summary"]["unknown_precondition_count"] = 1
            request = make_request(
                case_id=f"FG-P{i:03d}",
                cutoff_time="2026-01-01T00:00:00Z",
                witness=item,
            )
            local = set_verdict(MockBackend().predict(request, "test"), local_verdicts[i])
            remote = set_verdict(MockBackend(remote=True).predict(request, "test"), remote_verdicts[i])
            # The last remote response intentionally abstains to verify that
            # complete output files can still have incomplete decision coverage.
            if i == 3:
                remote = set_verdict(remote, "abstain")
            self.requests.append(request)
            self.labels.append({
                "episode_id": request["episode_id"],
                "split": "blind",
                "malicious": bool(truth[i]),
                "family": request["family"],
            })
            self.local.append(local)
            self.remote.append(remote)
            self.records.append({
                "schema_version": "1.0",
                "run_id": "remote-test",
                "episode_id": request["episode_id"],
                "model_role": "remote",
                "model_id": "mock-remote",
                "prompt_version": "remote_v1",
                "prompt_sha256": "a" * 64,
                "witness_digest": request["witness_digest"],
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:01+00:00",
                "output": remote,
                "latency_ms": 100 + i,
                "input_tokens": 100,
                "output_tokens": 20,
                "cost": {
                    "amount": 0.25,
                    "currency": "USD",
                    "pricing_snapshot": "2026-01-01",
                },
                "validation": {
                    "schema_valid": True,
                    "binding_valid": True,
                    "citation_valid": True,
                    "errors": [],
                },
            })

    def test_sweeps_every_exact_budget_and_includes_endpoints(self) -> None:
        report = analyze_pareto(self.requests, self.labels, self.local, self.remote, self.records)
        self.assertEqual(report["episode_count"], 4)
        for policy in ("random", "uncertainty", "flowgate-v0"):
            rows = [row for row in report["rows"] if row["policy"] == policy]
            self.assertEqual([row["budget"] for row in rows], [0, 1, 2, 3, 4])
            self.assertEqual([row["calls"] for row in rows], [0, 1, 2, 3, 4])
        self.assertTrue(any(row["policy"] == "never" for row in report["rows"]))
        self.assertTrue(any(row["policy"] == "always" for row in report["rows"]))

    def test_abstentions_reduce_coverage_and_calls_have_simulated_cost(self) -> None:
        report = analyze_pareto(self.requests, self.labels, self.local, self.remote, self.records)
        never = next(row for row in report["rows"] if row["policy"] == "never")
        always = next(row for row in report["rows"] if row["policy"] == "always")
        self.assertEqual(never["coverage"], 1.0)
        self.assertEqual(always["coverage"], 0.75)
        self.assertAlmostEqual(never["cost"], 0.0)
        self.assertAlmostEqual(always["cost"], 1.0)
        self.assertEqual(always["latency"]["remote_selected"]["count"], 4)
        self.assertEqual(len(pareto_csv_rows(report)), len(report["rows"]))

    def test_remote_output_binding_and_cost_are_fail_closed(self) -> None:
        changed = copy.deepcopy(self.records)
        changed[0]["output"]["witness_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "witness_digest|output does not match"):
            analyze_pareto(self.requests, self.labels, self.local, self.remote, changed)

        changed = copy.deepcopy(self.records)
        changed[0]["cost"]["amount"] = None
        with self.assertRaisesRegex(ValueError, "cost"):
            analyze_pareto(self.requests, self.labels, self.local, self.remote, changed)

    def test_duplicate_or_incomplete_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            analyze_pareto(self.requests + [self.requests[0]], self.labels, self.local, self.remote, self.records)
        with self.assertRaisesRegex(ValueError, "match request IDs exactly"):
            analyze_pareto(self.requests, self.labels, self.local[:-1], self.remote, self.records)

    def test_frontier_does_not_trade_away_coverage_for_cost_and_mcc(self) -> None:
        cheap_low_coverage = {
            "cost": 0.0,
            "mcc": 1.0,
            "fnr": 0.0,
            "coverage": 0.5,
            "missed_rescue_risk": 0.0,
        }
        fully_covered = {
            "cost": 1.0,
            "mcc": 0.8,
            "fnr": 0.0,
            "coverage": 1.0,
            "missed_rescue_risk": 0.0,
        }

        rows = _mark_frontier([cheap_low_coverage, fully_covered])

        self.assertTrue(rows[0]["pareto_optimal"])
        self.assertTrue(rows[1]["pareto_optimal"])

    def test_static_witness_only_rule_requires_all_three_positive_facts(self) -> None:
        cases = [
            ("verified", True, True, "attack"),
            ("verified", False, True, "benign"),
            ("verified", True, False, "benign"),
            ("possible", True, True, "benign"),
            ("unknown", None, True, "abstain"),
        ]
        for i, (state, gain, sensitive, expected) in enumerate(cases):
            item = witness()
            item["capability_state"] = state
            item["capability_gain"] = gain
            item["feature_summary"]["sensitive_action_observed"] = sensitive
            if state == "unknown":
                item["unknown_preconditions"] = ["missing_snapshot"]
                item["feature_summary"]["unknown_precondition_count"] = 1
            request = make_request(
                case_id=f"FG-RULE{i:03d}",
                cutoff_time="2026-01-01T00:00:00Z",
                witness=item,
            )
            with self.subTest(state=state, gain=gain, sensitive=sensitive):
                self.assertEqual(static_witness_only_decision(request), expected)

    def test_static_baseline_is_descriptive_and_outside_router_frontier(self) -> None:
        report = analyze_pareto(self.requests, self.labels, self.local, self.remote, self.records)
        baseline = report["descriptive_baselines"]["static_witness_only"]

        self.assertEqual(
            baseline["decision_counts"],
            {"attack": 3, "benign": 0, "abstain": 1},
        )
        self.assertEqual(baseline["coverage"], 0.75)
        self.assertEqual(baseline["covered_metrics"]["n"], 3)
        self.assertEqual(baseline["covered_metrics"]["tp"], 2)
        self.assertEqual(baseline["covered_metrics"]["fp"], 1)
        self.assertFalse(baseline["included_in_cost_mcc_frontier"])
        self.assertNotIn("cost", baseline)
        self.assertNotIn("latency", baseline)
        self.assertFalse(any(row.get("policy") == "static_witness_only" for row in report["rows"]))
        self.assertFalse(any(row.get("policy") == "static_witness_only" for row in report["frontier"]))
        self.assertEqual(len(pareto_csv_rows(report)), len(report["rows"]))


if __name__ == "__main__":
    unittest.main()
