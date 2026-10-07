from __future__ import annotations

import unittest

from flowgate.evaluate import evaluate_rows
from flowgate.metrics import (
    average_precision,
    classification_metrics,
    complementarity_counts,
    oracle_routing_at_budget,
    pilot_go_no_go,
    routing_metrics,
)


class MetricTests(unittest.TestCase):
    def test_binary_classification_metrics(self) -> None:
        metrics = classification_metrics(
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0.9, 0.4, 0.8, 0.1],
        )
        self.assertEqual((metrics["tp"], metrics["tn"]), (1, 1))
        self.assertEqual((metrics["fp"], metrics["fn"]), (1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["fnr"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)
        self.assertEqual(metrics["mcc"], 0.0)
        self.assertAlmostEqual(float(metrics["average_precision"]), 5 / 6)

    def test_average_precision_groups_tied_scores(self) -> None:
        first = average_precision([1, 0, 1], [0.8, 0.8, 0.2])
        reordered = average_precision([0, 1, 1], [0.8, 0.8, 0.2])
        self.assertAlmostEqual(first, reordered)
        self.assertAlmostEqual(first, 7 / 12)

    def test_complementarity_and_routing_risk(self) -> None:
        truth = [1, 1, 0, 0, 1, 0]
        local = [1, 0, 0, 0, 0, 1]
        remote = [1, 1, 1, 1, 1, 1]
        counts = complementarity_counts(truth, local, remote)
        self.assertEqual(counts["neutral"], 1)
        self.assertEqual(counts["rescue"], 2)
        self.assertEqual(counts["harm"], 2)
        self.assertEqual(counts["both_wrong"], 1)

        result = routing_metrics(
            truth,
            local,
            remote,
            [False, True, False, False, False, False],
        )
        self.assertEqual(result["captured_rescue_count"], 1)
        self.assertEqual(result["missed_rescue_count"], 1)
        self.assertEqual(result["rescue_capture"], 0.5)
        self.assertEqual(result["missed_attack_rescue_count"], 1)
        self.assertAlmostEqual(result["missed_rescue_risk"], 1 / 3)

    def test_oracle_uses_only_helpful_calls_within_budget(self) -> None:
        truth = [1, 1, 0, 0, 1, 0, 1, 0]
        local = [1, 0, 1, 0, 1, 0, 0, 0]
        remote = [1, 1, 0, 0, 0, 1, 1, 1]
        result = oracle_routing_at_budget(
            truth, local, remote, maximum_call_rate=0.25
        )
        self.assertLessEqual(result["remote_call_count"], 2)
        self.assertGreater(result["mcc_gain_over_local"], 0.0)
        self.assertEqual(result["routed_harm_count"], 0)

    def test_exact_engineering_go_no_go_gate(self) -> None:
        passed = pilot_go_no_go(
            episode_count=24,
            witness_success_count=20,
            invalid_false_pass_count=0,
            median_witness_tokens=700,
            parse_success_count=23,
            parse_total_count=24,
            local_errors=4,
            rescue_count=2,
            harm_count=1,
            oracle_call_rate=0.25,
            oracle_mcc_gain=0.05,
        )
        self.assertEqual(passed["decision"], "CONTINUE")

        failed = pilot_go_no_go(
            episode_count=24,
            witness_success_count=19,
            invalid_false_pass_count=1,
            median_witness_tokens=701,
            parse_success_count=22,
            parse_total_count=24,
            local_errors=3,
            rescue_count=1,
            harm_count=1,
            oracle_call_rate=0.30,
            oracle_mcc_gain=0.04,
        )
        self.assertEqual(failed["decision"], "KILL")
        self.assertIn("witness_success", failed["failed_checks"])
        self.assertIn("net_rescue", failed["failed_checks"])

        inconclusive = pilot_go_no_go(
            episode_count=24,
            witness_success_count=24,
            invalid_false_pass_count=0,
            median_witness_tokens=300,
            parse_success_count=24,
            parse_total_count=24,
            local_errors=3,
            rescue_count=2,
            harm_count=0,
            oracle_call_rate=0.20,
            oracle_mcc_gain=0.10,
        )
        self.assertEqual(inconclusive["decision"], "INCONCLUSIVE")

    def test_jsonl_contract_evaluation(self) -> None:
        labels = [
            {"episode_id": "E1", "malicious": True},
            {"episode_id": "E2", "malicious": False},
        ]
        predictions = [
            {
                "episode_id": "E1",
                "local": {"verdict": "authorized", "confidence": 0.8},
                "remote": {"verdict": "suspicious", "confidence": 0.9},
                "route_to_remote": True,
                "final": {"verdict": "suspicious", "confidence": 0.9},
            },
            {
                "episode_id": "E2",
                "local": {"verdict": "authorized", "confidence": 0.7},
                "remote": {"verdict": "suspicious", "confidence": 0.6},
                "route_to_remote": False,
                "final": {"verdict": "authorized", "confidence": 0.7},
            },
        ]
        result = evaluate_rows(labels, predictions)
        self.assertEqual(result["local"]["fn"], 1)
        self.assertEqual(result["remote"]["fp"], 1)
        self.assertEqual(result["final"]["accuracy"], 1.0)
        self.assertEqual(result["complementarity"]["rescue"], 1)
        self.assertEqual(result["complementarity"]["harm"], 1)

    def test_partial_remote_coverage_cannot_claim_go_or_no_go(self) -> None:
        labels = [
            {"episode_id": "E1", "malicious": True, "split": "blind"},
            {"episode_id": "E2", "malicious": False, "split": "blind"},
        ]
        predictions = [
            {
                "episode_id": "E1",
                "local": {"verdict": "benign", "confidence": 0.8},
                "remote": {"verdict": "attack", "confidence": 0.9},
                "route_to_remote": True,
                "final": {"verdict": "attack", "confidence": 0.9},
            },
            {
                "episode_id": "E2",
                "local": {"verdict": "benign", "confidence": 0.9},
                "remote": None,
                "route_to_remote": False,
                "final": {"verdict": "benign", "confidence": 0.9},
            },
        ]
        result = evaluate_rows(
            labels,
            predictions,
            engineering_stats={
                "witness_success_count": 24,
                "input_contract_valid_count": 24,
                "invalid_false_pass_count": 0,
                "median_witness_tokens": 100,
                "local_parse_success_count": 2,
                "local_parse_total_count": 2,
                "remote_parse_success_count": 1,
                "remote_parse_total_count": 2,
                "request_count": 24,
            },
        )
        self.assertFalse(result["full_remote_coverage"])
        self.assertEqual(result["go_no_go"]["decision"], "INCONCLUSIVE")

    def test_routed_remote_abstention_is_valid_but_uncovered(self) -> None:
        labels = [{"episode_id": "E1", "malicious": True}]
        abstain = {"verdict": "abstain", "malicious_probability": 0.5}
        predictions = [
            {
                "episode_id": "E1",
                "local": {"verdict": "benign", "malicious_probability": 0.2},
                "remote": abstain,
                "route_to_remote": True,
                "final": abstain,
            }
        ]
        result = evaluate_rows(labels, predictions)
        self.assertEqual(result["remote_output_count"], 1)
        self.assertEqual(result["final"]["coverage"], 0.0)
        self.assertEqual(result["final"]["abstention_count"], 1)
        self.assertEqual(result["routing"]["remote_call_count"], 1)
        self.assertEqual(result["routing"]["paid_remote_abstention_count"], 1)


if __name__ == "__main__":
    unittest.main()
