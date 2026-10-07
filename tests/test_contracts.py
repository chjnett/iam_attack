from __future__ import annotations

import unittest

from flowgate.contracts import make_request, validate_request, validate_response
from flowgate.inference import MockBackend


def witness(actor: str = "P001") -> dict:
    return {
        "schema_version": "1.0",
        "family": "policy_attachment_abuse",
        "split": "blind",
        "prefix_len": 1,
        "capability_state": "verified",
        "capability_gain": True,
        "provenance": ["event:E001:success"],
        "observed_events": [
            {
                "event_id": "E001",
                "timestamp": "2026-01-01T00:00:00Z",
                "action": "iam:AttachRolePolicy",
                "actor": actor,
                "target": "R001",
                "resource": "Y001",
                "outcome": "success",
                "details": {},
            }
        ],
        "candidate_path": ["P001", "R001"],
        "unknown_preconditions": [],
        "feature_summary": {
            "path_length": 2,
            "cross_service_count": 0,
            "observed_event_count": 1,
            "prefix_fraction": 1.0,
            "observed_edge_ratio": 1.0,
            "witness_completeness": 1.0,
            "unknown_precondition_count": 0,
            "sensitive_action_observed": True,
        },
    }


class ContractTests(unittest.TestCase):
    def test_request_is_label_blind_and_digest_bound(self) -> None:
        value = witness()
        value["ground_truth"] = {"malicious": True}
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=value,
        )
        self.assertNotIn("ground_truth", request)
        self.assertNotIn("split", request)
        validate_request(request)

    def test_response_must_match_episode_and_digest(self) -> None:
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness(),
        )
        response = MockBackend().predict(request, "Return JSON")
        validate_response(response, request, allowed_event_ids={"E001"})
        response["witness_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "witness_digest"):
            validate_response(response, request)

    def test_unknown_precondition_forces_mock_abstention(self) -> None:
        value = witness()
        value["capability_state"] = "unknown"
        value["capability_gain"] = None
        value["unknown_preconditions"] = ["scp_snapshot"]
        value["feature_summary"]["unknown_precondition_count"] = 1
        request = make_request(
            case_id="FG-002",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=value,
        )
        response = MockBackend().predict(request, "Return JSON")
        self.assertEqual(response["verdict"], "abstain")

    def test_request_rejects_duplicate_events_and_feature_drift(self) -> None:
        value = witness()
        value["observed_events"] = value["observed_events"] * 2
        value["prefix_len"] = 2
        value["feature_summary"]["observed_event_count"] = 2
        request = make_request(
            case_id="FG-003",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=value,
        )
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            validate_request(request)

        value = witness()
        value["feature_summary"]["path_length"] = 99
        request = make_request(
            case_id="FG-004",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=value,
        )
        with self.assertRaisesRegex(ValueError, "feature/path"):
            validate_request(request)

    def test_boolean_probability_is_not_a_json_number(self) -> None:
        request = make_request(
            case_id="FG-005",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness(),
        )
        response = MockBackend().predict(request, "Return JSON")
        response["malicious_probability"] = True
        with self.assertRaisesRegex(ValueError, "malicious_probability"):
            validate_response(response, request, allowed_event_ids={"E001"})


if __name__ == "__main__":
    unittest.main()
