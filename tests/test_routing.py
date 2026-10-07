from __future__ import annotations

import unittest

from flowgate.contracts import make_request
from flowgate.inference import MockBackend
from flowgate.routing import select_remote_calls
from tests.test_contracts import witness


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = []
        self.responses = []
        for index in range(8):
            episode_id = f"FG-T{index + 1:03d}"
            payload = witness()
            payload["capability_state"] = "unknown" if index < 2 else "verified"
            payload["capability_gain"] = None if index < 2 else True
            payload["unknown_preconditions"] = ["missing_state"] if index < 2 else []
            payload["feature_summary"]["unknown_precondition_count"] = int(index < 2)
            request = make_request(
                case_id=episode_id,
                cutoff_time="2026-01-01T00:00:00Z",
                witness=payload,
            )
            response = MockBackend().predict(request, "test")
            if response["verdict"] != "abstain":
                response["malicious_probability"] = 0.55 + index * 0.04
            self.requests.append(request)
            self.responses.append(response)

    def test_flowgate_respects_exact_budget_and_deprioritizes_unknown_evidence(
        self,
    ) -> None:
        decisions = select_remote_calls(
            self.requests,
            self.responses,
            policy="flowgate-v0",
            budget_fraction=0.25,
        )
        selected = [row.episode_id for row in decisions if row.selected]
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected), {"FG-T003", "FG-T004"})
        ranks = {row.episode_id: row.rank for row in decisions}
        self.assertGreater(ranks["FG-T001"], ranks["FG-T003"])
        unknown = next(row for row in decisions if row.episode_id == "FG-T001")
        self.assertIn("irreducible_unknowns=1", unknown.reasons)
        self.assertIn("capability=unknown", unknown.reasons)

    def test_complete_ambiguous_case_beats_incomplete_abstention(self) -> None:
        complete_request = self.requests[2]
        complete_response = dict(self.responses[2])
        complete_response["malicious_probability"] = 0.51
        incomplete_request = self.requests[0]
        incomplete_response = self.responses[0]

        decisions = select_remote_calls(
            [incomplete_request, complete_request],
            [incomplete_response, complete_response],
            policy="flowgate-v0",
            budget_fraction=0.5,
        )
        selected = [row.episode_id for row in decisions if row.selected]
        self.assertEqual(selected, ["FG-T003"])
        self.assertGreater(decisions[0].score, decisions[1].score)

    def test_call_budget_is_exact_and_overrides_fraction(self) -> None:
        decisions = select_remote_calls(
            self.requests,
            self.responses,
            policy="flowgate-v0",
            budget_fraction=0.0,
            call_budget=3,
        )
        self.assertEqual(sum(row.selected for row in decisions), 3)
        self.assertEqual({row.budget_size for row in decisions}, {3})

    def test_complex_complete_path_gets_marginal_rescue_bonus(self) -> None:
        simple_payload = witness()
        simple_request = make_request(
            case_id="FG-SIMPLE",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=simple_payload,
        )
        complex_payload = witness()
        complex_payload["candidate_path"] = [
            "P001",
            "iam:AttachRolePolicy",
            "R001",
            "sts:AssumeRole",
            "R002",
            "s3:GetObject",
        ]
        complex_payload["feature_summary"]["path_length"] = 6
        complex_payload["feature_summary"]["cross_service_count"] = 2
        complex_request = make_request(
            case_id="FG-COMPLEX",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=complex_payload,
        )
        simple_response = MockBackend().predict(simple_request, "test")
        complex_response = MockBackend().predict(complex_request, "test")
        simple_response["malicious_probability"] = 0.65
        complex_response["malicious_probability"] = 0.65

        decisions = select_remote_calls(
            [simple_request, complex_request],
            [simple_response, complex_response],
            policy="flowgate-v0",
            budget_fraction=0.5,
        )
        self.assertTrue(decisions[0].selected)
        self.assertEqual(decisions[0].episode_id, "FG-COMPLEX")
        self.assertIn("cross_service_count=2", decisions[0].reasons)

    def test_uncertainty_baseline_remains_abstention_first(self) -> None:
        decisions = select_remote_calls(
            self.requests,
            self.responses,
            policy="uncertainty",
            budget_fraction=0.25,
        )
        selected = {row.episode_id for row in decisions if row.selected}
        self.assertEqual(selected, {"FG-T001", "FG-T002"})

    def test_random_baseline_is_reproducible(self) -> None:
        first = select_remote_calls(
            self.requests,
            self.responses,
            policy="random",
            budget_fraction=0.25,
            seed=11,
        )
        second = select_remote_calls(
            self.requests,
            self.responses,
            policy="random",
            budget_fraction=0.25,
            seed=11,
        )
        self.assertEqual([row.to_dict() for row in first], [row.to_dict() for row in second])


if __name__ == "__main__":
    unittest.main()
