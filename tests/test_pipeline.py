from __future__ import annotations

import unittest

from flowgate.batch import merge_predictions, selected_requests
from flowgate.contracts import make_request
from flowgate.inference import MockBackend
from flowgate.routing import select_remote_calls
from tests.test_contracts import witness


class PipelineIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness(),
        )
        self.local = MockBackend().predict(self.request, "test")

    def test_router_rejects_tampered_response_binding(self) -> None:
        tampered = dict(self.local)
        tampered["witness_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "witness_digest"):
            select_remote_calls(
                [self.request], [tampered], policy="never", budget_fraction=0.0
            )

    def test_merge_rejects_duplicate_and_extra_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate local response"):
            merge_predictions(
                [self.request],
                [self.local, self.local],
                [],
                [],
            )
        extra = dict(self.local)
        extra["episode_id"] = "FG-EXTRA"
        with self.assertRaisesRegex(ValueError, "unexpected remote response"):
            merge_predictions(
                [self.request],
                [self.local],
                [],
                [extra],
            )

    def test_string_false_routing_decision_is_rejected(self) -> None:
        bad = [{"episode_id": "FG-001", "route_to_remote": "false"}]
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            selected_requests([self.request], bad)


if __name__ == "__main__":
    unittest.main()
