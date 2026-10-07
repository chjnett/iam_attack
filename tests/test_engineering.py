from __future__ import annotations

import unittest

from flowgate.contracts import make_request
from flowgate.engineering import engineering_stats
from flowgate.inference import MockBackend
from tests.test_contracts import witness


class EngineeringStatsTests(unittest.TestCase):
    def test_counts_are_explicit_and_token_estimate_is_nonzero(self) -> None:
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness(),
        )
        result = engineering_stats(
            [request],
            [MockBackend().predict(request, "test")],
            [],
            invalid_false_pass_count=0,
        )
        self.assertEqual(result["witness_success_count"], 1)
        self.assertEqual(result["parse_success_count"], 1)
        self.assertGreater(result["median_witness_tokens"], 0)
        self.assertIn("estimate", result["token_count_method"])


if __name__ == "__main__":
    unittest.main()
