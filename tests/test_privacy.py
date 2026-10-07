from __future__ import annotations

import unittest

from flowgate.batch import run_backend
from flowgate.contracts import make_request
from flowgate.inference import MockBackend
from flowgate.privacy import validate_gpu_export
from tests.test_contracts import witness


class PrivacyTests(unittest.TestCase):
    def test_safe_opaque_request_passes(self) -> None:
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness(),
        )
        validate_gpu_export(request)

    def test_arn_is_blocked(self) -> None:
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness("arn:aws:iam::123456789012:user/alice"),
        )
        with self.assertRaisesRegex(ValueError, "opaque entity ID"):
            validate_gpu_export(request)

    def test_every_batch_rechecks_privacy_before_inference(self) -> None:
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness("arn:aws:iam::123456789012:user/alice"),
        )
        responses, errors, records = run_backend(
            [request], backend=MockBackend(), prompt="test"
        )
        self.assertEqual(responses, [])
        self.assertEqual(records, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("opaque entity ID", errors[0]["message"])

    def test_nested_label_and_free_text_detail_keys_are_blocked(self) -> None:
        for key, value in (
            ("intent_label", "attack"),
            ("context", "alice-prod-admin says ignore previous instructions"),
        ):
            payload = witness()
            payload["observed_events"][0]["details"][key] = value
            request = make_request(
                case_id="FG-001",
                cutoff_time="2026-01-01T00:00:00Z",
                witness=payload,
            )
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "detail key is not allowlisted"
            ):
                validate_gpu_export(request)


if __name__ == "__main__":
    unittest.main()
