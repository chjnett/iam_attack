from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from flowgate.contracts import make_request
from flowgate.inference import OpenAICompatibleBackend
from tests.test_contracts import witness


class OpenAICompatibleBackendTests(unittest.TestCase):
    def test_predict_sends_max_completion_tokens(self) -> None:
        request = make_request(
            case_id="FG-001",
            cutoff_time="2026-01-01T00:00:00Z",
            witness=witness(),
        )
        backend = OpenAICompatibleBackend(
            base_url="https://example.test/v1",
            model_id="gpt-5.4-mini",
            api_key_env="FLOWGATE_TEST_API_KEY",
            max_output_tokens=321,
        )
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "{}"}}]}
        ).encode("utf-8")

        with (
            patch.dict(os.environ, {"FLOWGATE_TEST_API_KEY": "test-key"}),
            patch("flowgate.inference.urllib.request.urlopen") as urlopen,
        ):
            urlopen.return_value.__enter__.return_value = response
            backend.predict(request, "Return JSON")

        http_request = urlopen.call_args.args[0]
        body = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(body["max_completion_tokens"], 321)
        self.assertNotIn("max_tokens", body)
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        uncertainty = schema["properties"]["uncertainty_reasons"]
        self.assertEqual(uncertainty["type"], "array")
        self.assertIn("none", uncertainty["items"]["enum"])
        self.assertEqual(
            schema["properties"]["episode_id"]["enum"], [request["episode_id"]]
        )


if __name__ == "__main__":
    unittest.main()
