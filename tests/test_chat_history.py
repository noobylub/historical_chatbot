import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from chat_history import ENDPOINT, MODEL, chat, main


class ChatTests(unittest.TestCase):
    @patch("chat_history.urlopen")
    def test_sends_one_unauthenticated_chat_request(self, urlopen_mock):
        response = {"choices": [{"message": {"content": "Good day!"}}]}
        urlopen_mock.return_value.__enter__.return_value = io.BytesIO(
            json.dumps(response).encode()
        )

        self.assertEqual(chat("Hello!"), "Good day!")

        urlopen_mock.assert_called_once()
        request = urlopen_mock.call_args.args[0]
        self.assertEqual(
            request.full_url, ENDPOINT + "/v1/chat/completions"
        )
        self.assertEqual(request.method, "POST")
        self.assertFalse(request.has_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello!"}])
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 900)

    @patch("chat_history.urlopen")
    def test_supports_endpoint_override(self, urlopen_mock):
        response = {"choices": [{"message": {"content": "Good day!"}}]}
        urlopen_mock.return_value.__enter__.return_value = io.BytesIO(
            json.dumps(response).encode()
        )
        chat("Hello!", endpoint="https://example.modal.run/")
        self.assertEqual(
            urlopen_mock.call_args.args[0].full_url,
            "https://example.modal.run/v1/chat/completions",
        )

    @patch("sys.argv", ["chat_history.py", "Hello!"])
    @patch("chat_history.urlopen")
    def test_reports_server_error_body(self, urlopen_mock):
        urlopen_mock.side_effect = HTTPError(
            "https://example.modal.run/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error": "Prompt is too long"}'),
        )
        with self.assertRaisesRegex(SystemExit, "HTTP 400:.*Prompt is too long"):
            main()


if __name__ == "__main__":
    unittest.main()