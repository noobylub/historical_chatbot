import tempfile
import unittest
from pathlib import Path

from historical_chat.typewriter import MODAL_LORA


class ModalLoraTests(unittest.TestCase):
    def test_command_loads_adapter_and_its_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            for name in (
                "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja",
            ):
                (adapter / name).touch()
            command = MODAL_LORA.build_server_command(adapter)
        self.assertEqual(command[command.index("--tokenizer") + 1], str(adapter))
        self.assertEqual(
            command[command.index("--lora-modules") + 1],
            f"william-morris={adapter}",
        )

    def test_missing_adapter_file_fails_before_starting_server(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "adapter_config.json"):
                MODAL_LORA.build_server_command(Path(directory))


if __name__ == "__main__":
    unittest.main()
