import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from infra_lab.adapters import LlamaQwen25Coder3BB0Adapter, get_adapter


class E002AdapterTest(unittest.TestCase):
    def _runtime_fixture(self, root: Path, adapter: LlamaQwen25Coder3BB0Adapter) -> tuple[Path, Path, Path]:
        cli = root / "llama-cli"
        cli.write_text("", encoding="utf-8")
        cli.chmod(0o700)
        model = root / "model.gguf"
        model.write_bytes(b"model")
        manifest = root / "runtime.json"
        manifest.write_text(
            json.dumps(
                {
                    "llama_cpp": {"release": "b10218", "cli": str(cli)},
                    "model": {
                        "id": adapter.expected_model_id,
                        "quantization": adapter.expected_quant,
                        "path": str(model),
                        "sha256": "test-sha",
                    },
                }
            ),
            encoding="utf-8",
        )
        return cli, model, manifest

    def test_real_adapter_is_registered_and_applies_schema_constrained_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "calc.py"
            target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            adapter = LlamaQwen25Coder3BB0Adapter()
            _, _, manifest = self._runtime_fixture(root, adapter)
            fake = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "path": "calc.py",
                        "content": "def add(a, b):\n    return a + b\n",
                        "summary": "Fixed addition.",
                    }
                ),
                stderr="prompt eval time = 1.0 ms / 50 tokens (50.0 tokens per second)\n"
                "eval time = 1.0 ms / 20 runs (20.0 tokens per second)\n",
            )
            with patch.dict(os.environ, {"INFRA_LOCAL_MODEL_MANIFEST": str(manifest)}, clear=False):
                with patch("infra_lab.adapters.subprocess.run", return_value=fake) as run:
                    adapter = get_adapter("llama-qwen25-coder-3b-q4km-b0")
                    self.assertIsInstance(adapter, LlamaQwen25Coder3BB0Adapter)
                    result = adapter.run(workspace, "Fix addition")

            self.assertIn("return a + b", target.read_text(encoding="utf-8"))
            self.assertEqual(result.files_touched, ["calc.py"])
            self.assertEqual(result.tool_calls, 1)
            command = run.call_args.args[0]
            self.assertIn("-ngl", command)
            self.assertIn("all", command)
            self.assertIn("-sm", command)
            self.assertIn("none", command)
            self.assertIn("-j", command)
            self.assertIn("--output-file", command)
            self.assertEqual(result.metrics["output_transport"], "llama-cli-output-file")
            self.assertEqual(result.metrics["model_id"], "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")

    def test_output_file_assistant_section_is_preferred_over_cli_ui_stdout(self) -> None:
        adapter = LlamaQwen25Coder3BB0Adapter()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "llama-output.txt"
            output.write_text(
                "User:\nfix it\n\nAssistant:\n{\"path\":\"calc.py\",\"content\":\"fixed\"}\n\n",
                encoding="utf-8",
            )
            captured = adapter._assistant_output(output, "llama UI banner without JSON")
        self.assertEqual(captured, '{"path":"calc.py","content":"fixed"}')

    def test_model_cannot_write_a_file_outside_snapshot(self) -> None:
        adapter = LlamaQwen25Coder3BB0Adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "calc.py").write_text("x = 1\n", encoding="utf-8")
            _, _, manifest = self._runtime_fixture(root, adapter)
            fake = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout='{"path":"../escape.py","content":"bad"}', stderr="",
            )
            with patch.dict(os.environ, {"INFRA_LOCAL_MODEL_MANIFEST": str(manifest)}, clear=False):
                with patch("infra_lab.adapters.subprocess.run", return_value=fake):
                    with self.assertRaisesRegex(RuntimeError, "outside the bounded"):
                        adapter.run(workspace, "test")

    def test_invalid_model_output_preserves_bounded_stdout_and_stderr_for_diagnosis(self) -> None:
        adapter = LlamaQwen25Coder3BB0Adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "calc.py").write_text("x = 1\n", encoding="utf-8")
            _, _, manifest = self._runtime_fixture(root, adapter)
            fake = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout="not json from model\n",
                stderr="runtime diagnostic line\n",
            )
            with patch.dict(os.environ, {"INFRA_LOCAL_MODEL_MANIFEST": str(manifest)}, clear=False):
                with patch("infra_lab.adapters.subprocess.run", return_value=fake):
                    with self.assertRaises(RuntimeError) as caught:
                        adapter.run(workspace, "test")
            message = str(caught.exception)
            self.assertIn("assistant_output=not json from model", message)
            self.assertIn("llama_stdout=not json from model\\n", message)
            self.assertIn("llama_stderr=runtime diagnostic line\\n", message)


if __name__ == "__main__":
    unittest.main()
