from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infra_lab.runner import bounded_exception, run_experiment


class E001RunnerTests(unittest.TestCase):
    def test_fake_adapter_completes_t01_in_isolated_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_experiment(
                "E001",
                "T01",
                workspace_root=root / "workspaces",
                results_root=root / "results",
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["adapter"], "fake-known-patch-v1")
            self.assertEqual(result["files_changed"], ["calc.py"])
            self.assertIn("return a + b", result["diff"])
            self.assertEqual(result["acceptance"]["exit_code"], 0)
            self.assertIn("T01_ACCEPTANCE=PASS", result["acceptance"]["stdout"])
            self.assertEqual(len(result["task_base_revision"]), 40)
            result_path = Path(result["result_path"])
            self.assertTrue(result_path.is_file())
            persisted = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["run_id"], result["run_id"])

    def test_unknown_adapter_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(KeyError):
                run_experiment(
                    "E001",
                    "T01",
                    adapter_override="not-a-real-adapter",
                    workspace_root=root / "workspaces",
                    results_root=root / "results",
                )

    def test_bounded_exception_keeps_failure_tail(self) -> None:
        suffix = "FINAL_INTERNAL_RUNTIME_ERROR"
        error = RuntimeError("prefix-" + ("x" * 6000) + suffix)
        captured = bounded_exception(error, limit=1200)
        self.assertLessEqual(len(captured), 1200)
        self.assertIn("RuntimeError: prefix-", captured)
        self.assertIn("...<truncated>...", captured)
        self.assertTrue(captured.endswith(suffix))


if __name__ == "__main__":
    unittest.main()
