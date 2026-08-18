from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class AdapterResult:
    adapter_id: str
    tool_calls: int
    files_touched: list[str]
    notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(Protocol):
    adapter_id: str

    def run(self, workspace: Path, prompt: str) -> AdapterResult:
        ...


class FakeKnownPatchAdapter:
    """Deterministic adapter used only to validate the E001 harness.

    It is intentionally not intelligent: it applies one known patch to T01 so
    failures in E001 point at the runner/provenance/test plumbing rather than at
    model quality.
    """

    adapter_id = "fake-known-patch-v1"

    def run(self, workspace: Path, prompt: str) -> AdapterResult:
        target = workspace / "calc.py"
        before = target.read_text(encoding="utf-8")
        old = "    return a - b\n"
        new = "    return a + b\n"
        if before.count(old) != 1:
            raise RuntimeError("fake adapter expected exactly one known broken return statement")
        target.write_text(before.replace(old, new), encoding="utf-8")
        return AdapterResult(
            adapter_id=self.adapter_id,
            tool_calls=2,
            files_touched=["calc.py"],
            notes="Applied deterministic known patch for harness validation.",
        )


class LlamaQwen25Coder3BB0Adapter:
    """B0: one local-model turn over a bounded workspace snapshot.

    This deliberately has no iterative tool loop, self-review, test feedback,
    memory, or retry. The model receives the task plus the current text files
    and must return one JSON-contract full-file replacement. Later experiments
    can add capabilities one at a time against this baseline.
    """

    adapter_id = "llama-qwen25-coder-3b-q4km-b0"
    expected_model_id = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
    expected_quant = "Q4_K_M"
    expected_llama_release = "b10218"
    max_workspace_bytes = 48_000
    max_file_bytes = 32_000

    @staticmethod
    def _manifest_path() -> Path:
        override = os.environ.get("INFRA_LOCAL_MODEL_MANIFEST")
        return Path(override).expanduser() if override else Path.home() / ".infra-lab" / "local-model-runtime.json"

    def _runtime(self) -> tuple[Path, Path, dict[str, Any]]:
        path = self._manifest_path()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"local model runtime manifest unavailable: {exc}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("local model runtime manifest is not an object")
        llama = manifest.get("llama_cpp")
        model = manifest.get("model")
        if not isinstance(llama, dict) or not isinstance(model, dict):
            raise RuntimeError("local model runtime manifest is incomplete")
        if llama.get("release") != self.expected_llama_release:
            raise RuntimeError("local llama.cpp release does not match the B0 contract")
        if model.get("id") != self.expected_model_id or model.get("quantization") != self.expected_quant:
            raise RuntimeError("local model identity does not match the B0 contract")
        cli = Path(str(llama.get("cli") or "")).expanduser()
        model_path = Path(str(model.get("path") or "")).expanduser()
        if not cli.is_file() or not os.access(cli, os.X_OK):
            raise RuntimeError("pinned llama-cli is missing or not executable")
        if not model_path.is_file():
            raise RuntimeError("pinned Qwen model file is missing")
        return cli, model_path, manifest

    def _workspace_snapshot(self, workspace: Path) -> tuple[str, set[str]]:
        blocks: list[str] = []
        allowed: set[str] = set()
        total = 0
        for path in sorted(workspace.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace)
            if ".git" in rel.parts:
                continue
            if path.stat().st_size > self.max_file_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            rendered = f"\n--- FILE: {rel.as_posix()} ---\n{text}\n--- END FILE ---\n"
            encoded = len(rendered.encode("utf-8"))
            if total + encoded > self.max_workspace_bytes:
                break
            blocks.append(rendered)
            allowed.add(rel.as_posix())
            total += encoded
        if not blocks:
            raise RuntimeError("B0 adapter found no bounded text files in the task workspace")
        return "".join(blocks), allowed

    @staticmethod
    def _json_object(stdout: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stdout):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stdout[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise RuntimeError("local model did not return a JSON edit object")

    @staticmethod
    def _bounded_process_text(value: str, limit: int = 1600) -> str:
        if not value:
            return "<empty>"
        rendered = value.replace("\x00", "").replace("\r", "\\r").replace("\n", "\\n")
        return rendered[-limit:]

    @staticmethod
    def _timing_metrics(runtime_text: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        patterns = {
            "prompt_tokens": r"prompt eval time\s*=.*?/\s*(\d+) tokens",
            "generated_tokens": r"eval time\s*=.*?/\s*(\d+) runs",
            "prompt_tokens_per_second": r"prompt eval time\s*=.*?([0-9.]+) tokens per second",
            "generated_tokens_per_second": r"eval time\s*=.*?([0-9.]+) tokens per second",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, runtime_text, flags=re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1)) if "per_second" in key else int(match.group(1))
        ui_timing = re.search(
            r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]",
            runtime_text,
            flags=re.IGNORECASE,
        )
        if ui_timing:
            metrics["prompt_tokens_per_second"] = float(ui_timing.group(1))
            metrics["generated_tokens_per_second"] = float(ui_timing.group(2))
        return metrics

    @staticmethod
    def _assistant_output(output_path: Path, stdout: str) -> str:
        if output_path.is_file():
            captured = output_path.read_text(encoding="utf-8", errors="replace")
            marker = "Assistant:\n"
            if marker in captured:
                return captured.rsplit(marker, 1)[1].strip()
            return captured.strip()
        return stdout.strip()

    def run(self, workspace: Path, prompt: str) -> AdapterResult:
        cli, model_path, manifest = self._runtime()
        snapshot, allowed = self._workspace_snapshot(workspace)
        model_prompt = (
            "You are a coding agent in a controlled benchmark. Fix the user's task using the workspace snapshot. "
            "Return exactly one JSON object with `path`, `content`, and optionally `summary`. `path` must name "
            "exactly one existing workspace file shown below. `content` must be the complete replacement content "
            "for that file. Do not use markdown. Do not explain outside the JSON.\n\n"
            f"TASK:\n{prompt}\n\nWORKSPACE SNAPSHOT:{snapshot}"
        )
        main_gpu = os.environ.get("INFRA_LLAMA_MAIN_GPU", "0")
        if main_gpu not in {"0", "1"}:
            raise RuntimeError("INFRA_LLAMA_MAIN_GPU must be 0 or 1")
        timeout_raw = os.environ.get("INFRA_MODEL_TIMEOUT_SECONDS", "180")
        try:
            timeout_seconds = int(timeout_raw)
        except ValueError as exc:
            raise RuntimeError("INFRA_MODEL_TIMEOUT_SECONDS must be an integer") from exc
        if not 10 <= timeout_seconds <= 1800:
            raise RuntimeError("INFRA_MODEL_TIMEOUT_SECONDS must be from 10 to 1800")

        output_path = workspace.parent / "llama-assistant-output.txt"
        output_path.unlink(missing_ok=True)
        command = [
            str(cli),
            "-m", str(model_path),
            "-ngl", "all",
            "-sm", "none",
            "-mg", main_gpu,
            "-c", "4096",
            "-n", "1024",
            "--temp", "0",
            "--seed", "1",
            "--no-display-prompt",
            "--no-warmup",
            "-co", "off",
            "-st",
            "--simple-io",
            "--output-file", str(output_path),
            "-p", model_prompt,
        ]
        env = os.environ.copy()
        env["HOME"] = str(Path.home())
        env["LC_ALL"] = "C"
        env["NO_COLOR"] = "1"
        started = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
        if started.returncode != 0:
            detail = (started.stderr or started.stdout).strip().replace("\n", "; ")[-1000:]
            raise RuntimeError(f"llama.cpp inference failed with exit {started.returncode}: {detail}")
        model_output = self._assistant_output(output_path, started.stdout)
        try:
            edit = self._json_object(model_output)
        except RuntimeError as exc:
            output = self._bounded_process_text(model_output)
            stdout = self._bounded_process_text(started.stdout)
            stderr = self._bounded_process_text(started.stderr)
            raise RuntimeError(
                f"{exc}; assistant_output={output}; llama_stdout={stdout}; llama_stderr={stderr}"
            ) from exc
        relative = edit.get("path")
        content = edit.get("content")
        unknown = set(edit) - {"path", "content", "summary"}
        if unknown:
            raise RuntimeError("local model JSON edit contained unsupported keys")
        if not isinstance(relative, str) or relative not in allowed:
            raise RuntimeError("local model selected a file outside the bounded existing workspace snapshot")
        if not isinstance(content, str) or len(content.encode("utf-8")) > 100_000:
            raise RuntimeError("local model replacement content is invalid or too large")
        summary = edit.get("summary")
        if summary is not None and not isinstance(summary, str):
            raise RuntimeError("local model JSON summary must be a string when provided")
        target = (workspace / relative).resolve()
        root = workspace.resolve()
        if root not in target.parents or not target.is_file():
            raise RuntimeError("local model edit escaped the task workspace")
        target.write_text(content, encoding="utf-8")
        runtime_text = started.stdout + "\n" + started.stderr
        metrics = self._timing_metrics(runtime_text)
        metrics.update({
            "main_gpu": int(main_gpu),
            "llama_release": self.expected_llama_release,
            "model_id": self.expected_model_id,
            "quantization": self.expected_quant,
            "model_sha256": str(manifest.get("model", {}).get("sha256") or ""),
            "stderr_tail": started.stderr[-1500:],
            "output_transport": "llama-cli-output-file",
            "json_constraint": "post-generation-strict-validation",
        })
        return AdapterResult(
            adapter_id=self.adapter_id,
            tool_calls=1,
            files_touched=[relative],
            notes=summary[:500] if isinstance(summary, str) and summary else "Single JSON-contract local-model edit.",
            metrics=metrics,
        )


def get_adapter(adapter_id: str) -> AgentAdapter:
    if adapter_id == FakeKnownPatchAdapter.adapter_id:
        return FakeKnownPatchAdapter()
    if adapter_id == LlamaQwen25Coder3BB0Adapter.adapter_id:
        return LlamaQwen25Coder3BB0Adapter()
    raise KeyError(f"unknown adapter: {adapter_id}")
