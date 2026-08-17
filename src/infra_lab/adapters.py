from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AdapterResult:
    adapter_id: str
    tool_calls: int
    files_touched: list[str]
    notes: str = ""


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


def get_adapter(adapter_id: str) -> AgentAdapter:
    if adapter_id == FakeKnownPatchAdapter.adapter_id:
        return FakeKnownPatchAdapter()
    raise KeyError(f"unknown adapter: {adapter_id}")
