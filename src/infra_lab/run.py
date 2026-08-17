from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one infra-lab experiment task")
    parser.add_argument("experiment", help="experiment id, for example E001")
    parser.add_argument("task", help="task id, for example T01")
    parser.add_argument("--adapter", default=None, help="override the experiment adapter id")
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args()
    result = run_experiment(
        args.experiment,
        args.task,
        adapter_override=args.adapter,
        workspace_root=args.workspace_root,
        results_root=args.results_root,
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
