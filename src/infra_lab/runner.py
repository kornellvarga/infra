from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import get_adapter


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail[-500:]}")
    return result


def source_revision(root: Path) -> str:
    result = git(root, "rev-parse", "HEAD", check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else "unknown"


def initialize_task_repository(workspace: Path) -> str:
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "add", "--all")
    result = subprocess.run(
        [
            "git", "-C", str(workspace),
            "-c", "user.name=infra-lab",
            "-c", "user.email=infra-lab@localhost",
            "-c", "commit.gpgSign=false",
            "-c", "core.hooksPath=/dev/null",
            "commit", "--quiet", "--no-verify", "-m", "Task baseline",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not commit task baseline: {(result.stderr or result.stdout).strip()[-500:]}")
    return git(workspace, "rev-parse", "HEAD").stdout.strip()


def acceptance_result(script: Path, workspace: Path, timeout_seconds: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["INFRA_TASK_WORKSPACE"] = str(workspace)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=workspace,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
        timed_out = False
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def changed_files(workspace: Path) -> list[str]:
    output = git(workspace, "status", "--porcelain=v1", "--untracked-files=all").stdout
    files: list[str] = []
    for line in output.splitlines():
        if len(line) >= 4:
            files.append(line[3:])
    return files


def compact_diff(workspace: Path) -> str:
    return git(workspace, "diff", "--no-ext-diff", "--unified=3").stdout[-12000:]


def run_experiment(
    experiment_id: str,
    task_id: str,
    *,
    adapter_override: str | None = None,
    workspace_root: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    root = repository_root()
    experiment_path = root / "experiments" / experiment_id / "experiment.json"
    task_path = root / "benchmarks" / task_id / "task.json"
    experiment = load_json(experiment_path)
    task = load_json(task_path)
    if experiment.get("id") != experiment_id or task.get("id") != task_id:
        raise RuntimeError("experiment/task identity does not match its path")
    tasks = experiment.get("tasks")
    if not isinstance(tasks, list) or task_id not in tasks:
        raise RuntimeError(f"task {task_id} is not declared by experiment {experiment_id}")
    timeout_seconds = experiment.get("timeout_seconds", 60)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
        raise RuntimeError("experiment timeout_seconds must be an integer from 1 to 3600")
    adapter_id = adapter_override or experiment.get("default_adapter")
    if not isinstance(adapter_id, str) or not adapter_id:
        raise RuntimeError("experiment requires a default adapter")
    seed_rel = task.get("seed")
    acceptance_rel = task.get("acceptance")
    prompt = task.get("prompt")
    if not all(isinstance(value, str) and value for value in (seed_rel, acceptance_rel, prompt)):
        raise RuntimeError("task definition is incomplete")
    seed = (root / seed_rel).resolve()
    acceptance = (root / acceptance_rel).resolve()
    if root.resolve() not in seed.parents or root.resolve() not in acceptance.parents:
        raise RuntimeError("task paths escaped the repository")
    if not seed.is_dir() or not acceptance.is_file():
        raise RuntimeError("task seed or acceptance test is missing")

    external_root = Path.home() / ".infra-lab"
    workspace_root = (workspace_root or external_root / "workspaces").expanduser().resolve()
    results_root = (results_root or external_root / "results").expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{experiment_id}-{task_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = workspace_root / run_id
    workspace = run_dir / "task"
    run_dir.mkdir(parents=False, exist_ok=False)
    shutil.copytree(seed, workspace)

    infra_sha = source_revision(root)
    task_base_sha = initialize_task_repository(workspace)
    started_at = utc_now()
    started = time.perf_counter()
    adapter = get_adapter(adapter_id)
    adapter_error: str | None = None
    adapter_result: dict[str, Any] | None = None
    try:
        adapter_result = asdict(adapter.run(workspace, prompt))
    except Exception as exc:
        adapter_error = f"{type(exc).__name__}: {exc}"[:1000]

    acceptance = acceptance_result(acceptance, workspace, timeout_seconds)
    total_seconds = round(time.perf_counter() - started, 6)
    files = changed_files(workspace)
    result: dict[str, Any] = {
        "schema": 1,
        "run_id": run_id,
        "experiment": experiment_id,
        "task": task_id,
        "adapter": adapter_id,
        "success": adapter_error is None and acceptance["exit_code"] == 0,
        "source_revision": infra_sha,
        "task_base_revision": task_base_sha,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": total_seconds,
        "workspace": str(workspace),
        "prompt": prompt,
        "adapter_result": adapter_result,
        "adapter_error": adapter_error,
        "files_changed": files,
        "diff": compact_diff(workspace),
        "acceptance": acceptance,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
    }
    result_path = results_root / f"{run_id}.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"INFRA_RUN_ID={run_id}")
    print(f"INFRA_SOURCE_REVISION={infra_sha}")
    print(f"INFRA_TASK_BASE_REVISION={task_base_sha}")
    print(f"INFRA_RESULT_PATH={result_path}")
    print(f"INFRA_ACCEPTANCE={'PASS' if result['success'] else 'FAIL'}")
    return result
