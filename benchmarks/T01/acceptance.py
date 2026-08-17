from __future__ import annotations

import importlib.util
import os
from pathlib import Path

workspace = Path(os.environ["INFRA_TASK_WORKSPACE"])
module_path = workspace / "calc.py"
spec = importlib.util.spec_from_file_location("task_calc", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("could not load task module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

cases = [
    (2, 3, 5),
    (-4, 7, 3),
    (-5, -6, -11),
    (0, 19, 19),
]
for left, right, expected in cases:
    actual = module.add(left, right)
    if actual != expected:
        raise AssertionError(f"add({left}, {right}) returned {actual}, expected {expected}")
print("T01_ACCEPTANCE=PASS")
