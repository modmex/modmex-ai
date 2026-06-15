from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_openai_flow_example():
    path = Path(__file__).parents[2] / ".examples" / "openai" / "flow.py"
    spec = importlib.util.spec_from_file_location("openai_flow_example", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["openai_flow_example"] = module
    spec.loader.exec_module(module)
    return module
