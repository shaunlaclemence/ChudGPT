import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from chudgpt import ChudGPT

APP_NAME = "chudgpt-pytests"
OUTPUT_DIR = Path(__file__).parent / "outputs"


@pytest.fixture(scope="session")
def chud() -> ChudGPT:
    return ChudGPT().initialise(app_name=APP_NAME)


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


@pytest.fixture
def save_output(request):
    """Write a payload to tests/outputs/<test name>_<run>.json, never overwriting.

    The run number is one past the highest already on disk, so every run of a
    live test is kept and two runs can be diffed against each other.
    """

    def save(payload: Any) -> Path:
        OUTPUT_DIR.mkdir(exist_ok=True)
        name = request.node.name
        runs = [
            int(found.group(1))
            for path in OUTPUT_DIR.glob(f"{name}_*.json")
            if (found := re.fullmatch(rf"{re.escape(name)}_(\d+)", path.stem))
        ]
        path = OUTPUT_DIR / f"{name}_{max(runs, default=0) + 1:03d}.json"
        path.write_text(
            json.dumps(payload, indent=2, default=jsonable, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nsaved {path.relative_to(Path(__file__).parent.parent)}")
        return path

    return save
