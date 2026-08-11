#!/usr/bin/env python3
"""Regenerate src/chudgpt/schemas/__init__.py from the schema modules.

Every public top-level class or function in schemas/*.py is re-exported here,
so a new schema module is picked up without hand-editing this barrel file.
Wired to a PostToolUse hook in .claude/settings.json; run by hand after
adding or renaming a schema:
    uv run python scripts/generate_schemas_init.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = ROOT / "src" / "chudgpt" / "schemas"
OUTPUT = SCHEMAS_DIR / "__init__.py"
RUFF = ROOT / ".venv" / "Scripts" / "ruff.exe"


def public_names(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    definitions = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return sorted(
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, definitions) and not node.name.startswith("_")
    )


def main() -> None:
    modules = {
        path.stem: public_names(path)
        for path in sorted(SCHEMAS_DIR.glob("*.py"))
        if path.stem != "__init__"
    }
    modules = {name: exports for name, exports in modules.items() if exports}

    imports = "\n".join(
        f"from .{module} import {', '.join(exports)}"
        for module, exports in modules.items()
    )
    all_names = sorted(name for exports in modules.values() for name in exports)
    all_block = "\n".join(f'    "{name}",' for name in all_names)
    OUTPUT.write_text(f"{imports}\n\n__all__ = [\n{all_block}\n]\n", encoding="utf-8")

    if RUFF.exists():
        subprocess.run(
            [str(RUFF), "check", "--fix", "--select", "I", "--quiet", str(OUTPUT)],
            check=False,
        )
        subprocess.run([str(RUFF), "format", "--quiet", str(OUTPUT)], check=False)

    print(
        f"wrote {OUTPUT} ({len(all_names)} exports from {len(modules)} modules)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
