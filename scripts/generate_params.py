#!/usr/bin/env python3
"""Regenerate src/chudgpt/params.py from config.TIERS / config.MODEL_CATALOG.

Tier and Model are real, statically-typed enum classes (not built at runtime
via the functional Enum() API) so IDEs and type checkers can see every member.
Run this after editing src/chudgpt/config.json (the model catalog) or
config.TIERS, instead of hand-editing the generated members:

    uv run python scripts/generate_params.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chudgpt.config import MODEL_CATALOG, TIERS

OUTPUT = Path(__file__).resolve().parent.parent / "src" / "chudgpt" / "params.py"

HEADER = '''"""Discoverability enums for tier/model/temperature — what you can pass to
ChudClient.ask()/stream()/start_conversation() and Conversation.send()/ask().

GENERATED FILE — do not hand-edit Tier or Model. Regenerate with:
    uv run python scripts/generate_params.py
after changing config.TIERS or src/chudgpt/config.json (the model catalog).
"""

from __future__ import annotations

from enum import Enum
'''

TEMPERATURE_BLOCK = '''

class Temperature(float, Enum):
    """Named presets for the ``temperature=`` request kwarg (an OpenAI-compatible
    sampling parameter, typically 0.0-2.0). NOT exhaustive — pass any float for a
    value in between. Lower = more deterministic/focused; higher = more random.
    """

    PRECISE = 0.0
    BALANCED = 0.7
    CREATIVE = 1.0
    WILD = 1.4
'''


def _member_name(provider: str, model_id: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", model_id).strip("_").upper()
    prefix = f"{provider.upper()}_"
    return slug if slug.startswith(prefix) else f"{prefix}{slug}"


def main() -> None:
    tier_lines = "\n".join(f'    {t.upper()} = "{t}"' for t in TIERS)
    model_lines = "\n".join(
        f'    {_member_name(provider, model_id)} = "{model_id}"'
        for provider, model_ids in MODEL_CATALOG.items()
        for model_id in model_ids
    )

    content = (
        HEADER
        + '\n\nclass Tier(str, Enum):\n'
        + '    """Quality/speed tradeoff — resolved to a concrete model per provider\n'
        + '    via ``ProviderConfig.model_for()``. Works anywhere a plain\n'
        + '    ``tier="best"`` string does."""\n\n'
        + tier_lines
        + '\n\n\nclass Model(str, Enum):\n'
        + '    """Every real, currently-usable model id per provider (see config.json).\n'
        + '    A ``model=`` override is sent as-is regardless of which provider ends up\n'
        + '    serving the request during rotation — only pass one of these if you also\n'
        + '    constrain ``providers=`` to the single provider that actually serves it,\n'
        + '    otherwise a request that rotates to a different provider will fail with\n'
        + '    an unknown-model error."""\n\n'
        + model_lines
        + "\n"
        + TEMPERATURE_BLOCK
    )
    OUTPUT.write_text(content)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
