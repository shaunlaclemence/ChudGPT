from __future__ import annotations

import importlib
from typing import Any, ClassVar

from chudgpt._schemas.plugin import Plugin


class PluginRegistry:
    REPOSITORY = "git+https://github.com/shaunlaclemence/ChudGPT.git"
    PLUGINS: ClassVar[dict[str, Plugin]] = {
        "audio": Plugin(
            module="chudgpt.audio._main",
            service="AudioService",
            extra="chudgpt[audio]",
        ),
    }

    def attach(self, client: Any) -> dict[str, Any]:
        attached = {}
        for name, plugin in self.PLUGINS.items():
            service = self.__service(plugin)
            if service is not None:
                attached[name] = service(client)
        return attached

    @staticmethod
    def __service(plugin: Plugin) -> Any | None:
        try:
            return getattr(importlib.import_module(plugin.module), plugin.service)
        except ImportError:
            return None

    @classmethod
    def missing(cls, name: str) -> str:
        plugin = cls.PLUGINS.get(name)
        if plugin is None:
            return f"'ChudGPT' object has no attribute {name!r}"
        return (
            f"chud.{name} needs the {name} plugin, which is not installed. "
            f'install it with: uv add "{plugin.extra} @ {cls.REPOSITORY}"'
        )
