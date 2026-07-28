"""chudgpt — rotate free-tier AI API keys across providers, transparently.

from chudgpt import Rotor

rotor = Rotor.from_env()
print(rotor.chat("hello").text)
"""

from .config import PROVIDERS, ProviderConfig
from .errors import AllProvidersExhausted, ChudGPTError, ConfigError, ProviderError
from .rotor import Response, Rotor

__all__ = [
    "PROVIDERS",
    "AllProvidersExhausted",
    "ChudGPTError",
    "ConfigError",
    "ProviderConfig",
    "ProviderError",
    "Response",
    "Rotor",
]
