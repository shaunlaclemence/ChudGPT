"""chudgpt — rotate free-tier AI API keys across providers, transparently.

from chudgpt import Rotor

rotor = Rotor.from_env()
print(rotor.chat("hello").text)
"""

from .client import ChudClient, Conversation
from .config import PROVIDERS, ProviderConfig
from .errors import AllProvidersExhausted, ChudGPTError, ConfigError, ProviderError
from .params import Model, Temperature, Tier
from .rotor import Response, Rotor, StreamChunk

__all__ = [
    "PROVIDERS",
    "AllProvidersExhausted",
    "ChudClient",
    "ChudGPTError",
    "ConfigError",
    "Conversation",
    "Model",
    "ProviderConfig",
    "ProviderError",
    "Response",
    "Rotor",
    "StreamChunk",
    "Temperature",
    "Tier",
]
