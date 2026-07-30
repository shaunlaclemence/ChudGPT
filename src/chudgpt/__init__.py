"""chudgpt — rotate free-tier AI API keys across providers, transparently.

from chudgpt import Rotor

rotor = Rotor.from_env()
print(rotor.chat("hello").text)
"""

from .client import ChudClient, Conversation
from .config import PROVIDERS, ProviderConfig
from .errors import AllProvidersExhausted, ChudGPTError, ConfigError, ProviderError
from .rotor import Response, Rotor, StreamChunk

__all__ = [
    "PROVIDERS",
    "AllProvidersExhausted",
    "ChudClient",
    "ChudGPTError",
    "ConfigError",
    "Conversation",
    "ProviderConfig",
    "ProviderError",
    "Response",
    "Rotor",
    "StreamChunk",
]
