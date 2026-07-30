"""chudgpt — rotate free-tier AI API keys across providers, transparently.

    from chudgpt import ChudClient

    client = ChudClient(secrets_path="secrets.json")
    print(client.ask("hello").text)

``ChudClient`` is the entry point. Also public: the ``Tier``/``Model``/
``Temperature`` enums, the exception hierarchy (all deriving from
``ChudGPTError``), and the result types returned by client calls
(``Response``, ``StreamChunk``, ``KeyUsage``, ``Conversation``).

Everything else (``Rotor``, ``PROVIDERS``, ``ProviderConfig``, ``QuotaTracker``,
keystore/state helpers) is internal — reachable via submodules if you really need
it, but not part of the supported surface and free to change between versions.
"""

from .client import ChudClient, Conversation
from .errors import (
    AllProvidersExhausted,
    ChudGPTError,
    ConfigError,
    InvalidRequestError,
    InvalidTierError,
    ProviderError,
    SecretsFileError,
    StreamInterrupted,
    UnknownProviderError,
)
from .params import Model, Temperature, Tier
from .rotor import KeyUsage, Response, StreamChunk

__all__ = [
    "AllProvidersExhausted",
    "ChudClient",
    "ChudGPTError",
    "ConfigError",
    "Conversation",
    "InvalidRequestError",
    "InvalidTierError",
    "KeyUsage",
    "Model",
    "ProviderError",
    "Response",
    "SecretsFileError",
    "StreamChunk",
    "StreamInterrupted",
    "Temperature",
    "Tier",
    "UnknownProviderError",
]
