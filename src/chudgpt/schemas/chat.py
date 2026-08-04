from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt: int = 0
    completion: int = 0
    total: int = 0
    requests: int = 1

    @property
    def reasoning(self) -> int:
        """Hidden thinking tokens: what total leaves over prompt + completion."""
        return max(0, self.total - self.prompt - self.completion)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
            total=self.total + other.total,
            requests=self.requests + other.requests,
        )

    @classmethod
    def from_completion(cls, result: Any) -> Usage:
        usage = getattr(result, "usage", None)
        if usage is None:
            return cls()
        return cls(
            prompt=usage.prompt_tokens or 0,
            completion=usage.completion_tokens or 0,
            total=usage.total_tokens or 0,
        )


@dataclass(repr=False)
class Provider:
    account: str
    name: str
    project_name: str
    project_number: str
    api_key: str

    @property
    def masked_key(self) -> str:
        """The count of hidden characters, then the key's last 5."""
        if len(self.api_key) <= 5:
            return f"**{len(self.api_key)}**"
        return f"**{len(self.api_key) - 5}**{self.api_key[-5:]}"

    def __repr__(self) -> str:
        return (
            f"Provider(account={self.account!r}, name={self.name!r}, "
            f"project_name={self.project_name!r}, "
            f"project_number={self.project_number!r}, api_key={self.masked_key!r})"
        )


@dataclass(repr=False)
class Response:
    text: str
    service: str
    model: str
    usage: Usage
    provider: Provider

    @property
    def message(self) -> Message:
        return {"role": "assistant", "content": self.text}

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"text:     {self.text!r}",
                f"service:  {self.service}",
                f"model:    {self.model}",
                f"usage:    {self.usage}",
                f"provider: {self.provider}",
            ]
        )
