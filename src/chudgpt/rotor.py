"""The rotor: one chat() entry point that transparently rotates providers/keys."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import openai

from .config import PROVIDERS, ProviderConfig
from .errors import AllProvidersExhausted, ConfigError, ProviderError
from .keystore import key_id, load_keys
from .quota import QuotaTracker, next_reset
from .state import load_state, save_state

TRANSIENT_COOLDOWN_S = 60  # 5xx / network blip: retry this key after a minute
AUTH_FAILURE_COOLDOWN_S = 24 * 3600  # bad key: park it for a day instead of hammering


@dataclass
class Response:
    text: str
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass
class StreamChunk:
    delta: str
    provider: str
    model: str
    done: bool = False
    usage: dict[str, int] | None = None
    raw: Any = None


def _retry_after(exc: openai.APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (ValueError, AttributeError):
        return None


class Rotor:
    """Routes chat completions across free-tier providers, rotating on quota errors.

    Providers are tried in priority order; a key that returns 429 is cooled down
    (honouring Retry-After when present, else until the provider's daily reset)
    and the next candidate is used transparently.
    """

    def __init__(
        self,
        keys: dict[str, list[str]],
        providers: tuple[ProviderConfig, ...] = PROVIDERS,
        state_file: Path | str | None = None,
        now: Callable[[], datetime] | None = None,
        client_factory: Callable[..., Any] | None = None,
        timeout: float = 120.0,
    ):
        if not keys:
            raise ConfigError("Rotor needs at least one provider key")
        self.providers = tuple(sorted(providers, key=lambda p: p.priority))
        self.keys = {p.name: keys[p.name] for p in self.providers if p.name in keys}
        if not self.keys:
            raise ConfigError("none of the supplied keys match a known provider")
        self._state_file = state_file
        self._now = now or (lambda: datetime.now(UTC))
        self._client_factory = client_factory or openai.OpenAI
        self._timeout = timeout
        self.tracker: QuotaTracker = load_state(state_file)

    @classmethod
    def from_env(cls, **kwargs: Any) -> Rotor:
        return cls(load_keys(), **kwargs)

    def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        tier: str = "fast",
        model: str | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """Send a chat completion, rotating to the next healthy key on quota errors."""
        if (prompt is None) == (messages is None):
            raise ValueError("pass exactly one of `prompt` or `messages`")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        statuses: dict[str, str] = {}
        resets: list[datetime] = []

        for cfg, key in self._candidates(statuses, resets):
            kid = key_id(cfg.name, key)
            now = self._now()
            try:
                result = self._call(
                    cfg, key, messages, model or cfg.model_for(tier), request_kwargs
                )
            except openai.RateLimitError as e:
                retry_s = _retry_after(e)
                until = (
                    now + timedelta(seconds=retry_s)
                    if retry_s
                    else next_reset(cfg, now)
                )
                self.tracker.mark_exhausted(kid, cfg, now, until)
                statuses[kid] = (
                    f"rate limited (429), cooling down until {until.isoformat()}"
                )
                resets.append(until)
            except openai.AuthenticationError:
                until = now + timedelta(seconds=AUTH_FAILURE_COOLDOWN_S)
                self.tracker.mark_exhausted(kid, cfg, now, until)
                statuses[kid] = "authentication failed (bad key?)"
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                until = now + timedelta(seconds=TRANSIENT_COOLDOWN_S)
                self.tracker.mark_exhausted(kid, cfg, now, until)
                statuses[kid] = (
                    f"transient error ({type(e).__name__}), retrying in {TRANSIENT_COOLDOWN_S}s"
                )
                resets.append(until)
            else:
                usage = getattr(result, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                self.tracker.record(kid, cfg, now, tokens=total_tokens)
                return Response(
                    text=result.choices[0].message.content or "",
                    provider=cfg.name,
                    model=result.model,
                    usage={
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0)
                        or 0,
                        "total_tokens": total_tokens,
                    },
                    raw=result,
                )
            finally:
                self._persist()

        raise AllProvidersExhausted(statuses, min(resets) if resets else None)

    async def chat_stream(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        tier: str = "fast",
        model: str | None = None,
        **request_kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion, rotating to the next healthy key on quota errors.

        Rotation only happens before the first chunk of a stream is yielded — once
        content has reached the caller, a mid-stream failure is raised as
        ``ProviderError`` rather than silently retried (that would duplicate output).
        """
        if (prompt is None) == (messages is None):
            raise ValueError("pass exactly one of `prompt` or `messages`")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        statuses: dict[str, str] = {}
        resets: list[datetime] = []

        for cfg, key in self._candidates(statuses, resets):
            kid = key_id(cfg.name, key)
            now = self._now()
            resolved_model = model or cfg.model_for(tier)
            try:
                stream = await self._call_stream(
                    cfg, key, messages, resolved_model, request_kwargs
                )
            except openai.RateLimitError as e:
                retry_s = _retry_after(e)
                until = (
                    now + timedelta(seconds=retry_s)
                    if retry_s
                    else next_reset(cfg, now)
                )
                self.tracker.mark_exhausted(kid, cfg, now, until)
                statuses[kid] = (
                    f"rate limited (429), cooling down until {until.isoformat()}"
                )
                resets.append(until)
                self._persist()
                continue
            except openai.AuthenticationError:
                until = now + timedelta(seconds=AUTH_FAILURE_COOLDOWN_S)
                self.tracker.mark_exhausted(kid, cfg, now, until)
                statuses[kid] = "authentication failed (bad key?)"
                self._persist()
                continue
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                until = now + timedelta(seconds=TRANSIENT_COOLDOWN_S)
                self.tracker.mark_exhausted(kid, cfg, now, until)
                statuses[kid] = (
                    f"transient error ({type(e).__name__}), retrying in {TRANSIENT_COOLDOWN_S}s"
                )
                resets.append(until)
                self._persist()
                continue

            total_tokens = 0
            try:
                async for event in stream:
                    if event.usage is not None:
                        total_tokens = event.usage.total_tokens or 0
                        yield StreamChunk(
                            delta="",
                            provider=cfg.name,
                            model=resolved_model,
                            done=True,
                            usage={
                                "prompt_tokens": event.usage.prompt_tokens or 0,
                                "completion_tokens": event.usage.completion_tokens
                                or 0,
                                "total_tokens": total_tokens,
                            },
                            raw=event,
                        )
                        continue
                    delta = event.choices[0].delta.content if event.choices else None
                    if delta:
                        yield StreamChunk(
                            delta=delta,
                            provider=cfg.name,
                            model=resolved_model,
                            raw=event,
                        )
            except Exception as e:
                raise ProviderError(cfg.name, "stream interrupted", e) from e
            finally:
                self.tracker.record(kid, cfg, now, tokens=total_tokens)
                self._persist()
            return

        raise AllProvidersExhausted(statuses, min(resets) if resets else None)

    def status(self) -> dict[str, str]:
        """Current health of every configured key: 'ok' or the reason it's skipped."""
        now = self._now()
        report = {}
        for cfg in self.providers:
            for key in self.keys.get(cfg.name, []):
                kid = key_id(cfg.name, key)
                report[kid] = self.tracker.status(kid, cfg, now) or "ok"
        return report

    def _candidates(self, statuses: dict[str, str], resets: list[datetime]):
        """Yield (provider, key) pairs that are healthy right now, priority order."""
        for cfg in self.providers:
            for key in self.keys.get(cfg.name, []):
                kid = key_id(cfg.name, key)
                now = self._now()
                reason = self.tracker.status(kid, cfg, now)
                if reason is not None:
                    statuses.setdefault(kid, reason)
                    resets.append(self.tracker.available_at(kid, cfg, now))
                    continue
                yield cfg, key

    def _call(
        self,
        cfg: ProviderConfig,
        key: str,
        messages: list[dict[str, str]],
        model: str,
        request_kwargs: dict[str, Any],
    ) -> Any:
        client = self._client_factory(
            api_key=key,
            base_url=cfg.base_url,
            timeout=self._timeout,
            default_headers=cfg.extra_headers or None,
            max_retries=0,  # the rotor owns retry behaviour
        )
        return client.chat.completions.create(
            model=model, messages=messages, **request_kwargs
        )

    async def _call_stream(
        self,
        cfg: ProviderConfig,
        key: str,
        messages: list[dict[str, str]],
        model: str,
        request_kwargs: dict[str, Any],
    ) -> Any:
        client = openai.AsyncOpenAI(
            api_key=key,
            base_url=cfg.base_url,
            timeout=self._timeout,
            default_headers=cfg.extra_headers or None,
            max_retries=0,  # the rotor owns retry behaviour
        )
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
        )

    def _persist(self) -> None:
        save_state(self.tracker, self._state_file)
