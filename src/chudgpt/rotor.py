"""The rotor: one chat() entry point that transparently rotates providers/keys."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import openai

from .config import PROVIDERS, ProviderConfig
from .errors import (
    AllProvidersExhausted,
    ConfigError,
    InvalidRequestError,
    ProviderError,
    StreamInterrupted,
)
from .keystore import key_id, load_keys
from .quota import QuotaTracker, next_reset
from .state import load_state, save_state

TRANSIENT_COOLDOWN_S = 60  # 5xx / network blip: retry this key after a minute
AUTH_FAILURE_COOLDOWN_S = 24 * 3600  # bad key: park it for a day instead of hammering


def _is_bad_key_400(exc: openai.BadRequestError) -> bool:
    """Whether a 400 is really an auth failure in disguise.

    Gemini answers an invalid API key with 400 INVALID_ARGUMENT rather than the
    401 the openai client maps to AuthenticationError, so the status code alone
    can't distinguish "your key is junk" (rotate to the next key) from "your
    request is malformed" (rotating won't help).
    """
    return "api key" in str(exc).lower()


@dataclass
class Response:
    text: str
    provider: str
    model: str
    key_id: str = (
        ""  # which key served it, e.g. "gemini:1e3ff2cc" — never the key itself
    )
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass
class StreamChunk:
    delta: str
    provider: str
    model: str
    key_id: str = (
        ""  # which key served it, e.g. "gemini:1e3ff2cc" — never the key itself
    )
    done: bool = False
    usage: dict[str, int] | None = None
    raw: Any = None


@dataclass
class KeyUsage:
    provider: str
    status: str  # "ok" or why it's currently skipped
    requests_today: int
    known_rpd: (
        int | None
    )  # the provider's known daily cap, if any — a hint, not a guarantee
    percent_used: (
        float | None
    )  # requests_today / known_rpd * 100, or None if known_rpd is unset
    tokens_today: int
    resets_at: datetime


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
        messages: list[dict[str, Any]] | None = None,
        tier: str = "fast",
        model: str | None = None,
        providers: Sequence[str] | None = None,
        **request_kwargs: Any,
    ) -> Response:
        """Send a chat completion, rotating to the next healthy key on quota errors."""
        if (prompt is None) == (messages is None):
            raise InvalidRequestError("pass exactly one of `prompt` or `messages`")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        statuses: dict[str, str] = {}
        resets: list[datetime] = []

        for cfg, key in self._candidates(statuses, resets, only=providers):
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
            except openai.NotFoundError as e:
                # pinned a model this provider doesn't serve — a caller error that
                # rotating cannot fix
                raise InvalidRequestError(f"[{cfg.name}] {e}") from e
            except openai.BadRequestError as e:
                if not _is_bad_key_400(e):
                    # malformed request (bad kwarg, bad message shape): every
                    # provider will reject it identically, so fail loudly instead
                    # of burning the whole key pool on a caller error.
                    raise InvalidRequestError(f"[{cfg.name}] {e}") from e
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
            except openai.OpenAIError as e:
                # anything else the client can raise stays inside the ChudGPTError
                # hierarchy rather than leaking a raw openai exception to callers
                raise ProviderError(cfg.name, str(e), e) from e
            else:
                usage = getattr(result, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                self.tracker.record(kid, cfg, now, tokens=total_tokens)
                return Response(
                    text=result.choices[0].message.content or "",
                    provider=cfg.name,
                    model=result.model,
                    key_id=kid,
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
        messages: list[dict[str, Any]] | None = None,
        tier: str = "fast",
        model: str | None = None,
        providers: Sequence[str] | None = None,
        **request_kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion, rotating to the next healthy key on quota errors.

        Rotation only happens before the first chunk of a stream is yielded — once
        content has reached the caller, a mid-stream failure is raised as
        ``StreamInterrupted`` rather than silently retried (would duplicate output).
        """
        if (prompt is None) == (messages is None):
            raise InvalidRequestError("pass exactly one of `prompt` or `messages`")
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        statuses: dict[str, str] = {}
        resets: list[datetime] = []

        for cfg, key in self._candidates(statuses, resets, only=providers):
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
            except openai.NotFoundError as e:
                # pinned a model this provider doesn't serve — a caller error that
                # rotating cannot fix
                raise InvalidRequestError(f"[{cfg.name}] {e}") from e
            except openai.BadRequestError as e:
                if not _is_bad_key_400(e):
                    # malformed request (bad kwarg, bad message shape): every
                    # provider will reject it identically, so fail loudly instead
                    # of burning the whole key pool on a caller error.
                    raise InvalidRequestError(f"[{cfg.name}] {e}") from e
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
            except openai.OpenAIError as e:
                # anything else the client can raise stays inside the ChudGPTError
                # hierarchy rather than leaking a raw openai exception to callers
                raise ProviderError(cfg.name, str(e), e) from e

            total_tokens = 0
            usage: dict[str, int] | None = None
            last_event: Any = None
            try:
                async for event in stream:
                    # usage is NOT an end-of-stream marker: OpenAI sends it once in a
                    # final choices-less chunk, but Gemini attaches it to every chunk
                    # — including ones carrying text. Record it and keep going, or
                    # content gets swallowed.
                    if event.usage is not None:
                        total_tokens = event.usage.total_tokens or 0
                        usage = {
                            "prompt_tokens": event.usage.prompt_tokens or 0,
                            "completion_tokens": event.usage.completion_tokens or 0,
                            "total_tokens": total_tokens,
                        }
                    last_event = event
                    delta = event.choices[0].delta.content if event.choices else None
                    if delta:
                        yield StreamChunk(
                            delta=delta,
                            provider=cfg.name,
                            model=resolved_model,
                            key_id=kid,
                            raw=event,
                        )
            except Exception as e:
                raise StreamInterrupted(cfg.name, e) from e
            finally:
                self.tracker.record(kid, cfg, now, tokens=total_tokens)
                self._persist()
            # exactly one terminal chunk, once the stream is actually done
            yield StreamChunk(
                delta="",
                provider=cfg.name,
                model=resolved_model,
                key_id=kid,
                done=True,
                usage=usage,
                raw=last_event,
            )
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

    def usage(self) -> dict[str, KeyUsage]:
        """Today's request/token count per key, and % of the known daily cap used.

        ``percent_used`` is only as good as ``known_rpd`` (a proactive-skip hint,
        not a guarantee from the provider) — free-tier limits drift, so treat this
        as an estimate, not an authoritative quota dashboard.
        """
        now = self._now()
        report: dict[str, KeyUsage] = {}
        for cfg in self.providers:
            for key in self.keys.get(cfg.name, []):
                kid = key_id(cfg.name, key)
                count, tokens = self.tracker.counts(kid, cfg, now)
                percent = (
                    round(100 * count / cfg.known_rpd, 1) if cfg.known_rpd else None
                )
                report[kid] = KeyUsage(
                    provider=cfg.name,
                    status=self.tracker.status(kid, cfg, now) or "ok",
                    requests_today=count,
                    known_rpd=cfg.known_rpd,
                    percent_used=percent,
                    tokens_today=tokens,
                    resets_at=self.tracker.available_at(kid, cfg, now),
                )
        return report

    def _candidates(
        self,
        statuses: dict[str, str],
        resets: list[datetime],
        only: Sequence[str] | None = None,
    ):
        """Yield (provider, key) pairs that are healthy right now, priority order.

        ``only`` narrows to a subset of provider names (e.g. the audio-capable ones
        for a transcription call); None considers every configured provider."""
        allowed = set(only) if only is not None else None
        for cfg in self.providers:
            if allowed is not None and cfg.name not in allowed:
                continue
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
        messages: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
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
