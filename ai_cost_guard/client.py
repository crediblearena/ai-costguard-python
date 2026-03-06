"""AI Cost Guard client implementation."""

import json
import logging
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

import requests

from ai_cost_guard.types import AICostGuardConfig, TrackEventParams, TrackResponse

logger = logging.getLogger("ai_cost_guard")


class AICostGuard:
    """Client for AI Cost Guard — track, analyze, and optimize AI/LLM API costs.

    Example::

        from ai_cost_guard import AICostGuard

        guard = AICostGuard(api_key="acg_live_xxx")
        guard.track(provider="openai", model="gpt-4", input_tokens=500, output_tokens=150)
        guard.shutdown()
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.aicostguard.com/api/v1",
        debug: bool = False,
        batch_events: bool = True,
        batch_interval_ms: int = 5000,
        max_batch_size: int = 50,
        max_retries: int = 3,
        max_events_per_second: int = 100,
        default_feature: str = "",
    ):
        if not api_key:
            raise ValueError("AI Cost Guard: api_key is required")

        if not api_key.startswith("acg_live_") and not api_key.startswith("acg_test_"):
            warnings.warn(
                "AI Cost Guard: API key format may be invalid. "
                "Expected prefix: acg_live_ or acg_test_",
                UserWarning,
                stacklevel=2,
            )

        self._config = AICostGuardConfig(
            api_key=api_key,
            api_url=api_url.rstrip("/"),
            debug=debug,
            batch_events=batch_events,
            batch_interval_ms=batch_interval_ms,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
            max_events_per_second=max_events_per_second,
            default_feature=default_feature,
        )

        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._is_shutting_down = False

        # Token-bucket rate limiter
        self._rate_limit_max = max_events_per_second
        self._rate_limit_tokens = float(max_events_per_second)
        self._rate_limit_last_refill = time.monotonic()
        self._rate_lock = threading.Lock()

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "User-Agent": "ai-cost-guard-python-sdk/1.0.0",
            }
        )

        # Start batch timer
        self._timer: Optional[threading.Timer] = None
        if batch_events:
            self._start_batch_timer()

        self._log("AI Cost Guard Python SDK initialized")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def track(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: Optional[float] = None,
        cost: Optional[float] = None,
        success: bool = True,
        feature: Optional[str] = None,
        user_id: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackResponse]:
        """Track an AI API usage event.

        Args:
            provider: AI provider name (e.g. "openai", "anthropic")
            model: Model identifier (e.g. "gpt-4", "claude-3-opus")
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            latency_ms: Request latency in milliseconds
            cost: Explicit cost override (auto-calculated if omitted)
            success: Whether the API call succeeded
            feature: Feature/module tag
            user_id: End-user identifier
            prompt_hash: Hash of the prompt for grouping
            prompt_name: Human-readable prompt label
            metadata: Arbitrary key-value metadata

        Returns:
            TrackResponse if sent immediately, None if batched.
        """
        # Rate limiting
        if not self._consume_rate_token():
            self._log(
                f"Rate limit exceeded ({self._config.max_events_per_second}/sec). Event dropped."
            )
            return None

        event: Dict[str, Any] = {
            "provider": provider.lower(),
            "model": model.lower(),
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "success": success,
        }

        if latency_ms is not None:
            event["latencyMs"] = latency_ms
        if cost is not None:
            event["cost"] = cost
        if feature or self._config.default_feature:
            event["feature"] = feature or self._config.default_feature
        if user_id:
            event["userId"] = user_id
        if prompt_hash:
            event["promptHash"] = prompt_hash
        if prompt_name:
            event["promptName"] = prompt_name
        if metadata:
            event["metadata"] = metadata

        if self._config.batch_events:
            with self._buffer_lock:
                self._buffer.append(event)
                buf_len = len(self._buffer)
            self._log(f"Event buffered ({buf_len}/{self._config.max_batch_size})")

            if buf_len >= self._config.max_batch_size:
                self.flush()
            return None

        # Send immediately
        return self._send_event(event)

    def flush(self) -> None:
        """Flush all buffered events to the API."""
        with self._buffer_lock:
            if not self._buffer:
                return
            events = list(self._buffer)
            self._buffer.clear()

        self._log(f"Flushing {len(events)} events")

        try:
            self._send_batch(events)
        except Exception as exc:
            if not self._is_shutting_down:
                with self._buffer_lock:
                    self._buffer = events + self._buffer
            self._log(f"Flush failed: {exc}")

    def shutdown(self) -> None:
        """Gracefully shut down the SDK, flushing pending events."""
        self._is_shutting_down = True
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self.flush()
        self._session.close()
        self._log("SDK shut down")

    # ------------------------------------------------------------------ #
    # Provider helpers
    # ------------------------------------------------------------------ #

    def track_openai(
        self,
        model: str,
        usage: Any,
        latency_ms: Optional[float] = None,
        feature: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackResponse]:
        """Track an OpenAI API call.

        Args:
            model: Model name (e.g. "gpt-4")
            usage: OpenAI usage object with ``prompt_tokens`` and ``completion_tokens``
            latency_ms: Request latency in milliseconds
            feature: Feature tag
            user_id: End-user ID
            metadata: Extra metadata
        """
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or (
            usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        )
        completion_tokens = getattr(usage, "completion_tokens", 0) or (
            usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        )
        return self.track(
            provider="openai",
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            latency_ms=latency_ms,
            feature=feature,
            user_id=user_id,
            metadata=metadata,
        )

    def track_anthropic(
        self,
        model: str,
        usage: Any,
        latency_ms: Optional[float] = None,
        feature: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackResponse]:
        """Track an Anthropic API call.

        Args:
            model: Model name (e.g. "claude-3-opus-20240229")
            usage: Anthropic usage object with ``input_tokens`` and ``output_tokens``
        """
        input_tokens = getattr(usage, "input_tokens", 0) or (
            usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
        )
        output_tokens = getattr(usage, "output_tokens", 0) or (
            usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
        )
        return self.track(
            provider="anthropic",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            feature=feature,
            user_id=user_id,
            metadata=metadata,
        )

    def track_gemini(
        self,
        model: str,
        usage: Any,
        latency_ms: Optional[float] = None,
        feature: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackResponse]:
        """Track a Google Gemini API call.

        Args:
            model: Model name (e.g. "gemini-pro")
            usage: Gemini usage_metadata with ``promptTokenCount`` / ``candidatesTokenCount``
        """
        prompt_tokens = getattr(usage, "promptTokenCount", 0) or getattr(
            usage, "prompt_token_count", 0
        ) or (
            usage.get("promptTokenCount", usage.get("prompt_token_count", 0))
            if isinstance(usage, dict)
            else 0
        )
        candidates_tokens = getattr(usage, "candidatesTokenCount", 0) or getattr(
            usage, "candidates_token_count", 0
        ) or (
            usage.get("candidatesTokenCount", usage.get("candidates_token_count", 0))
            if isinstance(usage, dict)
            else 0
        )
        return self.track(
            provider="google",
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=candidates_tokens,
            latency_ms=latency_ms,
            feature=feature,
            user_id=user_id,
            metadata=metadata,
        )

    def track_cohere(
        self,
        model: str,
        usage: Any,
        latency_ms: Optional[float] = None,
        feature: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackResponse]:
        """Track a Cohere API call.

        Args:
            model: Model name (e.g. "command-r-plus")
            usage: Cohere usage/billed_units with ``input_tokens``/``output_tokens``
                   or ``prompt_tokens``/``response_tokens``
        """
        input_tokens = (
            getattr(usage, "input_tokens", 0)
            or getattr(usage, "prompt_tokens", 0)
            or (
                usage.get("input_tokens", usage.get("prompt_tokens", 0))
                if isinstance(usage, dict)
                else 0
            )
        )
        output_tokens = (
            getattr(usage, "output_tokens", 0)
            or getattr(usage, "response_tokens", 0)
            or (
                usage.get("output_tokens", usage.get("response_tokens", 0))
                if isinstance(usage, dict)
                else 0
            )
        )
        return self.track(
            provider="cohere",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            feature=feature,
            user_id=user_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _consume_rate_token(self) -> bool:
        """Token-bucket rate limiter. Returns True if event is allowed."""
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._rate_limit_last_refill
            self._rate_limit_tokens = min(
                self._rate_limit_max,
                self._rate_limit_tokens + elapsed * self._rate_limit_max,
            )
            self._rate_limit_last_refill = now

            if self._rate_limit_tokens >= 1:
                self._rate_limit_tokens -= 1
                return True
            return False

    def _send_event(self, event: Dict[str, Any]) -> TrackResponse:
        """Send a single event to the API."""
        data = self._request("POST", "/track", event)
        return TrackResponse(success=True, data=data)

    def _send_batch(self, events: List[Dict[str, Any]]) -> TrackResponse:
        """Send a batch of events to the API."""
        data = self._request("POST", "/track/batch", {"events": events})
        return TrackResponse(success=True, data=data)

    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make an HTTP request with retries and exponential backoff."""
        url = f"{self._config.api_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    json=body,
                    timeout=10,
                )
                resp.raise_for_status()
                result = resp.json()
                return result.get("data", result)
            except Exception as exc:
                last_error = exc
                self._log(
                    f"Request failed (attempt {attempt}/{self._config.max_retries}): {exc}"
                )
                if attempt < self._config.max_retries:
                    time.sleep(2**attempt)

        raise last_error or RuntimeError("Request failed after all retries")

    def _start_batch_timer(self) -> None:
        """Start the periodic batch flush timer."""

        def _tick() -> None:
            if self._is_shutting_down:
                return
            try:
                self.flush()
            except Exception as exc:
                self._log(f"Auto-flush error: {exc}")
            finally:
                if not self._is_shutting_down:
                    self._start_batch_timer()

        interval = self._config.batch_interval_ms / 1000.0
        self._timer = threading.Timer(interval, _tick)
        self._timer.daemon = True
        self._timer.start()

    def _log(self, message: str) -> None:
        if self._config.debug:
            logger.debug(f"[AI Cost Guard] {message}")
            print(f"[AI Cost Guard] {message}")
