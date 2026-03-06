"""Tests for AI Cost Guard Python SDK."""

import time
from unittest.mock import MagicMock, patch

import pytest

from ai_cost_guard import AICostGuard


class TestAICostGuard:
    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            AICostGuard(api_key="")

    def test_init_warns_invalid_key_format(self):
        with pytest.warns(UserWarning, match="format may be invalid"):
            guard = AICostGuard(api_key="invalid_key", batch_events=False)
        guard.shutdown()

    def test_init_accepts_valid_key(self):
        guard = AICostGuard(api_key="acg_live_test123", batch_events=False)
        assert guard._config.api_key == "acg_live_test123"
        guard.shutdown()

    def test_init_accepts_test_key(self):
        guard = AICostGuard(api_key="acg_test_test123", batch_events=False)
        assert guard._config.api_key == "acg_test_test123"
        guard.shutdown()

    def test_default_config(self):
        guard = AICostGuard(api_key="acg_test_abc")
        assert guard._config.api_url == "https://api.aicostguard.com/api/v1"
        assert guard._config.batch_events is True
        assert guard._config.max_batch_size == 50
        assert guard._config.max_retries == 3
        assert guard._config.max_events_per_second == 100
        guard.shutdown()

    def test_custom_config(self):
        guard = AICostGuard(
            api_key="acg_test_abc",
            api_url="http://localhost:3001/api/v1",
            debug=True,
            batch_events=False,
            max_batch_size=10,
            max_retries=5,
            max_events_per_second=50,
            default_feature="test-app",
        )
        assert guard._config.api_url == "http://localhost:3001/api/v1"
        assert guard._config.debug is True
        assert guard._config.batch_events is False
        assert guard._config.max_batch_size == 10
        assert guard._config.max_retries == 5
        assert guard._config.max_events_per_second == 50
        assert guard._config.default_feature == "test-app"
        guard.shutdown()


class TestRateLimiter:
    def test_allows_events_within_limit(self):
        guard = AICostGuard(
            api_key="acg_test_rl",
            max_events_per_second=10,
            batch_events=False,
        )
        # Should allow 10 events quickly
        for _ in range(10):
            assert guard._consume_rate_token() is True
        guard.shutdown()

    def test_blocks_events_over_limit(self):
        guard = AICostGuard(
            api_key="acg_test_rl",
            max_events_per_second=5,
            batch_events=False,
        )
        # Consume all 5 tokens
        for _ in range(5):
            guard._consume_rate_token()
        # Next should be blocked
        assert guard._consume_rate_token() is False
        guard.shutdown()

    def test_refills_tokens_over_time(self):
        guard = AICostGuard(
            api_key="acg_test_rl",
            max_events_per_second=10,
            batch_events=False,
        )
        # Consume all tokens
        for _ in range(10):
            guard._consume_rate_token()
        assert guard._consume_rate_token() is False

        # Wait for refill (~200ms should give ~2 tokens)
        time.sleep(0.25)
        assert guard._consume_rate_token() is True
        guard.shutdown()


class TestTrackEvent:
    @patch.object(AICostGuard, "_request")
    def test_track_immediate(self, mock_request):
        mock_request.return_value = {"id": "evt_123"}
        guard = AICostGuard(api_key="acg_test_t", batch_events=False)

        result = guard.track(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
        )

        assert result is not None
        assert result.success is True
        mock_request.assert_called_once_with(
            "POST",
            "/track",
            {
                "provider": "openai",
                "model": "gpt-4",
                "inputTokens": 100,
                "outputTokens": 50,
                "success": True,
            },
        )
        guard.shutdown()

    def test_track_batched(self):
        guard = AICostGuard(api_key="acg_test_t", batch_events=True)
        result = guard.track(
            provider="openai", model="gpt-4", input_tokens=100, output_tokens=50
        )
        assert result is None  # Batched, no immediate response
        assert len(guard._buffer) == 1
        guard.shutdown()


class TestProviderHelpers:
    @patch.object(AICostGuard, "_request")
    def test_track_openai(self, mock_request):
        mock_request.return_value = {"id": "evt_1"}
        guard = AICostGuard(api_key="acg_test_p", batch_events=False)

        # Dict usage
        guard.track_openai(
            model="gpt-4",
            usage={"prompt_tokens": 200, "completion_tokens": 100},
        )
        call_body = mock_request.call_args[0][2]
        assert call_body["inputTokens"] == 200
        assert call_body["outputTokens"] == 100
        assert call_body["provider"] == "openai"
        guard.shutdown()

    @patch.object(AICostGuard, "_request")
    def test_track_openai_object(self, mock_request):
        mock_request.return_value = {"id": "evt_1"}
        guard = AICostGuard(api_key="acg_test_p", batch_events=False)

        # Object-style usage
        usage = MagicMock()
        usage.prompt_tokens = 300
        usage.completion_tokens = 150
        guard.track_openai(model="gpt-4o", usage=usage)
        call_body = mock_request.call_args[0][2]
        assert call_body["inputTokens"] == 300
        assert call_body["outputTokens"] == 150
        guard.shutdown()

    @patch.object(AICostGuard, "_request")
    def test_track_anthropic(self, mock_request):
        mock_request.return_value = {"id": "evt_1"}
        guard = AICostGuard(api_key="acg_test_p", batch_events=False)

        guard.track_anthropic(
            model="claude-3-opus",
            usage={"input_tokens": 400, "output_tokens": 200},
        )
        call_body = mock_request.call_args[0][2]
        assert call_body["inputTokens"] == 400
        assert call_body["outputTokens"] == 200
        assert call_body["provider"] == "anthropic"
        guard.shutdown()

    @patch.object(AICostGuard, "_request")
    def test_track_gemini(self, mock_request):
        mock_request.return_value = {"id": "evt_1"}
        guard = AICostGuard(api_key="acg_test_p", batch_events=False)

        guard.track_gemini(
            model="gemini-pro",
            usage={"promptTokenCount": 500, "candidatesTokenCount": 250},
        )
        call_body = mock_request.call_args[0][2]
        assert call_body["inputTokens"] == 500
        assert call_body["outputTokens"] == 250
        assert call_body["provider"] == "google"
        guard.shutdown()

    @patch.object(AICostGuard, "_request")
    def test_track_cohere(self, mock_request):
        mock_request.return_value = {"id": "evt_1"}
        guard = AICostGuard(api_key="acg_test_p", batch_events=False)

        guard.track_cohere(
            model="command-r-plus",
            usage={"input_tokens": 600, "output_tokens": 300},
        )
        call_body = mock_request.call_args[0][2]
        assert call_body["inputTokens"] == 600
        assert call_body["outputTokens"] == 300
        assert call_body["provider"] == "cohere"
        guard.shutdown()

    @patch.object(AICostGuard, "_request")
    def test_track_cohere_alt_fields(self, mock_request):
        mock_request.return_value = {"id": "evt_1"}
        guard = AICostGuard(api_key="acg_test_p", batch_events=False)

        guard.track_cohere(
            model="command-r",
            usage={"prompt_tokens": 700, "response_tokens": 350},
        )
        call_body = mock_request.call_args[0][2]
        assert call_body["inputTokens"] == 700
        assert call_body["outputTokens"] == 350
        guard.shutdown()


class TestFlushAndShutdown:
    @patch.object(AICostGuard, "_request")
    def test_flush_sends_batch(self, mock_request):
        mock_request.return_value = {"processed": 2}
        guard = AICostGuard(api_key="acg_test_f", batch_events=True)

        guard.track(provider="openai", model="gpt-4", input_tokens=100, output_tokens=50)
        guard.track(provider="anthropic", model="claude-3", input_tokens=200, output_tokens=100)

        assert len(guard._buffer) == 2
        guard.flush()
        assert len(guard._buffer) == 0
        mock_request.assert_called_once()
        guard.shutdown()

    def test_flush_empty_buffer_noop(self):
        guard = AICostGuard(api_key="acg_test_f", batch_events=True)
        guard.flush()  # Should not throw
        guard.shutdown()

    @patch.object(AICostGuard, "_request")
    def test_shutdown_flushes(self, mock_request):
        mock_request.return_value = {"processed": 1}
        guard = AICostGuard(api_key="acg_test_s", batch_events=True)

        guard.track(provider="openai", model="gpt-4", input_tokens=100, output_tokens=50)
        guard.shutdown()

        mock_request.assert_called_once()
        assert len(guard._buffer) == 0
