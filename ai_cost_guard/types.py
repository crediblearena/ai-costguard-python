"""Type definitions for AI Cost Guard SDK."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrackEventParams:
    """Parameters for tracking an AI usage event."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: Optional[float] = None
    cost: Optional[float] = None
    success: bool = True
    feature: Optional[str] = None
    user_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    prompt_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrackResponse:
    """Response from the tracking API."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AICostGuardConfig:
    """Configuration for the AI Cost Guard client."""

    api_key: str
    api_url: str = "https://api.aicostguard.com/api/v1"
    debug: bool = False
    batch_events: bool = True
    batch_interval_ms: int = 5000
    max_batch_size: int = 50
    max_retries: int = 3
    max_events_per_second: int = 100
    default_feature: str = ""
