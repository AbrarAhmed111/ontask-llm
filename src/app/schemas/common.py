"""
Pydantic Schemas shared across every LLM Gateway consumer.
Kept separate from any single feature's schemas (e.g. summary.py) since future
AI features in this service will reuse the same usage/status-event shape.
"""

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ProviderStatusEventSchema(BaseModel):
    """Structured status event emitted when a provider switches or fails over."""
    type: str = "provider_status"
    status: str = Field(..., description="Event status: 'fallback' or 'switched'")
    message: str = Field(..., description="User-facing status message")
    provider: str = Field(..., description="Name of the provider deployment")
