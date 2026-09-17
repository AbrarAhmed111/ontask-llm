"""
Daily Summary Endpoints (Phase 10 -- Shared "Yesterday's Work" Summary).
"""

from fastapi import APIRouter

from src.app.schemas.summary import SummaryGenerateRequest, SummaryGenerateResponse
from src.app.services import summary_service

router = APIRouter(prefix="/summary", tags=["Summary"])


@router.post(
    "/generate",
    response_model=SummaryGenerateResponse,
    summary="Narrate a workspace-day's structured activity snapshot",
)
async def generate_summary(request: SummaryGenerateRequest) -> SummaryGenerateResponse:
    """
    Accepts a `StructuredSnapshot` already computed by the caller (aggregated from
    `task_time_entries` for one workspace + day) and returns a validated AI narrative
    layered on top of it. This service is stateless: it does not persist anything --
    idempotency, `(workspace_id, summary_date)` uniqueness, and regeneration semantics
    (10.3) are owned by the caller. Never raises for provider failures: when every
    configured provider is exhausted, `generate_summary` degrades to the deterministic
    template narrative rather than erroring out (see summary_service.py).
    """
    return await summary_service.generate_summary(request.snapshot)
