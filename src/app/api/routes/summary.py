"""
Daily Report Endpoints (Phase 11 -- automatic rolling-24h Daily Report).
"""

from fastapi import APIRouter

from src.app.schemas.summary import SummaryGenerateRequest, SummaryGenerateResponse
from src.app.services import summary_service

router = APIRouter(prefix="/summary", tags=["Summary"])


@router.post(
    "/generate",
    response_model=SummaryGenerateResponse,
    summary="Narrate a workspace's structured activity snapshot for one reporting window",
)
async def generate_summary(request: SummaryGenerateRequest) -> SummaryGenerateResponse:
    """
    Accepts a `StructuredSnapshot` already computed by the caller (aggregated from
    `task_time_entries` for one workspace's rolling 24h reporting window) and returns a
    validated AI narrative layered on top of it. This service is stateless: it does not
    persist anything -- idempotency, `(workspace_id, report_end)` uniqueness, and
    regeneration semantics are owned by the caller. Never raises for provider failures:
    when every configured provider is exhausted, `generate_summary` degrades to the
    deterministic template narrative rather than erroring out (see summary_service.py).
    """
    return await summary_service.generate_summary(request.snapshot)
