"""
Pydantic Schemas for the Shared AI "Yesterday's Work" Summary feature (Phase 10).

`StructuredSnapshot` is the contract with the caller (the OnTask Next.js backend):
it is the already-computed, ground-truth dataset for one workspace-day, aggregated
from `task_time_entries` there. This service never touches a database and never
invents facts -- it only narrates the snapshot it is handed (see 10.2 and 10.4 of
project_document/ontask-evolution-plan.md).
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Set

from pydantic import BaseModel, Field

from src.app.schemas.common import ProviderStatusEventSchema, UsageInfo


class TaskStatus(str, Enum):
    completed = "completed"
    in_progress = "in_progress"
    skipped = "skipped"


class TaskEntry(BaseModel):
    """A single task (or subtask) a member worked on, within one day's snapshot."""

    task_id: str
    name: str
    parent_task_id: Optional[str] = Field(
        default=None, description="Set when this entry is a subtask; None for a standalone/parent task"
    )
    status: TaskStatus
    focused_seconds: int = Field(ge=0)


class MemberEntry(BaseModel):
    """One workspace member's activity for the day."""

    user_id: str
    display_name: str
    focused_seconds: int = Field(ge=0)
    tasks: List[TaskEntry] = Field(default_factory=list)


class StructuredSnapshot(BaseModel):
    """
    The full ground-truth dataset for one `(workspace_id, summary_date)` pair --
    this is what `workspace_daily_summaries.structured_snapshot` stores per 10.2.4.
    Every fact shown to the user must be traceable to this object.
    """

    workspace_id: str
    workspace_name: str
    summary_date: date
    timezone: str
    total_focused_seconds: int = Field(ge=0)
    members: List[MemberEntry] = Field(default_factory=list)

    @property
    def has_activity(self) -> bool:
        return self.total_focused_seconds > 0 and any(m.tasks for m in self.members)

    @property
    def known_member_ids(self) -> Set[str]:
        return {m.user_id for m in self.members}

    @property
    def known_task_ids(self) -> Set[str]:
        return {t.task_id for m in self.members for t in m.tasks}


class SummaryGenerateRequest(BaseModel):
    """Request body for POST /api/summary/generate."""

    snapshot: StructuredSnapshot


class MemberNote(BaseModel):
    """AI-authored narrative blurb attached to one already-known member."""

    user_id: str = Field(..., description="Must match a member's user_id from the input snapshot")
    note: str = Field(..., description="1-2 sentence neutral, factual narrative for this member's day")


class SummaryNarrative(BaseModel):
    """
    The ONLY thing the model is allowed to produce: prose narration layered on top
    of the structured snapshot. It must never carry new facts (times, names,
    statuses, counts) -- those always come from StructuredSnapshot, which the
    caller renders independently. This is also the LangChain structured-output
    target schema passed to LLMGateway.generate_structured().
    """

    overall_summary: str = Field(..., description="2-4 sentence neutral overview of the workspace's day")
    member_notes: List[MemberNote] = Field(default_factory=list)
    highlights: List[str] = Field(
        default_factory=list,
        description="Short factual bullet highlights (e.g. notable completions); no opinions",
    )


class GenerationMetadata(BaseModel):
    """Traceability info for one generated summary (Phase 10.4: 'store model +
    generation_metadata per summary')."""

    provider: str
    model: str
    generated_at: datetime
    usage: UsageInfo = Field(default_factory=UsageInfo)
    status_events: List[ProviderStatusEventSchema] = Field(default_factory=list)
    used_fallback_template: bool = Field(
        default=False,
        description="True when the AI narrative failed validation twice (or the gateway "
        "was fully exhausted) and a deterministic, non-AI narrative was returned instead",
    )
    validation_warnings: List[str] = Field(default_factory=list)


class SummaryGenerateResponse(BaseModel):
    """Response body for POST /api/summary/generate."""

    snapshot: StructuredSnapshot
    narrative: SummaryNarrative
    meta: GenerationMetadata
