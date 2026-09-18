"""
Pydantic Schemas for the automatic Daily Report feature (Phase 11 -- replaces
Phase 10's on-demand "Yesterday's Work" summary).

`StructuredSnapshot` is the contract with the caller (the OnTask Next.js backend):
it is the already-computed, ground-truth dataset for one workspace's rolling
24-hour reporting window (report_start -> report_end, NOT a calendar day),
aggregated from `task_events` / `task_time_entries` / `workspace_invitations` /
`workspace_members` there (see
supabase/migrations/0018_automatic_daily_reports.sql). This service never
touches a database and never invents facts -- it only narrates the snapshot
it is handed (per project_document/update-ai.md's "Source of Truth Rules").

This is intentionally more than a time/status report: `members[].events` is
the window's actual activity log (created/assigned/started/completed/...),
and `workspace_changes` covers invitations and membership changes. The
frontend renders that structured data directly (task titles, bullet lists,
hierarchy) -- the model's only job is the prose layered on top
(`SummaryNarrative`).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from src.app.schemas.common import ProviderStatusEventSchema, UsageInfo


class TaskStatus(str, Enum):
    completed = "completed"
    in_progress = "in_progress"
    skipped = "skipped"


class MemberEventEntry(BaseModel):
    """
    One raw, factual thing a member did within the reporting window (task_events row, or a
    synthesized 'invitation_sent' entry under the inviter -- see the SQL
    aggregator). `type` is one of task_events.event_type's values, or
    'invitation_sent'. `metadata` carries whatever extra fields that type
    needs (from/to for assignment or progress changes, invited_email/status
    for invitations, ...) -- forwarded as-is from the database, never
    reinterpreted here.
    """

    type: str
    timestamp: datetime
    task_id: Optional[str] = None
    task_title: Optional[str] = None
    parent_title: Optional[str] = Field(
        default=None, description="Set when task_title is a subtask; None for a standalone/parent task"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskActivityEntry(BaseModel):
    """A single task (or subtask) a member worked on, within one reporting window's snapshot."""

    task_id: str
    title: str
    parent_task_id: Optional[str] = None
    parent_title: Optional[str] = Field(
        default=None, description="Set when this entry is a subtask; None for a standalone/parent task"
    )
    focused_seconds: int = Field(ge=0)
    progress_start: Optional[int] = Field(
        default=None, ge=0, le=100, description="Only set if a progress_changed event occurred within the reporting window"
    )
    progress_end: Optional[int] = Field(default=None, ge=0, le=100)
    status_end: TaskStatus


class MemberEntry(BaseModel):
    """One workspace member's activity within the reporting window."""

    user_id: str
    display_name: str
    focused_seconds: int = Field(ge=0)
    events: List[MemberEventEntry] = Field(default_factory=list)
    task_activity: List[TaskActivityEntry] = Field(default_factory=list)


class InvitationActivityEntry(BaseModel):
    """An invitation sent and/or responded to within the reporting window.
    `rejection_reason` is deliberately never included: OnTask restricts it to
    the workspace owner's view, but this summary is visible to every member."""

    invited_email: str
    invited_by_user_id: str
    invited_by_name: str
    status: str
    responded_at: Optional[datetime] = None


class MemberChangeEntry(BaseModel):
    user_id: str
    display_name: str


class WorkspaceChanges(BaseModel):
    """Workspace-level (not task-level) activity within the reporting window."""

    invitations: List[InvitationActivityEntry] = Field(default_factory=list)
    members_joined: List[MemberChangeEntry] = Field(default_factory=list)
    members_removed: List[MemberChangeEntry] = Field(default_factory=list)
    tasks_created: int = Field(default=0, ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_skipped: int = Field(default=0, ge=0)
    tasks_deleted: int = Field(default=0, ge=0)

    @property
    def has_any(self) -> bool:
        return bool(
            self.invitations
            or self.members_joined
            or self.members_removed
            or self.tasks_created
            or self.tasks_completed
            or self.tasks_skipped
            or self.tasks_deleted
        )


class StructuredSnapshot(BaseModel):
    """
    The full ground-truth dataset for one `(workspace_id, report_end)` pair --
    this is what `workspace_daily_summaries.structured_snapshot` stores per 10.2.4.
    Every fact shown to the user must be traceable to this object.

    `report_start`/`report_end` are the exact rolling-24h window boundaries
    (report_start == report_end - 24h) -- NOT a calendar day. The caller
    (OnTask's Next.js backend) has already resolved these to absolute instants
    before calling this service; this service never computes or reinterprets
    them, it only narrates what happened inside that window.
    """

    workspace_id: str
    workspace_name: str
    report_start: datetime
    report_end: datetime
    timezone: str
    total_focused_seconds: int = Field(ge=0)
    members: List[MemberEntry] = Field(default_factory=list)
    workspace_changes: WorkspaceChanges = Field(default_factory=WorkspaceChanges)

    @property
    def has_activity(self) -> bool:
        """True if there is ANYTHING worth narrating -- not just recorded time.
        A window with only a sent invitation (no timer use at all) still deserves a
        real report, not the "no activity recorded" short-circuit."""
        return (
            self.total_focused_seconds > 0
            or any(m.events or m.task_activity for m in self.members)
            or self.workspace_changes.has_any
        )

    @property
    def known_member_ids(self) -> Set[str]:
        return {m.user_id for m in self.members}

    @property
    def known_task_ids(self) -> Set[str]:
        ids: Set[str] = set()
        for m in self.members:
            ids.update(t.task_id for t in m.task_activity)
            ids.update(e.task_id for e in m.events if e.task_id)
        return ids


class SummaryGenerateRequest(BaseModel):
    """Request body for POST /api/summary/generate."""

    snapshot: StructuredSnapshot


class MemberNarrative(BaseModel):
    """AI-authored narrative paragraph attached to one already-known member,
    grounded only in that member's `events`/`task_activity` from the input snapshot."""

    user_id: str = Field(..., description="Must match a member's user_id from the input snapshot")
    note: str = Field(
        ...,
        description=(
            "A few sentences narrating this member's activity during the reporting window: what they created, worked on, "
            "completed, were assigned, assigned to others, and any invitations they sent -- "
            "in a sensible chronological order when the sequence matters. Neutral, factual tone."
        ),
    )


class SummaryNarrative(BaseModel):
    """
    The ONLY thing the model is allowed to produce: prose narration layered on top
    of the structured snapshot. It must never carry new facts (times, names,
    statuses, counts) -- those always come from StructuredSnapshot, which the
    caller renders independently. This is also the LangChain structured-output
    target schema passed to LLMGateway.generate_structured().

    Deliberately NOT included here: completed_work[]/in_progress[]/skipped_work[]
    groupings -- those are 100% mechanically derivable from
    members[].task_activity[].status_end, so they're computed by the caller
    directly from the snapshot rather than risking the model re-deriving (and
    possibly miscounting) them.
    """

    overall_summary: str = Field(
        ..., description="2-4 sentence neutral overview of the workspace's activity during the reporting window"
    )
    members: List[MemberNarrative] = Field(default_factory=list)
    workspace_changes_summary: str = Field(
        default="",
        description=(
            "1-2 neutral sentences on workspace-level changes (invitations sent/answered, members "
            "joining/leaving) if any occurred within the reporting window; empty string if workspace_changes was empty"
        ),
    )
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
