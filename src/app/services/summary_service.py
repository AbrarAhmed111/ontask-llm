"""
AI Daily Report Service (Phase 11 -- automatic rolling-24h Daily Report,
formerly Phase 10's on-demand "Yesterday's Work" summary).

Structured-first, AI-narrates-only (see project_document/update-ai.md's "Source
of Truth Rules" and 10.2/10.4 of project_document/ontask-evolution-plan.md):
- Every hard fact (events, times, names, task statuses, counts, invitations,
  progress values) is already computed by the caller and lives in
  `StructuredSnapshot`. This service never touches a database and never invents
  a number or an event -- it only asks the model to narrate what's already
  there, then validates that the narration didn't smuggle in anything new
  before handing it back.
- No recorded activity at all (not just no time -- also no events, no
  invitations, no membership changes) -> short-circuit to a deterministic
  sentence without calling the AI at all.
- If the AI's narrative keeps referencing unknown members or untraceable
  numbers after one corrective retry, fall back to a deterministic,
  template-built narrative rather than surfacing a broken or fabricated
  summary.

completed_work/in_progress/skipped_work groupings and the raw activity bullet
list the OnTask UI shows per member are NOT produced here -- they're 100%
derivable from `StructuredSnapshot.members[].task_activity`/`events` directly,
so the frontend renders them straight from the snapshot. This service's only
output is prose: an overall summary, a short per-member narrative paragraph,
and (when relevant) a workspace-changes summary.
"""

import logging
import re
from datetime import datetime, timezone as dt_timezone
from typing import List, Set

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.core.config import get_settings
from src.app.gateway import LLMGateway, ProviderStatusEvent
from src.app.schemas.common import ProviderStatusEventSchema, UsageInfo
from src.app.schemas.summary import (
    GenerationMetadata,
    MemberEntry,
    MemberNarrative,
    StructuredSnapshot,
    SummaryGenerateResponse,
    SummaryNarrative,
    TaskStatus,
)

logger = logging.getLogger("SummaryService")
settings = get_settings()

# Initialize the shared LLM Gateway instance (multi-provider, automatic failover).
gateway = LLMGateway(
    max_attempts=settings.GATEWAY_MAX_ATTEMPTS,
    cooldown_seconds=settings.GATEWAY_COOLDOWN_SECONDS,
)

SYSTEM_PROMPT = """You are a neutral reporting assistant for OnTask, a team focus-time tracker.

You will be given a STRUCTURED JSON snapshot of everything that happened in a workspace during \
its reporting window (report_start to report_end, a rolling 24-hour period, not necessarily a \
calendar day): what each member created, worked on, was assigned, completed, skipped, and any \
invitations sent or membership changes. Your job is to turn this into a factual work narrative \
-- "here's what happened in our workspace during the previous 24 hours" -- NOT a time report and \
NOT a productivity evaluation. Never call this window "yesterday" or "today" -- refer to it as \
"the reporting period" or "the previous 24 hours", since it does not align to calendar days.

What to cover, per member (in `members[].note`):
- What they created, worked on, completed, or skipped -- named by title.
- What they were assigned, and what they assigned to others (by name).
- Any invitations they sent, including the invitation's current status (still pending / accepted
  / rejected -- never invent or guess a status, and never mention a rejection reason).
- How task progress changed, ONLY when `task_activity[].progress_start`/`progress_end` are both
  present for that task -- e.g. "progress moved from 20% to 80%". If only one of those two values
  is present (or neither), do not state a progress change for that task at all.
- When a member's `events` show a clear order (e.g. created, then started, then assigned), prefer
  a natural chronological sentence ("created X, worked on it, then assigned it to Y") over an
  unordered list of facts.
- A subtask (an event or task_activity entry with a non-null `parent_title`) should always be
  described with its parent, e.g. "worked on **Authentication** under **School Management MVP**" --
  never just the subtask name alone.
- If a member has no events and no task_activity, say so plainly (e.g. "no recorded activity")
  instead of padding or omitting them.

Workspace-level activity (in `workspace_changes_summary`): only write this when
`workspace_changes` in the snapshot is non-empty (has any invitations, joins, removals, or task
counts) -- 1-2 sentences covering what changed, e.g. task creation/completion counts, members
joining/leaving. Leave `workspace_changes_summary` as an empty string when there is nothing to
report there, even if members[] has activity.

Hard rules (violating any of these makes the narrative unusable):
1. Never invent an event, task, member, time value, or progress value.
2. Never state a fact (a name, a number, a status, an invitation outcome) that is not present in
   the snapshot.
3. Never attribute work to anyone other than the member whose `events`/`task_activity` it appears
   under -- the snapshot already resolved historical attribution correctly (it reflects who
   actually did the work during the reporting window, not who a task is currently assigned to);
   do not second-guess or "correct" it using outside assumptions.
4. Never infer that a task was completed unless a task_activity entry's status_end is
   "completed" (or an event of type "completed" is present) -- recorded time, high progress, or
   reaching the planned duration are NOT completion.
5. Preserve parent/subtask relationships (see the subtask rule above).
6. Only summarize information present in the supplied snapshot -- if something is not there,
   omit it rather than guessing or padding.
7. Use neutral, factual language. No productivity judgments of any kind -- do not say someone
   was "productive", "behind", "did well", "should have done more", or compare members against
   each other. Report what was recorded, nothing more:
     - Write: "Ali recorded 2h of focused work on Complete Module 2."
     - Never: "Ali only worked 2h." / "Abrar was the most productive this period."
8. No motivational filler, no exclamation marks, no emoji.
9. Every `user_id` you use in `members[]` MUST be exactly one of the member ids listed in the
   snapshot's `members[]`. Do not invent member ids or add an entry for someone not present.
10. Every number, time, or percentage you state must be exactly derivable from the snapshot
    (already-provided totals, counts, or progress_start/progress_end values) -- never calculate,
    round differently, or estimate a new figure.
"""

_NUMBER_RE = re.compile(r"\d[\d,.]*%?")


def _to_status_schemas(events: List[ProviderStatusEvent]) -> List[ProviderStatusEventSchema]:
    return [
        ProviderStatusEventSchema(type=ev.type, status=ev.status, message=ev.message, provider=ev.provider)
        for ev in events
    ]


def _format_hm(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m"


def _fallback_workspace_changes_summary(snapshot: StructuredSnapshot) -> str:
    changes = snapshot.workspace_changes
    if not changes.has_any:
        return ""
    bits: List[str] = []
    if changes.invitations:
        bits.append(f"{len(changes.invitations)} invitation(s) sent or updated")
    if changes.members_joined:
        bits.append(f"{len(changes.members_joined)} member(s) joined")
    if changes.members_removed:
        bits.append(f"{len(changes.members_removed)} member(s) left")
    if changes.tasks_created:
        bits.append(f"{changes.tasks_created} task(s) created")
    if changes.tasks_completed:
        bits.append(f"{changes.tasks_completed} task(s) completed")
    if changes.tasks_skipped:
        bits.append(f"{changes.tasks_skipped} task(s) skipped")
    if changes.tasks_deleted:
        bits.append(f"{changes.tasks_deleted} task(s) deleted")
    return (", ".join(bits) + ".") if bits else ""


def _fallback_narrative(snapshot: StructuredSnapshot) -> SummaryNarrative:
    """
    Deterministic, non-AI narrative built directly from the snapshot. Used both for the
    "no activity" short-circuit and as a last-resort fallback when the AI keeps
    failing validation -- the feature must never surface a fabricated or broken summary.
    """
    if not snapshot.has_activity:
        return SummaryNarrative(
            overall_summary="No significant workspace activity was recorded during the previous 24 hours.",
            members=[],
            workspace_changes_summary="",
            highlights=[],
        )

    active_members: List[MemberEntry] = [
        m for m in snapshot.members if m.focused_seconds > 0 or m.events or m.task_activity
    ]
    if snapshot.total_focused_seconds > 0:
        overall = (
            f"{len(active_members)} member{'s' if len(active_members) != 1 else ''} were active, "
            f"logging {_format_hm(snapshot.total_focused_seconds)} of focused work during the "
            f"previous 24 hours."
        )
    else:
        overall = (
            f"{len(active_members)} member{'s' if len(active_members) != 1 else ''} had recorded "
            f"activity during the previous 24 hours, with no focused time logged."
        )

    notes: List[MemberNarrative] = []
    for member in active_members:
        if member.task_activity:
            completed = sum(1 for t in member.task_activity if t.status_end == TaskStatus.completed)
            note = (
                f"{member.display_name} focused {_format_hm(member.focused_seconds)} across "
                f"{len(member.task_activity)} task{'s' if len(member.task_activity) != 1 else ''}, "
                f"{completed} completed."
            )
        elif member.events:
            note = (
                f"{member.display_name} recorded {len(member.events)} "
                f"activity item{'s' if len(member.events) != 1 else ''}, no focused time logged."
            )
        else:
            note = f"{member.display_name} had no recorded task activity."
        notes.append(MemberNarrative(user_id=member.user_id, note=note))

    return SummaryNarrative(
        overall_summary=overall,
        members=notes,
        workspace_changes_summary=_fallback_workspace_changes_summary(snapshot),
        highlights=[],
    )


def _expected_numeric_tokens(snapshot: StructuredSnapshot) -> Set[str]:
    """Numbers that are legitimately derivable from the snapshot -- used to sanity-check
    that the AI narrative isn't inventing new figures."""
    tokens: Set[str] = {str(snapshot.total_focused_seconds), str(len(snapshot.members))}

    total_hours, total_rem = divmod(snapshot.total_focused_seconds, 3600)
    tokens.update({str(total_hours), str(total_rem // 60)})

    # The window's boundary dates are legitimately restated in prose (e.g. "on the 17th" or
    # "into the 18th"), not a fabricated figure -- allow both report_start's and report_end's
    # day/month/year components (a rolling 24h window can span two calendar dates).
    for boundary in (snapshot.report_start, snapshot.report_end):
        tokens.update({str(boundary.day), str(boundary.month), str(boundary.year)})

    for member in snapshot.members:
        tokens.update({str(member.focused_seconds), str(len(member.task_activity)), str(len(member.events))})
        hours, rem = divmod(member.focused_seconds, 3600)
        tokens.update({str(hours), str(rem // 60)})
        completed = sum(1 for t in member.task_activity if t.status_end == TaskStatus.completed)
        in_progress = sum(1 for t in member.task_activity if t.status_end == TaskStatus.in_progress)
        skipped = sum(1 for t in member.task_activity if t.status_end == TaskStatus.skipped)
        tokens.update({str(completed), str(in_progress), str(skipped)})
        for t in member.task_activity:
            t_hours, t_rem = divmod(t.focused_seconds, 3600)
            tokens.update({str(t.focused_seconds), str(t_hours), str(t_rem // 60)})
            if t.progress_start is not None:
                tokens.add(str(t.progress_start))
            if t.progress_end is not None:
                tokens.add(str(t.progress_end))

    changes = snapshot.workspace_changes
    tokens.update(
        {
            str(len(changes.invitations)),
            str(len(changes.members_joined)),
            str(len(changes.members_removed)),
            str(changes.tasks_created),
            str(changes.tasks_completed),
            str(changes.tasks_skipped),
            str(changes.tasks_deleted),
        }
    )

    return tokens


def _validate_narrative(narrative: SummaryNarrative, snapshot: StructuredSnapshot) -> List[str]:
    """Returns a list of human-readable validation warnings; empty means the narrative is clean."""
    warnings: List[str] = []

    known_ids = snapshot.known_member_ids
    for note in narrative.members:
        if note.user_id not in known_ids:
            warnings.append(f"members references unknown member_id '{note.user_id}'")

    expected_numbers = _expected_numeric_tokens(snapshot)
    text_blob = " ".join(
        [
            narrative.overall_summary,
            narrative.workspace_changes_summary,
            *(n.note for n in narrative.members),
            *narrative.highlights,
        ]
    )
    for raw in _NUMBER_RE.findall(text_blob):
        cleaned = raw.rstrip("%").replace(",", "")
        if not cleaned or len(cleaned) <= 1:
            # Single digits ("a 1-on-1", list markers, etc.) aren't worth flagging.
            continue
        if cleaned in expected_numbers:
            continue
        warnings.append(f"figure '{raw}' in narrative is not traceable to the snapshot")

    return warnings


class SummaryService:
    """Orchestrates structured-snapshot -> validated AI narrative for the Daily Report."""

    def __init__(self, gateway_instance: LLMGateway = gateway):
        self.gateway = gateway_instance

    async def generate_summary(self, snapshot: StructuredSnapshot) -> SummaryGenerateResponse:
        if not snapshot.has_activity:
            logger.info(
                f"📭 No activity for workspace={snapshot.workspace_id} "
                f"window={snapshot.report_start}..{snapshot.report_end}; skipping AI call."
            )
            return SummaryGenerateResponse(
                snapshot=snapshot,
                narrative=_fallback_narrative(snapshot),
                meta=GenerationMetadata(
                    provider="none",
                    model="rule_based",
                    generated_at=datetime.now(dt_timezone.utc),
                    usage=UsageInfo(),
                    status_events=[],
                    used_fallback_template=True,
                    validation_warnings=[],
                ),
            )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "STRUCTURED SNAPSHOT (the only source of truth):\n"
                    f"{snapshot.model_dump_json(indent=2)}\n\nNarrate this."
                )
            ),
        ]

        last_warnings: List[str] = []
        max_attempts = 1 + max(settings.SUMMARY_MAX_VALIDATION_RETRIES, 0)
        attempts_made = 0

        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            try:
                narrative, provider, model, usage, status_events = await self.gateway.generate_structured(
                    messages=messages,
                    schema=SummaryNarrative,
                    temperature=settings.SUMMARY_TEMPERATURE,
                )
            except Exception as exc:
                logger.error(f"❌ Structured summary generation failed after gateway exhaustion: {exc}")
                break

            warnings = _validate_narrative(narrative, snapshot)
            if not warnings:
                return SummaryGenerateResponse(
                    snapshot=snapshot,
                    narrative=narrative,
                    meta=GenerationMetadata(
                        provider=provider,
                        model=model,
                        generated_at=datetime.now(dt_timezone.utc),
                        usage=UsageInfo(**usage),
                        status_events=_to_status_schemas(status_events),
                        used_fallback_template=False,
                        validation_warnings=[],
                    ),
                )

            last_warnings = warnings
            logger.warning(f"⚠️ Narrative validation failed (attempt {attempt}/{max_attempts}): {warnings}")
            if attempt < max_attempts:
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous answer referenced data not present in the snapshot: "
                            + "; ".join(warnings)
                            + ". Try again, using ONLY facts from the snapshot above."
                        )
                    )
                )

        logger.error(
            f"❌ Falling back to the deterministic template for workspace={snapshot.workspace_id} "
            f"window={snapshot.report_start}..{snapshot.report_end} after {attempts_made} attempt(s)."
        )
        return SummaryGenerateResponse(
            snapshot=snapshot,
            narrative=_fallback_narrative(snapshot),
            meta=GenerationMetadata(
                provider="none",
                model="rule_based",
                generated_at=datetime.now(dt_timezone.utc),
                usage=UsageInfo(),
                status_events=[],
                used_fallback_template=True,
                validation_warnings=last_warnings,
            ),
        )


# Singleton instance
summary_service = SummaryService()
