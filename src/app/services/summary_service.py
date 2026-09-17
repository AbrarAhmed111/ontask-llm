"""
AI Daily Summary Service (Phase 10 -- Shared "Yesterday's Work" Summary).

Structured-first, AI-narrates-only (see 10.2 and 10.4 of
project_document/ontask-evolution-plan.md):
- Every hard fact (times, names, task statuses, counts) is already computed by the
  caller and lives in `StructuredSnapshot`. This service never touches a database
  and never invents a number -- it only asks the model to narrate what's already
  there, then validates that the narration didn't smuggle in anything new before
  handing it back.
- No recorded work -> short-circuit to a deterministic sentence without calling
  the AI at all (10.3).
- If the AI's narrative keeps referencing unknown members or untraceable numbers
  after one corrective retry, fall back to a deterministic, template-built
  narrative rather than surfacing a broken or fabricated summary.
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
    MemberNote,
    StructuredSnapshot,
    SummaryGenerateResponse,
    SummaryNarrative,
)

logger = logging.getLogger("SummaryService")
settings = get_settings()

# Initialize the shared LLM Gateway instance (multi-provider, automatic failover).
gateway = LLMGateway(
    max_attempts=settings.GATEWAY_MAX_ATTEMPTS,
    cooldown_seconds=settings.GATEWAY_COOLDOWN_SECONDS,
)

SYSTEM_PROMPT = """You are a neutral reporting assistant for OnTask, a team focus-time tracker.

You will be given a STRUCTURED JSON snapshot of everything a workspace's members focused \
on during one day. Your ONLY job is to narrate that data in plain, factual prose.

Hard rules:
- Summarize ONLY the data given. Never invent a task, member, time value, or status.
- Never state a number (time, percentage, count) that is not derivable from the JSON.
- Preserve parent/subtask relationships when describing task groups.
- Clearly separate completed, in-progress, and skipped work.
- Use a neutral, factual tone. No productivity judgments ("great job", "behind schedule",
  "productive", "slacking") unless that exact word appears in the input.
- No motivational filler, no exclamation marks, no emoji.
- If activity was minimal or absent for a member, say so plainly instead of padding.
- Every `user_id` you reference in member_notes MUST be exactly one of the member ids
  listed in the snapshot. Do not invent member ids.
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


def _fallback_narrative(snapshot: StructuredSnapshot) -> SummaryNarrative:
    """
    Deterministic, non-AI narrative built directly from the snapshot. Used both for the
    "no activity" short-circuit (10.3) and as a last-resort fallback when the AI keeps
    failing validation -- the feature must never surface a fabricated or broken summary.
    """
    if not snapshot.has_activity:
        return SummaryNarrative(
            overall_summary=f"No focused work was recorded on {snapshot.summary_date.isoformat()}.",
            member_notes=[],
            highlights=[],
        )

    active_members: List[MemberEntry] = [m for m in snapshot.members if m.focused_seconds > 0]
    overall = (
        f"{len(active_members)} member{'s' if len(active_members) != 1 else ''} logged "
        f"{_format_hm(snapshot.total_focused_seconds)} of focused work on "
        f"{snapshot.summary_date.isoformat()}."
    )

    notes: List[MemberNote] = []
    for member in active_members:
        completed = sum(1 for t in member.tasks if t.status == "completed")
        notes.append(
            MemberNote(
                user_id=member.user_id,
                note=(
                    f"{member.display_name} focused {_format_hm(member.focused_seconds)} "
                    f"across {len(member.tasks)} task{'s' if len(member.tasks) != 1 else ''}, "
                    f"{completed} completed."
                ),
            )
        )

    return SummaryNarrative(overall_summary=overall, member_notes=notes, highlights=[])


def _expected_numeric_tokens(snapshot: StructuredSnapshot) -> Set[str]:
    """Numbers that are legitimately derivable from the snapshot -- used to sanity-check
    that the AI narrative isn't inventing new figures."""
    tokens: Set[str] = {str(snapshot.total_focused_seconds), str(len(snapshot.members))}

    total_hours, total_rem = divmod(snapshot.total_focused_seconds, 3600)
    tokens.update({str(total_hours), str(total_rem // 60)})

    # The date itself is legitimately restated in prose (e.g. "on the 16th"),
    # not a fabricated figure -- allow its day/month/year components too.
    tokens.update(
        {str(snapshot.summary_date.day), str(snapshot.summary_date.month), str(snapshot.summary_date.year)}
    )

    for member in snapshot.members:
        tokens.update({str(member.focused_seconds), str(len(member.tasks))})
        hours, rem = divmod(member.focused_seconds, 3600)
        tokens.update({str(hours), str(rem // 60)})
        completed = sum(1 for t in member.tasks if t.status == "completed")
        in_progress = sum(1 for t in member.tasks if t.status == "in_progress")
        skipped = sum(1 for t in member.tasks if t.status == "skipped")
        tokens.update({str(completed), str(in_progress), str(skipped)})

    return tokens


def _validate_narrative(narrative: SummaryNarrative, snapshot: StructuredSnapshot) -> List[str]:
    """Returns a list of human-readable validation warnings; empty means the narrative is clean."""
    warnings: List[str] = []

    known_ids = snapshot.known_member_ids
    for note in narrative.member_notes:
        if note.user_id not in known_ids:
            warnings.append(f"member_notes references unknown member_id '{note.user_id}'")

    expected_numbers = _expected_numeric_tokens(snapshot)
    text_blob = " ".join(
        [narrative.overall_summary, *(n.note for n in narrative.member_notes), *narrative.highlights]
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
    """Orchestrates structured-snapshot -> validated AI narrative for Phase 10."""

    def __init__(self, gateway_instance: LLMGateway = gateway):
        self.gateway = gateway_instance

    async def generate_summary(self, snapshot: StructuredSnapshot) -> SummaryGenerateResponse:
        if not snapshot.has_activity:
            logger.info(
                f"📭 No activity for workspace={snapshot.workspace_id} date={snapshot.summary_date}; "
                f"skipping AI call."
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
            f"date={snapshot.summary_date} after {attempts_made} attempt(s)."
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
