"""
Extract structured project events from WhatsApp messages using NVIDIA NIM.

The public extract_events() interface and returned event schema are kept
compatible with state_engine.py.
"""

import os
import json
import re
import time

from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel,ConfigDict

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)

BASE_URL = "https://integrate.api.nvidia.com/v1"
# NVIDIA NIM extraction settings
MAX_OUTPUT_TOKENS = 8192
MAX_API_ATTEMPTS = 3

# Conservative batching for reliable event extraction
DEFAULT_BATCH_SIZE = 1
DEFAULT_CONTEXT_SIZE = 2

EventType = Literal[
    "decision_proposed",
    "decision_confirmed",
    "decision_changed",
    "task_assigned",
    "task_reassigned",
    "task_progress",
    "task_completed",
    "task_blocked",
    "deadline_set",
    "deadline_changed",
    "requirement_added",
    "meeting_scheduled",
    "other",
]

ConfirmationStatus = Literal[
    "proposed",
    "confirmed",
    "uncertain",
    "rejected",
    "not_applicable",
]

TASK_EVENT_TYPES = {
    "task_assigned",
    "task_reassigned",
    "task_progress",
    "task_completed",
    "task_blocked",
}
ASSIGNMENT_EVENT_TYPES = {"task_assigned", "task_reassigned"}
VALID_CONFIRMATION_STATUSES = {
    "proposed",
    "confirmed",
    "uncertain",
    "rejected",
    "not_applicable",
}


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

class ProjectEvent(BaseModel):
    """One project event grounded in one or more WhatsApp messages."""

    model_config = ConfigDict(extra="ignore")

    event_type: EventType
    subject: str
    details: str
    actor: str | None = None
    owner: str | None = None
    confirmation_status: ConfirmationStatus = "not_applicable"
    evidence_message_ids: list[int]
    event_message_id: int
    timestamp: str


# ---------------------------------------------------------------------------
# LLM instructions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = r"""
You are a careful information-extraction system for WhatsApp project chats.

Extract project events ONLY from new_messages. context_messages are provided
only to understand references and earlier discussion. Do not create an event
whose event_message_id belongs only to context_messages.

The chat may contain English, Hinglish, abbreviations, typos, and informal
language. Do not invent facts. If the messages do not establish something,
leave it unknown.

Return valid JSON only, in this exact outer structure:
{
  "events": [
    {
      "event_type": "task_assigned",
      "subject": "Authentication",
      "details": "Rohan was assigned authentication.",
      "actor": "Aarav",
      "owner": "Rohan",
      "confirmation_status": "confirmed",
      "evidence_message_ids": [17],
      "event_message_id": 17,
      "timestamp": "2026-10-05T20:02:00"
    }
  ]
}

If there are no project events, return {"events": []}.

OUTPUT SIZE RULES:
- Return only the JSON object. Do not explain your reasoning.
- Keep each event's details to 20 words or fewer.
- Extract only meaningful project events; skip casual conversation.
- Do not repeat the same event in one response.
- Do not add optional commentary or Markdown fences.

ALLOWED EVENT TYPES:
decision_proposed, decision_confirmed, decision_changed,
task_assigned, task_reassigned, task_progress, task_completed,
task_blocked, deadline_set, deadline_changed, requirement_added,
meeting_scheduled, other.

ACTOR AND OWNER ARE DIFFERENT:
1. actor is always the sender of event_message_id.
2. owner is the person actually responsible for the task.
3. Never assume the sender owns a task just because they sent the message.
4. If ownership is not explicit, owner must be null.
5. For first-person statements such as "I'll handle payments",
   "main deployment kar lunga", or "chat main kar lunga", the sender is owner.
6. If a message explicitly allocates tasks to named people, use the person
   named next to that specific task.
7. A single task-allocation message can produce multiple separate events.
8. Example:
   Aarav: Task split:
   Rohan - authentication
   Priya - listings UI
   Meera - payments
   Kabir - search and chat
   Main - deployment
   Correct owners are Rohan, Priya, Meera, Kabir, and Aarav respectively.
   Aarav is the actor for all events, but not the owner of all tasks.
9. Do not infer owner from a previous unrelated task or from who usually
   performs that type of work.
10. If someone asks another person to take a task, that is not confirmed
    until the person accepts.
11. A question such as "search kaun karega?" is not an assignment.
12. "Search Saturday ko discuss karte hain" is not an assignment.
13. A meeting agenda asking the team to decide an owner is not a reassignment.
14. A proposed assignment must have confirmation_status "proposed" or
    "uncertain", not "confirmed".
15. A task_reassigned event requires evidence of an actual ownership change.
16. Do not mark a task blocked merely because it is pending. Use task_blocked
    only when an actual blocker/prevention is stated.
17. Do not mark a task completed merely because someone promises to finish it.
18. Preserve stated progress percentages and important qualifiers.
19. Do not merge different tasks just because they have the same owner.
20. If an owner is unclear, use null rather than guessing.

TASK EVENT TYPE:
- task_assigned: a new assignment is explicitly made and accepted/confirmed.
- task_reassigned: an existing task is explicitly transferred and accepted.
- task_progress: work has started or progress is reported.
- task_completed: work is explicitly finished, merged, or delivered.
- task_blocked: work cannot proceed because of a stated blocker.

DECISIONS:
1. A suggestion/recommendation is decision_proposed.
2. An explicit agreement is decision_confirmed.
3. A replacement of an earlier confirmed choice is decision_changed.
4. "Agreed", "+1", "Django fine", and similar statements count as agreement
   only when the referenced choice is clear from the supplied context.
5. Do not treat a question as a decision.
6. Track separate decisions independently: backend, frontend, database,
   payment gateway, deployment platform, and other distinct topics.
7. Do not add technologies or details that are not supported by evidence.
8. Use confirmation_status consistently:
   decision_proposed -> proposed
   decision_confirmed -> confirmed
   decision_changed -> confirmed

DEADLINES AND MEETINGS:
1. Extract a deadline only when a date/time is stated or unambiguously
   resolved from the supplied conversation.
2. If a deadline is changed, use deadline_changed.
3. Distinguish a proposed date from a confirmed date.
4. Extract meeting date, time, and location only when present.
5. Do not assume a meeting happened just because it was scheduled.
6. Preserve ambiguity instead of inventing a date.

EVIDENCE AND SOURCE FIELDS:
1. Every event must include evidence_message_ids.
2. IDs must come from messages in the supplied input.
3. event_message_id must be the new message that actually communicates
   the event, not merely a related earlier context message.
4. Include relevant context IDs as additional evidence when necessary.
5. Do not use unrelated messages as evidence.
6. actor must equal the sender of event_message_id.
7. Do not invent IDs, sender names, or timestamps.
8. The program will replace actor and timestamp with source values, but you
   must still identify the correct event_message_id.
"""


# ---------------------------------------------------------------------------
# API client and batching
# ---------------------------------------------------------------------------

def create_open_client():
    if not API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY was not found. Add it to your .env file."
        )

    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=120,
        max_retries=0
    )


def build_batch_windows(messages, batch_size, context_size):
    """Return (context, new_messages) windows without losing message order."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if context_size < 0:
        raise ValueError("context_size cannot be negative.")

    windows = []
    for start in range(0, len(messages), batch_size):
        end = start + batch_size
        context_start = max(0, start - context_size)
        windows.append((messages[context_start:start], messages[start:end]))
    return windows


def prepare_messages(messages):
    """Keep only fields needed by the model; preserve full message text."""

    return [
        {
            "id": message["id"],
            "timestamp": str(message["timestamp"]),
            "sender": str(message["sender"]),
            "message": str(message["message"]),
        }
        for message in messages
    ]


# ---------------------------------------------------------------------------
# JSON response handling
# ---------------------------------------------------------------------------

def extract_json_response(content):
    """Parse JSON, tolerating surrounding Markdown or explanatory text."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty response.")

    cleaned = content.strip()
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("The model response contains no JSON object.")
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
            return parsed
        except json.JSONDecodeError as error:
            raise ValueError(
                f"The model returned malformed JSON: {error}"
            ) from error


def _response_content(response):
    """Read standard chat-completion content and detect token truncation."""

    if not response.choices:
        raise ValueError("The API response contained no choices.")

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError(
            "OpenRouter/model truncated the response (finish_reason='length'). "
            "Reduce batch_size to 1 or use a model with a larger output limit."
        )

    content = choice.message.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "\n".join(parts)

    if not content:
        raise ValueError("The model returned an empty response.")
    return content


def extract_events_from_batch(client, context_messages, new_messages):
    """Call OpenRouter for one batch and return raw event dictionaries."""

    payload = {
        "context_messages": prepare_messages(context_messages),
        "new_messages": prepare_messages(new_messages),
    }

    last_error = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
            )

            result = extract_json_response(_response_content(response))
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object from the model.")

            events = result.get("events")
            if not isinstance(events, list):
                raise ValueError(
                    'The model response must contain an "events" list.'
                )
            return events

        except Exception as error:
            last_error = error
            if attempt < MAX_API_ATTEMPTS:
                wait_seconds = min(2 ** (attempt - 1), 4)
                print(
                    f"Batch request attempt {attempt} failed: "
                    f"{type(error).__name__}: {error}"
                )
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"NVIDIA extraction failed after {MAX_API_ATTEMPTS} attempts: "
        f"{last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Conservative ownership validation
# ---------------------------------------------------------------------------

def _name_appears_in_text(name, text):
    if not name or not text:
        return False
    escaped = re.escape(name.strip())
    return re.search(
        rf"(?<!\w){escaped}(?!\w)", text, flags=re.IGNORECASE
    ) is not None


def _is_explicit_first_person_ownership(message_text):
    """Detect a clear first-person commitment, not a vague intention."""

    text = message_text.lower()
    patterns = [
        r"\bi(?:'ll| will)\s+(?:take|handle|do|own|implement|finish|complete)\b",
        r"\bi(?:'m| am)\s+(?:taking|handling|doing|implementing)\b",
        r"\bi(?:'ll| will)\s+be\s+(?:handling|doing|working on)\b",
        r"\bmain\s+(?:yeh\s+)?(?:kaam\s+)?(?:kar lunga|kar lungi|sambhal lunga|sambhal lungi)\b",
        r"\bmain\s+(?:deployment|auth|authentication|payments|payment|search|chat)\s+(?:kar lunga|kar lungi|sambhal lunga|sambhal lungi)\b",
        r"\b(?:chat|auth|authentication|payments|payment|search)\s+main\s+kar\s+(?:lunga|lungi)\b",
        r"\b(?:theek hai|okay|ok),?\s+main\s+(?:le leta hoon|le leti hoon|kar lunga|kar lungi)\b",
        r"\b(?:i accept|i can take|i'll take it|i will take it)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _is_self_allocation_line(message_text, subject):
    """
    Match roster syntax such as 'Main - deployment' for the message sender.
    This is intentionally narrow to avoid treating ordinary mentions of
    'main' as an assignment.
    """

    subject_words = [
        word.lower()
        for word in re.findall(r"[a-zA-Z]+", subject or "")
        if len(word) > 2
    ]
    if not subject_words:
        return False

    for line in message_text.splitlines():
        normalized = line.strip().lower()
        if not re.search(r"^\s*main\s*[-:–]\s*", normalized):
            continue
        if any(word in normalized for word in subject_words):
            return True
    return False


def resolve_task_owner(raw_owner, event_message, known_senders, subject=""):
    """
    Accept an owner only when supported by the event's source message.

    A named third-party owner must appear in the source message. The sender
    can own a task through an explicit first-person commitment or a roster
    line such as 'Main - deployment'.
    """

    message_text = str(event_message.get("message", ""))
    sender = event_message.get("sender")
    owner_candidate = str(raw_owner or "").strip()

    if not owner_candidate or owner_candidate.lower() in {
        "none", "null", "unknown", "unassigned", "not specified",
        "not identified", "n/a",
    }:
        if sender and (
            _is_explicit_first_person_ownership(message_text)
            or _is_self_allocation_line(message_text, subject)
        ):
            return sender
        return None

    matched_sender = next(
        (
            known_sender
            for known_sender in known_senders
            if known_sender.casefold() == owner_candidate.casefold()
        ),
        None,
    )
    if matched_sender is None:
        return None

    if _name_appears_in_text(matched_sender, message_text):
        return matched_sender

    if (
        sender
        and matched_sender.casefold() == str(sender).casefold()
        and (
            _is_explicit_first_person_ownership(message_text)
            or _is_self_allocation_line(message_text, subject)
        )
    ):
        return sender

    return None


# ---------------------------------------------------------------------------
# Event validation and source grounding
# ---------------------------------------------------------------------------

def _normalise_confirmation(value):
    if not isinstance(value, str):
        return "not_applicable"
    value = value.strip().lower()
    return value if value in VALID_CONFIRMATION_STATUSES else "uncertain"


def _normalise_event_text(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event is missing a valid {field_name}.")
    return value.strip()


def validate_and_build_event(
    raw_event,
    message_lookup,
    core_message_ids,
    known_senders,
):
    """Validate one model event and replace source fields with trusted data."""

    if not isinstance(raw_event, dict):
        return None

    event_message_id = raw_event.get("event_message_id")
    if (
        not isinstance(event_message_id, int)
        or isinstance(event_message_id, bool)
        or event_message_id not in core_message_ids
        or event_message_id not in message_lookup
    ):
        return None

    event_message = message_lookup[event_message_id]

    try:
        event_type = raw_event.get("event_type")
        subject = _normalise_event_text(raw_event.get("subject"), "subject")
        details = _normalise_event_text(raw_event.get("details"), "details")

        raw_evidence = raw_event.get("evidence_message_ids", [])
        if not isinstance(raw_evidence, list):
            raw_evidence = []

        valid_evidence_ids = []
        for message_id in raw_evidence:
            if (
                isinstance(message_id, int)
                and not isinstance(message_id, bool)
                and message_id in message_lookup
                and message_id not in valid_evidence_ids
            ):
                valid_evidence_ids.append(message_id)
        if event_message_id not in valid_evidence_ids:
            valid_evidence_ids.append(event_message_id)

        confirmation_status = _normalise_confirmation(
            raw_event.get("confirmation_status")
        )

        owner = None
        if event_type in TASK_EVENT_TYPES:
            owner = resolve_task_owner(
                raw_event.get("owner"),
                event_message,
                known_senders,
                subject=subject,
            )

        if event_type in ASSIGNMENT_EVENT_TYPES and owner is None:
            if confirmation_status != "rejected":
                confirmation_status = "uncertain"

        source_text = str(event_message.get("message", "")).lower()
        agenda_patterns = [
            r"\bkaun\s+(?:karega|karegi)\b",
            r"\bwho\s+(?:will|can)\s+(?:do|handle|take)\b",
            r"\bwho's\s+(?:doing|handling)\b",
            r"\bdiscuss\s+(?:the\s+)?(?:owner|ownership|who)",
            r"\bdecide\s+(?:the\s+)?(?:owner|ownership)",
            r"\bagenda\b",
        ]
        looks_like_agenda = any(
            re.search(pattern, source_text) for pattern in agenda_patterns
        )
        if (
            event_type in ASSIGNMENT_EVENT_TYPES
            and looks_like_agenda
            and owner is None
            and confirmation_status != "rejected"
        ):
            confirmation_status = "uncertain"

        event_data = {
            "event_type": event_type,
            "subject": subject,
            "details": details,
            "actor": event_message.get("sender"),
            "owner": owner,
            "confirmation_status": confirmation_status,
            "evidence_message_ids": valid_evidence_ids,
            "event_message_id": event_message_id,
            "timestamp": str(event_message["timestamp"]),
        }
        return ProjectEvent(**event_data)

    except (ValidationError, ValueError, TypeError) as error:
        print(
            f"Skipping invalid event from message {event_message_id}: {error}"
        )
        return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def remove_duplicate_events(events):
    """Remove duplicate outputs while retaining distinct later updates."""

    unique_events = []
    seen = set()
    for event in events:
        signature = (
            event.event_type,
            " ".join(event.subject.casefold().split()),
            event.event_message_id,
            event.owner.casefold() if event.owner else None,
        )
        if signature not in seen:
            seen.add(signature)
            unique_events.append(event)
    return unique_events


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------

def extract_events(
    messages,
    batch_size=DEFAULT_BATCH_SIZE,
    context_size=DEFAULT_CONTEXT_SIZE,
):
    """
    Extract events from parsed WhatsApp messages.

    Required fields per message: id, timestamp, sender, message.
    Optional field: message_type (defaults to 'user').
    Returns dictionaries compatible with state_engine.py.
    """

    if not isinstance(messages, list):
        raise TypeError("messages must be a list of parsed message dictionaries.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if context_size < 0:
        raise ValueError("context_size cannot be negative.")

    user_messages = [
        message for message in messages
        if message.get("message_type", "user") == "user"
    ]
    if not user_messages:
        print("No user messages found. Nothing to extract.")
        return []

    seen_ids = set()
    for message in user_messages:
        required = ("id", "timestamp", "sender", "message")
        missing = [key for key in required if key not in message]
        if missing:
            raise ValueError(
                f"Parsed message is missing required fields {missing}: {message}"
            )
        message_id = message["id"]
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise ValueError(f"Message ID must be an integer: {message_id!r}")
        if message_id in seen_ids:
            raise ValueError(f"Duplicate parsed message ID: {message_id}")
        seen_ids.add(message_id)

    known_senders = sorted(
        {
            str(message["sender"]).strip()
            for message in user_messages
            if message.get("sender")
        },
        key=str.casefold,
    )

    client = create_open_client()
    batch_windows = build_batch_windows(
        user_messages, batch_size, context_size
    )
    extracted_events = []

    print(f"Messages sent for event extraction: {len(user_messages)}")
    print(f"Extraction batches: {len(batch_windows)}")
    print(f"NVIDIA model: {MODEL_NAME}")

    for batch_number, (context_messages, core_messages) in enumerate(
        batch_windows, start=1
    ):
        print(
            f"\nExtracting events from batch "
            f"{batch_number}/{len(batch_windows)}..."
        )
        print(f"Context messages: {len(context_messages)}")
        print(f"New messages: {len(core_messages)}")

        try:
            raw_events = extract_events_from_batch(
                client, context_messages, core_messages
            )
        except Exception as error:
            # Do not silently skip a failed batch: partial extraction could
            # make the project state misleading.
            raise RuntimeError(
                f"Event extraction failed at batch {batch_number}/"
                f"{len(batch_windows)}. No complete result was produced. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error

        all_batch_messages = context_messages + core_messages
        message_lookup = {
            message["id"]: message for message in all_batch_messages
        }
        core_message_ids = {message["id"] for message in core_messages}

        accepted_count = 0
        for raw_event in raw_events:
            event = validate_and_build_event(
                raw_event,
                message_lookup,
                core_message_ids,
                known_senders,
            )
            if event is not None:
                extracted_events.append(event)
                accepted_count += 1

        print(
            f"Raw events: {len(raw_events)} | "
            f"Validated events: {accepted_count}"
        )

    extracted_events = remove_duplicate_events(extracted_events)
    extracted_events.sort(
        key=lambda event: (event.timestamp, event.event_message_id)
    )

    result = [event.model_dump() for event in extracted_events]
    print(f"\nTotal validated unique events: {len(result)}")
    return result


if __name__ == "__main__":
    print("Event extractor module loaded successfully.")
    print(f"Configured NVIDIA model: {MODEL_NAME}")
    print("Call extract_events(parsed_messages) from your pipeline.")
