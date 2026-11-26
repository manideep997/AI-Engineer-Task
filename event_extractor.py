
import os
import json
import re
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# We're loading the NVIDIA API key and model settings from our .env file.
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
model_name = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)


# These are the different kinds of events our agent can identify.
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
    "other"
]

ConfirmationStatus = Literal[
    "proposed",
    "confirmed",
    "uncertain",
    "rejected",
    "not_applicable"
]


class ProjectEvent(BaseModel):
    # Here we're describing what happened and what it relates to.
    event_type: EventType
    subject: str
    details: str | None = None

    # The person who communicated the event and the actual task owner
    # are two different things, so we're keeping them separate.
    actor: str | None = None
    owner: str | None = None

    # This helps us distinguish a suggestion from an actual team decision.
    confirmation_status: ConfirmationStatus

    # All the original messages that support this event are stored here.
    evidence_message_ids: list[int] = Field(min_length=1)

    # This identifies the message where the event actually becomes clear.
    event_message_id: int

    # We'll take the timestamp directly from that original message.
    timestamp: str


# These instructions tell the LLM how to interpret the WhatsApp conversation.
SYSTEM_PROMPT = """
You are an assistant that analyses WhatsApp group conversations
and extracts accurate project-related events.

Read the messages carefully. Understand the meaning and context,
including English, Hinglish and informal conversations.

Identify meaningful project events.

EVENT TYPES:

- decision_proposed:
  Someone suggests an option. The team has not confirmed it yet.

- decision_confirmed:
  A person clearly confirms or accepts a decision.

- decision_changed:
  An earlier confirmed decision is explicitly replaced or reversed.

- task_assigned:
  A task is assigned to someone for the first time.

- task_reassigned:
  Responsibility for an existing task is transferred to someone else.

- task_progress:
  Someone gives an update about work that is currently happening.

- task_completed:
  Someone clearly confirms that work is finished.

- task_blocked:
  A specific issue is actually preventing the task from progressing.

- deadline_set: A deadline is established for the first time.
- deadline_changed: An existing deadline is explicitly changed or replaced.

- requirement_added:
  A new project requirement is introduced.

- other:
  A meaningful project event that does not fit the categories above.

IMPORTANT RULES:

1. A suggestion is not a confirmed decision.

2. Someone saying "maybe", "I think", "haan shayad" or something
   similarly uncertain does not establish a confirmed decision.

3. A decision is changed only when an earlier confirmed decision
   is explicitly replaced or reversed.

4. Do not confuse the person sending a message with the person
   responsible for a task.

5. The actor is the person who communicated the event.

6. The owner is the person explicitly responsible for doing the task.
   If the owner is unclear, return null.

7. Do not assign task ownership to someone just because they reported
   a problem or discussed the task.

8. A task being difficult does not automatically mean it is blocked.
   Use task_blocked only when an actual obstacle prevents progress.

9. Do not treat promises as completed work.

10. A task can be mentioned using different names. Use a short,
    consistent subject where the relationship is clear.
    For example, use "Database schema" rather than switching
    between "DB schema" and "Database schema".

11. Do not merge two genuinely different tasks.

12. A single message can contain multiple independent events.

13. If multiple messages are needed to understand an event,
    include all relevant message IDs as evidence.

14. event_message_id must identify the message where the event
    is established. For a confirmed decision, this should normally
    be the confirmation message, not the original suggestion.

15. Every evidence_message_id and event_message_id must come
    from the messages provided.

16. Do not invent people, dates, tasks, decisions or information.

17. Preserve the original meaning of Hinglish messages.

18. Ignore greetings, jokes and unrelated conversations.

19. If a message does not contain a meaningful project event,
    do not create an event for it.

CONFIRMATION STATUS:

- proposed: An option has been suggested but not confirmed.
- confirmed: The event or decision has been explicitly confirmed.
- uncertain: The conversation does not establish a clear outcome.
- rejected: An option has explicitly been rejected.
- not_applicable: Confirmation status does not apply to this event.

Return only valid JSON in this structure:

{
    "events": [
        {
            "event_type": "task_assigned",
            "subject": "Authentication",
            "details": "Authentication work was assigned to Rohan.",
            "actor": "Aarav",
            "owner": "Rohan",
            "confirmation_status": "confirmed",
            "evidence_message_ids": [17],
            "event_message_id": 17
        }
    ]
}

If no meaningful events are found, return:

{
    "events": []
}
"""


def create_nvidia_client():
    # Before connecting, let's check whether our NVIDIA API key is available.
    if not api_key:
        raise ValueError(
            "NVIDIA API key not found. Please check your .env file."
        )

    # We're using the OpenAI library to connect to NVIDIA's API.
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=90.0,
        max_retries=0
    )


def split_into_batches(messages, batch_size):
    # Here we're dividing the conversation into smaller groups.
    for start in range(0, len(messages), batch_size):
        yield messages[start:start + batch_size]


def extract_json_response(response_text):
    response_text = response_text.strip()

    # Sometimes the model wraps its JSON response in Markdown code blocks.
    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(r"\s*```$", "", response_text)

    try:
        return json.loads(response_text)

    except json.JSONDecodeError as error:
        print("\nInvalid JSON received from the LLM.")
        print("Error:", error)
        print("\nActual LLM response:")
        print(response_text)

        raise


def extract_events_from_batch(client, messages):
    # We're preparing the messages in a simple format for the LLM.
    chat_content = []

    for message in messages:
        chat_content.append({
            "id": message["id"],
            "timestamp": message["timestamp"],
            "sender": message["sender"],
            "message": message["message"]
        })

    # Now we're converting the messages into JSON.
    user_prompt = json.dumps(
        chat_content,
        ensure_ascii=False,
        indent=2
    )

    # We're asking Nemotron to extract structured project events.
    response = client.chat.completions.create(
    model=model_name,
    temperature=0,
    max_tokens=8192,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)

    response_text = response.choices[0].message.content

    print("\nLLM response details:")
    print("Finish reason:", response.choices[0].finish_reason)

    if not response_text:
        print("The model returned an empty response.")
        raise ValueError("The NVIDIA model returned an empty response.")

    print("Received response successfully.")

    result = extract_json_response(response_text)

    if not isinstance(result, dict) or not isinstance(
        result.get("events"), list
    ):
        raise ValueError("The LLM returned an invalid events structure.")

    return result["events"]


def extract_events(messages, batch_size=10):
    # System notifications are not needed for this extraction stage.
    user_messages = [
        message for message in messages
        if message["message_type"] == "user"
    ]

    client = create_nvidia_client()

    all_events = []

    # We're processing the messages batch by batch.
    for batch_number, batch in enumerate(
        split_into_batches(user_messages, batch_size),
        start=1
    ):
        print(f"Extracting events from batch {batch_number}...")

        raw_events = extract_events_from_batch(client, batch)

        # This lets us look up the original message using its ID.
        message_lookup = {
            message["id"]: message
            for message in batch
        }

        for raw_event in raw_events:

            evidence_ids = raw_event.get("evidence_message_ids", [])

            if not isinstance(evidence_ids, list) or not evidence_ids:
                continue

            # Every evidence message must belong to this batch.
            if any(
                message_id not in message_lookup
                for message_id in evidence_ids
            ):
                continue

            # If the model hasn't supplied a valid event message ID,
            # we'll use the latest supporting message as a fallback.
            event_message_id = raw_event.get("event_message_id")

            if event_message_id not in evidence_ids:
                event_message_id = max(
                    evidence_ids,
                    key=lambda message_id: message_lookup[message_id]["timestamp"]
                )

            # We're taking the actual sender and timestamp from the chat.
            event_message = message_lookup[event_message_id]

            event = ProjectEvent(
                event_type=raw_event["event_type"],
                subject=raw_event["subject"],
                details=raw_event.get("details"),
                actor=event_message.get("sender"),
                owner=raw_event.get("owner"),
                confirmation_status=raw_event["confirmation_status"],
                evidence_message_ids=evidence_ids,
                event_message_id=event_message_id,
                timestamp=event_message["timestamp"]
            )

            all_events.append(event.model_dump())

    # Events from different batches are arranged in chronological order.
    all_events.sort(key=lambda event: event["timestamp"])

    return all_events
