
import os
import json
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


# We're loading the API key and other settings from our .env file.
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openrouter/free")


# Here we're defining the kinds of events our agent should understand.
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
    "requirement_added",
    "other"
]


class ProjectEvent(BaseModel):
    # These fields describe what happened in the project.
    event_type: EventType
    subject: str
    owner: str | None = None
    details: str | None = None

    # We're keeping the IDs of the original messages that support this event.
    evidence_message_ids: list[int]

    # We'll fill this using the original chat timestamp, not an LLM guess.
    timestamp: str


# We're giving the LLM instructions about how to understand the conversation.
SYSTEM_PROMPT = """
You are an assistant that extracts project-related events from WhatsApp chats.

Read the messages carefully and identify meaningful project events.

The possible event types are:
- decision_proposed: Someone suggests an option, but it is not final.
- decision_confirmed: The team clearly agrees on a decision.
- decision_changed: An earlier decision is replaced or reversed.
- task_assigned: A task is given to someone.
- task_reassigned: An existing task is transferred to someone else.
- task_progress: Someone shares an update about ongoing work.
- task_completed: Someone clearly says that a task is finished.
- task_blocked: Someone reports a problem preventing progress.
- deadline_set: A deadline or due date is established.
- requirement_added: A new project requirement is introduced.
- other: A meaningful project event that does not fit the categories above.

Important rules:
1. Do not treat suggestions as confirmed decisions.
2. Do not treat promises as completed work.
3. Do not assume that a task is finished unless the message clearly says so.
4. A single message can contain multiple events.
5. Ignore greetings, jokes and unrelated conversations.
6. Do not invent people, dates, tasks or decisions.
7. Preserve the original meaning, including Hinglish messages.
8. Every event must refer to one or more message IDs from the messages provided.
9. If a message does not contain a meaningful project event, ignore it.

Return only valid JSON in this exact structure:

{
    "events": [
        {
            "event_type": "task_assigned",
            "subject": "Authentication",
            "owner": "Rohan",
            "details": "Rohan was assigned authentication work.",
            "evidence_message_ids": [12]
        }
    ]
}

If there are no meaningful events, return:

{
    "events": []
}
"""


def create_openrouter_client():
    # Before connecting, let's make sure our API key is available.
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Please check your .env file."
        )

    # We're using the OpenAI library to communicate with OpenRouter.
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def split_into_batches(messages, batch_size):
    # Instead of sending the entire chat in one request,
    # we're dividing it into smaller groups of messages.
    for start in range(0, len(messages), batch_size):
        yield messages[start:start + batch_size]


def extract_events_from_batch(client, messages):
    # Here we're preparing the messages in a simple format for the LLM.
    chat_content = []

    for message in messages:
        chat_content.append({
            "id": message["id"],
            "timestamp": message["timestamp"],
            "sender": message["sender"],
            "message": message["message"]
        })

    # We're converting the messages into JSON so the LLM can read them clearly.
    user_prompt = json.dumps(chat_content, ensure_ascii=False, indent=2)

    # Now we're asking the LLM to identify the project events.
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
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

    # The LLM's answer should contain our events in JSON format.
    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("The LLM returned an empty response.")

    # We're converting the JSON response into a Python dictionary.
    result = json.loads(response_text)

    return result.get("events", [])


def extract_events(messages, batch_size=10):
    # System notifications are not useful for our first round of event extraction.
    # So we're passing only messages sent by group members.
    user_messages = [
        message for message in messages
        if message["message_type"] == "user"
    ]

    client = create_openrouter_client()

    all_events = []

    # We're processing the conversation in smaller batches.
    for batch_number, batch in enumerate(
        split_into_batches(user_messages, batch_size),
        start=1
    ):
        print(f"Extracting events from batch {batch_number}...")

        raw_events = extract_events_from_batch(client, batch)

        # We'll use the original messages to verify the evidence IDs
        # and find the actual timestamp of each extracted event.
        message_lookup = {
            message["id"]: message
            for message in batch
        }

        for raw_event in raw_events:

            evidence_ids = raw_event.get("evidence_message_ids", [])

            # We're ignoring events that refer to messages outside this batch.
            if not evidence_ids:
                continue

            if any(message_id not in message_lookup for message_id in evidence_ids):
                continue

            # The event gets its timestamp from the earliest message supporting it.
            event_timestamp = min(
                message_lookup[message_id]["timestamp"]
                for message_id in evidence_ids
            )

            # Pydantic checks whether the extracted event has the correct structure.
            event = ProjectEvent(
                event_type=raw_event["event_type"],
                subject=raw_event["subject"],
                owner=raw_event.get("owner"),
                details=raw_event.get("details"),
                evidence_message_ids=evidence_ids,
                timestamp=event_timestamp
            )

            all_events.append(event.model_dump())

    return all_events
