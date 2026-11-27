
import json

from parser import parse_chat_file
from event_extractor import extract_events
from state_engine import build_project_state
from qa_engine import answer_question, print_answer


# Read the WhatsApp conversation.
messages = parse_chat_file("data/team_chat.txt")

print("Messages parsed:", len(messages))


import os

if os.path.exists("data/extracted_events.json"):
    with open(
        "data/extracted_events.json",
        "r",
        encoding="utf-8"
    ) as file:
        events = json.load(file)

    print("Loaded previously extracted events.")

else:
    events = extract_events(
        messages,
        batch_size=4,
        context_size=3
    )

    with open(
        "data/extracted_events.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            events,
            file,
            indent=4,
            ensure_ascii=False
        )

print("Events extracted:", len(events))

project_state = build_project_state(events)


# Build the current project state.
project_state = build_project_state(events)


# Save the project state for inspection.
with open(
    "data/project_state.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        project_state,
        file,
        indent=4,
        ensure_ascii=False,
        default=str
    )


# Ask a question.
question = "Who is responsible for authentication?"

result = answer_question(
    question=question,
    messages=messages,
    project_state=project_state
)

print_answer(result)
