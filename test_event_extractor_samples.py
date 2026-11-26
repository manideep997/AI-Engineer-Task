
from parser import parse_chat_file
from event_extractor import extract_events
import json


# First, we're loading the complete WhatsApp conversation.
messages = parse_chat_file("data/team_chat.txt")

# These are the message ranges we want to examine.
# We're selecting different parts of the conversation
# to test different kinds of project events.
test_ranges = [
    ("Task assignments", 14, 35),
    ("Payment decisions", 45, 75),
    ("Task progress", 75, 102)
]

for test_name, start_id, end_id in test_ranges:

    print("\n" + "=" * 60)
    print("Testing:", test_name)
    print("=" * 60)

    # We're selecting only messages from the current range.
    selected_messages = [
        message for message in messages
        if start_id <= message["id"] <= end_id
        and message["message_type"] == "user"
    ]

    print("Messages selected:", len(selected_messages))

    # We're extracting events from this selected part.
    events = extract_events(
        selected_messages,
        batch_size=5
    )

    print("\nExtracted events:")
    print(json.dumps(events, indent=4, ensure_ascii=False))

    print("\nEvents extracted:", len(events))
