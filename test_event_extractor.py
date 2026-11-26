
import json

from parser import parse_chat_file
from event_extractor import extract_events


# First, we're reading the WhatsApp conversation using our existing parser.
messages = parse_chat_file("data/team_chat.txt")


# For the first test, we're taking only the first 10 messages sent by group members.
# This lets us check the extractor without processing the entire conversation.
user_messages = [
    message for message in messages
    if message["message_type"] == "user"
][:10]


print("Messages sent for event extraction:", len(user_messages))


# Now we're passing those messages to our event extractor.
events = extract_events(user_messages, batch_size=10)


# Let's print the extracted events in a readable format.
print("\nExtracted project events:\n")

print(json.dumps(events, indent=4, ensure_ascii=False))


# We're checking whether every extracted event has the fields we expect.
for event in events:
    assert event["event_type"]
    assert event["subject"]
    assert event["timestamp"]
    assert event["evidence_message_ids"]

print("\nEvent extraction test completed successfully!")
