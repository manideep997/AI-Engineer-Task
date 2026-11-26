
import json

from parser import parse_chat_file
from event_extractor import extract_events
from state_engine import build_project_state, print_project_state


# First, we're reading the WhatsApp chat and converting it into messages.
chat_file = "data/team_chat.txt"

print("\nReading WhatsApp chat...")
messages = parse_chat_file(chat_file)

print("Total messages:", len(messages))


# Now we're sending those messages to the LLM to identify project events.
print("\nExtracting project events...")
events = extract_events(messages, batch_size=10)

print("Total events extracted:", len(events))


# Let's save the extracted events so we can inspect them later.
with open("extracted_events.json", "w", encoding="utf-8") as file:
    json.dump(events, file, indent=4, ensure_ascii=False)


# We're passing the extracted events to the state engine.
print("\nBuilding current project state...")
project_state = build_project_state(events)


# Finally, let's display what the agent understood from the conversation.
print_project_state(project_state)


# We're also saving the final state for further testing.
with open("project_state.json", "w", encoding="utf-8") as file:
    json.dump(project_state, file, indent=4, ensure_ascii=False)


print("\nFull pipeline test completed!")
print("Saved extracted_events.json and project_state.json")
