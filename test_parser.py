
from datetime import datetime
from parser import parse_chat_file


# First, we're loading the sample WhatsApp chat using our parser.
chat_file = "data/team_chat.txt"
messages = parse_chat_file(chat_file)


# Let's quickly check how many messages our parser has extracted.
print("Total messages:", len(messages))

print("\nFirst 5 messages:\n")

# Printing a few messages helps us manually check whether the output looks correct.
for message in messages[:5]:
    print(message)
    print("-" * 60)

print("\nLast message:\n")
print(messages[-1])


# TEST 1: Check whether multiline messages are handled correctly.

# This is the task allocation message sent by Aarav.
# We're checking whether all the details are still present in that single message.
assignment = next(
    msg for msg in messages
    if "Task split karte hain" in msg["message"]
)

assert assignment["sender"] == "Aarav"
assert "Rohan - auth" in assignment["message"]
assert "Priya - listings UI" in assignment["message"]
assert "Meera - payments" in assignment["message"]
assert "Kabir - search" in assignment["message"]
assert "Main - deployment" in assignment["message"]

# The message should cover more than one original line.
assert assignment["line_end"] > assignment["line_start"]

print("\nPASS: Multiline messages are handled correctly.")


# TEST 2: Check whether media placeholders are preserved.

# WhatsApp exports sometimes contain placeholders instead of actual media.
media_messages = [
    msg for msg in messages
    if "<Media omitted>" in msg["message"]
]

assert len(media_messages) > 0

print("PASS: Media messages are preserved.")


# TEST 3: Check whether edited messages are preserved.

# We're making sure the edited-message notification is not lost.
edited_messages = [
    msg for msg in messages
    if "<This message was edited>" in msg["message"]
]

assert len(edited_messages) > 0

print("PASS: Edited messages are preserved.")


# TEST 4: Check whether deleted message notifications are preserved.

# We're checking whether WhatsApp's deleted-message text is still available.
deleted_messages = [
    msg for msg in messages
    if "This message was deleted" in msg["message"]
]

assert len(deleted_messages) > 0

print("PASS: Deleted message notifications are preserved.")


# TEST 5: Check whether messages are in chronological order.

# We're converting the stored timestamps back into datetime values
# and checking whether they are arranged from oldest to newest.
timestamps = [
    datetime.fromisoformat(msg["timestamp"])
    for msg in messages
]

assert timestamps == sorted(timestamps)

print("PASS: Messages are arranged in chronological order.")


print("\nAll parser validation tests passed!")
