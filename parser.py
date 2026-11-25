
import re
from datetime import datetime


# See, every new WhatsApp message starts with a date and time.
# We're using this pattern to identify where each new message begins.
MESSAGE_PATTERN = re.compile(
    r"^(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2} [AP]M) - (.*)$"
)


def parse_chat_text(chat_text):
    messages = []
    current_message = None

    # First, we're splitting the complete chat into individual lines.
    lines = chat_text.splitlines()

    # Now we're going through each line one by one.
    # We're also keeping track of the original line number for reference.
    for line_number, line in enumerate(lines, start=1):

        # Here we're checking whether this line starts a new WhatsApp message.
        match = MESSAGE_PATTERN.match(line)

        if match:

            # If we were already reading another message, let's save it first.
            if current_message is not None:
                messages.append(current_message)

            # These three values are taken from the date and time pattern above.
            date_part = match.group(1)
            time_part = match.group(2)
            content = match.group(3)

            # We're combining the date and time and converting them into a
            # datetime value so that Python can work with them properly.
            timestamp = datetime.strptime(
                f"{date_part}, {time_part}",
                "%d/%m/%y, %I:%M %p"
            )

            # Here we're checking whether the message was sent by a group member
            # or whether it is just a WhatsApp system notification.
            if ": " in content:
                sender, message = content.split(": ", 1)
                message_type = "user"
            else:
                sender = None
                message = content
                message_type = "system"

            # We're storing all the details of this message in a dictionary.
            # The line numbers will help us find the original message later.
            current_message = {
                "id": len(messages) + 1,
                "timestamp": timestamp.isoformat(),
                "sender": sender,
                "message": message,
                "message_type": message_type,
                "line_start": line_number,
                "line_end": line_number
            }

        else:
            # Not every WhatsApp message fits into a single line.
            # If this line is a continuation of the previous message,
            # we're simply adding it to the message we're already reading.
            if current_message is not None:
                current_message["message"] += "\n" + line
                current_message["line_end"] = line_number

    # We've reached the end of the chat, so let's save the final message too.
    if current_message is not None:
        messages.append(current_message)

    # Now we're assigning IDs to all messages in the order they were parsed.
    for index, message in enumerate(messages, start=1):
        message["id"] = index

    return messages


def parse_chat_file(file_path):

    # Here we're opening the WhatsApp export and reading its complete content.
    # utf-8-sig also helps us handle files that contain a UTF-8 byte order mark.
    with open(file_path, "r", encoding="utf-8-sig") as file:
        chat_text = file.read()

    # Once the file is read, we're passing its content to our parsing function.
    return parse_chat_text(chat_text)
