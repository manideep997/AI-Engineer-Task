"""
Test the event extractor against a WhatsApp export.

Usage:
    python test_event_extractor.py
    python test_event_extractor.py --chat data/team_chat.txt
    python test_event_extractor.py --chat path/to/another_chat.txt --batch-size 1

The script first tries to use the project's parser.py. If it cannot find a
known parser function, it uses a small WhatsApp-export fallback parser for the
standard text-export format.
"""

import argparse
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def _normalise_parser_result(result):
    """Accept common parser return shapes and extract the message list."""

    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("messages", "parsed_messages", "data"):
            if isinstance(result.get(key), list):
                return result[key]
    raise TypeError(
        "The parser function did not return a list of messages "
        "or a dictionary containing a messages list."
    )


def _try_project_parser(chat_path):
    """
    Try common public parser function names from parser.py.

    This keeps the test script compatible with small naming differences
    between parser implementations.
    """

    try:
        parser_module = importlib.import_module("parser")
    except Exception as error:
        print(f"Could not import parser.py ({error}); using fallback parser.")
        return None

    function_names = (
        "parse_whatsapp_chat",
        "parse_whatsapp_file",
        "parse_chat_file",
        "parse_chat",
        "parse_whatsapp_messages",
        "parse_messages",
        "parse_file",
    )

    chat_text = chat_path.read_text(encoding="utf-8-sig")
    for function_name in function_names:
        function = getattr(parser_module, function_name, None)
        if not callable(function):
            continue

        for argument in (str(chat_path), chat_text):
            try:
                result = function(argument)
                messages = _normalise_parser_result(result)
                print(f"Using parser.py function: {function_name}")
                return messages
            except TypeError:
                continue
            except (FileNotFoundError, OSError):
                continue
            except Exception as error:
                print(
                    f"Parser function {function_name} could not parse the "
                    f"file ({type(error).__name__}: {error}); trying fallback."
                )
                break

    print("No compatible parser function found; using fallback parser.")
    return None


def _fallback_parse_whatsapp_export(chat_path):
    """
    Minimal parser for standard WhatsApp text exports.

    It supports multiline messages. For full parser-specific features such as
    edit/delete/system-event classification, use the project's parser.py.
    """

    start_pattern = re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*"
        r"(\d{1,2}:\d{2}\s*[APap][Mm])\s*-\s*(.*)$"
    )

    messages = []
    current = None

    def flush():
        if current is not None:
            current["message"] = "\n".join(current.pop("_lines")).strip()
            messages.append(current.copy())

    for line in chat_path.read_text(encoding="utf-8-sig").splitlines():
        match = start_pattern.match(line)
        if match:
            flush()
            date_part, time_part, remainder = match.groups()
            sender, separator, body = remainder.partition(": ")
            if not separator:
                # A system notification, not a sender-authored message.
                current = None
                continue

            timestamp_text = f"{date_part}, {time_part}"
            parsed_timestamp = None
            for date_format in (
                "%d/%m/%y, %I:%M %p",
                "%d/%m/%Y, %I:%M %p",
                "%m/%d/%y, %I:%M %p",
                "%m/%d/%Y, %I:%M %p",
            ):
                try:
                    parsed_timestamp = datetime.strptime(
                        timestamp_text, date_format
                    )
                    break
                except ValueError:
                    pass

            current = {
                "id": len(messages) + 1,
                "timestamp": (
                    parsed_timestamp.isoformat(sep=" ")
                    if parsed_timestamp else timestamp_text
                ),
                "sender": sender.strip(),
                "message_type": "user",
                "_lines": [body],
            }
        elif current is not None:
            current["_lines"].append(line)

    flush()
    return messages


def _validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("No messages were parsed from the chat file.")

    required = ("id", "timestamp", "sender", "message")
    for index, message in enumerate(messages, start=1):
        missing = [key for key in required if key not in message]
        if missing:
            raise ValueError(
                f"Parsed message #{index} is missing fields: {missing}. "
                f"Message: {message}"
            )
        message.setdefault("message_type", "user")

    return messages


def main():
    project_root = Path(__file__).resolve().parent

    argument_parser = argparse.ArgumentParser(
        description="Test WhatsApp project-event extraction with OpenRouter."
    )
    argument_parser.add_argument(
        "--chat",
        default=str(project_root / "data" / "team_chat.txt"),
        help="Path to the WhatsApp .txt export.",
    )
    argument_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of new messages per model request (default: 2).",
    )
    argument_parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="Number of previous messages supplied as context (default: 2).",
    )
    argument_parser.add_argument(
        "--output",
        default=str(project_root / "data" / "extracted_events.json"),
        help="Where to save extracted events.",
    )
    args = argument_parser.parse_args()

    chat_path = Path(args.chat).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not chat_path.exists():
        raise FileNotFoundError(f"Chat file not found: {chat_path}")

    # Load environment variables before importing the extractor.
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    messages = _try_project_parser(chat_path)
    if messages is None:
        messages = _fallback_parse_whatsapp_export(chat_path)
    messages = _validate_messages(messages)

    print(f"Parsed messages: {len(messages)}")
    print(f"Chat file: {chat_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context size: {args.context_size}")

    from event_extractor import extract_events

    events = extract_events(
        messages,
        batch_size=args.batch_size,
        context_size=args.context_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(events, output_file, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(events)} events to: {output_path}")

    print("\nExtracted project events:")
    if not events:
        print("No project events were extracted.")
    else:
        for event in events:
            print(
                f"- {event['event_type']} | {event['subject']} | "
                f"actor={event.get('actor')} | owner={event.get('owner')} | "
                f"status={event.get('confirmation_status')} | "
                f"message_id={event['event_message_id']}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nTEST FAILED: {type(error).__name__}: {error}")
        sys.exit(1)
