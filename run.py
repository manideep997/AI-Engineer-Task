
import argparse
import json
import os

from parser import parse_chat_file
from event_extractor import extract_events
from state_engine import build_project_state, print_project_state
from qa_engine import answer_question, print_answer


def load_or_extract_events(messages, reuse_cache=False):
    cache_path = "data/extracted_events.json"

    if reuse_cache and os.path.exists(cache_path):
        print("\nLoading previously extracted events...")
        with open(cache_path, "r", encoding="utf-8") as file:
            events = json.load(file)
    else:
        print("\nExtracting project events...")
        events = extract_events(
            messages,
            batch_size=6,
            context_size=3
        )

        os.makedirs("data", exist_ok=True)

        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(events, file, indent=4, ensure_ascii=False)

    return events


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Project Chat Analysis Agent"
    )

    parser.add_argument(
        "--chat",
        required=True,
        help="Path to the WhatsApp chat text file"
    )

    parser.add_argument(
        "--question",
        help="Ask a question about the project"
    )

    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse previously extracted events instead of calling the API"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.chat):
        print(f"Error: Chat file not found: {args.chat}")
        return

    print("\n============================================")
    print("       WHATSAPP CHAT ANALYSIS AGENT")
    print("============================================")

    # Step 1: Parse the WhatsApp chat
    print("\n[1/4] Parsing chat...")

    messages = parse_chat_file(args.chat)

    user_messages = [
        message for message in messages
        if message["message_type"] == "user"
    ]

    print(f"Total messages parsed: {len(messages)}")
    print(f"User messages: {len(user_messages)}")

    if not user_messages:
        print("No user messages found in this chat.")
        return

    # Step 2: Extract project events
    print("\n[2/4] Extracting project events...")

    events = load_or_extract_events(
        messages,
        reuse_cache=args.reuse_cache
    )

    print(f"Events extracted: {len(events)}")

    # Step 3: Build the current project state
    print("\n[3/4] Building project state...")

    project_state = build_project_state(events)

    print("\n============================================")
    print("             PROJECT STATE")
    print("============================================")

    print_project_state(project_state)

    # Step 4: Answer questions
    print("\n[4/4] Question answering")

    if args.question:
        questions = [args.question]
    else:
        print("\nAsk questions about the project.")
        print("Type 'exit' to quit.")

        questions = None

    def ask(question):
        print("\nAnalyzing your question...")

        result = answer_question(
            question,
            messages,
            project_state
        )

        print_answer(result)

    if questions:
        for question in questions:
            ask(question)
    else:
        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in ["exit", "quit"]:
                print("\nExiting the agent. Goodbye!")
                break

            if not question:
                continue

            ask(question)


if __name__ == "__main__":
    main()
