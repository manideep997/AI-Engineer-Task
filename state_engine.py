
# These names refer to the same task or topic in our chat.
# We're using these aliases to avoid creating duplicate entries.

TASK_ALIASES = {
    "auth (login/signup)": "authentication",
    "auth": "authentication",
    "authentication": "authentication",

    "listings ui": "listings ui",
    "listing page first version": "listings ui",
    "ui listing and filters": "listings ui",

    "deployment": "deployment",
    "render deployment": "deployment",
}

DECISION_ALIASES = {
    "tech stack": "tech stack",
    "technology stack": "technology stack",

    "payment method": "payment gateway",
    "payment gateway": "payment gateway",
    "payment gateway (razorpay)": "payment gateway",

    "deployment platform": "deployment platform",
    "deployment": "deployment platform",
}

DEADLINE_ALIASES = {
    "project report": "final report",
    "report": "final report",
    "report deadline": "final report",
    "final report": "final report",

    "demo day": "demo day",
    "demo day date": "demo day",

    "tests": "tests",
}


def clean_subject(subject):
    # First, we're making the subject consistent.
    return " ".join(subject.strip().lower().split())


def get_canonical_subject(subject, aliases):
    # If we know two names refer to the same thing, we'll use one common name.
    cleaned_subject = clean_subject(subject)

    return aliases.get(cleaned_subject, cleaned_subject)


def build_project_state(events):
    tasks = {}
    decisions = {}
    deadlines = {}
    other_events = []

    # Events need to be processed in chronological order.
    sorted_events = sorted(
        events,
        key=lambda event: event["timestamp"]
    )

    for event in sorted_events:
        event_type = event["event_type"]
        subject = event["subject"]
        confirmation = event.get("confirmation_status", "confirmed")

        # ---------------- TASKS ----------------

        if event_type in [
            "task_assigned",
            "task_reassigned",
            "task_progress",
            "task_completed",
            "task_blocked"
        ]:
            subject_key = get_canonical_subject(subject, TASK_ALIASES)

            if subject_key not in tasks:
                tasks[subject_key] = {
                    "subject": subject,
                    "initial_owner": None,
                    "owner": None,
                    "status": "not_started",
                    "details": None,
                    "history": []
                }

            task = tasks[subject_key]
            task["history"].append(event)

            # We keep proposed changes in history, but don't treat them
            # as confirmed changes to the task.
            if confirmation in ["proposed", "uncertain", "rejected"]:
                continue

            # If an owner is explicitly mentioned, update the current owner.
            # This also lets progress and completion events update ownership.
            new_owner = event.get("owner")

            if new_owner:
                if task["initial_owner"] is None:
                    task["initial_owner"] = new_owner

                task["owner"] = new_owner

            if event_type == "task_assigned":
                if task["status"] != "completed":
                    task["status"] = "assigned"

            elif event_type == "task_reassigned":
                if task["status"] != "completed":
                    task["status"] = "assigned"

            elif event_type == "task_progress":
                task["status"] = "in_progress"

            elif event_type == "task_completed":
                task["status"] = "completed"

            elif event_type == "task_blocked":
                task["status"] = "blocked"

            if event.get("details"):
                task["details"] = event["details"]

        # ---------------- DECISIONS ----------------

        elif event_type in [
            "decision_proposed",
            "decision_confirmed",
            "decision_changed"
        ]:
            subject_key = get_canonical_subject(
                subject,
                DECISION_ALIASES
            )

            if subject_key not in decisions:
                decisions[subject_key] = {
                    "subject": subject,
                    "current_decision": None,
                    "proposals": [],
                    "history": []
                }

            decision = decisions[subject_key]
            decision["history"].append(event)

            if event_type == "decision_proposed":
                decision["proposals"].append(event)

            elif event_type in [
                "decision_confirmed",
                "decision_changed"
            ]:
                if confirmation == "confirmed":
                    decision["current_decision"] = event

        # ---------------- DEADLINES ----------------

        elif event_type in ["deadline_set", "deadline_changed"]:
            subject_key = get_canonical_subject(
                subject,
                DEADLINE_ALIASES
            )

            if subject_key not in deadlines:
                deadlines[subject_key] = {
                    "subject": subject,
                    "current_deadline": None,
                    "proposals": [],
                    "history": []
                }

            deadline = deadlines[subject_key]
            deadline["history"].append(event)

            if confirmation in ["proposed", "uncertain"]:
                deadline["proposals"].append(event)

            elif confirmation == "confirmed":
                deadline["current_deadline"] = event

        else:
            other_events.append(event)

    return {
        "tasks": tasks,
        "decisions": decisions,
        "deadlines": deadlines,
        "other_events": other_events
    }


def print_project_state(project_state):
    print("\n========== CURRENT PROJECT STATE ==========\n")

    print("TASKS")

    if not project_state["tasks"]:
        print("No task information found.")

    for task in project_state["tasks"].values():
        print(f"\nTask: {task['subject']}")
        print(f"Initial owner: {task['initial_owner'] or 'Not identified'}")
        print(f"Current owner: {task['owner'] or 'Not assigned'}")
        print(f"Status: {task['status']}")
        print(f"Latest update: {task['details'] or 'No details available'}")

    print("\nDECISIONS")

    if not project_state["decisions"]:
        print("No decision information found.")

    for decision in project_state["decisions"].values():
        current = decision["current_decision"]

        print(f"\nTopic: {decision['subject']}")

        if current:
            print(f"Current decision: {current.get('details')}")
        else:
            print("Current decision: Not confirmed yet")

    print("\nDEADLINES")

    if not project_state["deadlines"]:
        print("No deadline information found.")

    for deadline in project_state["deadlines"].values():
        current = deadline["current_deadline"]

        print(f"\nTopic: {deadline['subject']}")

        if current:
            print(f"Latest confirmed deadline: {current.get('details')}")
        else:
            print("No confirmed deadline found.")

    print("\n===========================================\n")
