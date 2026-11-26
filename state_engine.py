
from collections import defaultdict


def clean_subject(subject):
    # Sometimes the same task appears with slightly different spacing or capitalisation.
    # We're cleaning the subject so those events can be grouped together.
    return " ".join(subject.strip().lower().split())


def build_project_state(events):
    tasks = {}
    decisions = {}
    deadlines = {}
    other_events = []

    # We're sorting the events by timestamp before processing them.
    # This helps us understand what happened first and what changed later.
    sorted_events = sorted(
        events,
        key=lambda event: event["timestamp"]
    )

    for event in sorted_events:
        event_type = event["event_type"]
        subject = event["subject"]
        subject_key = clean_subject(subject)

        # First, we're handling events related to tasks.
        if event_type in [
            "task_assigned",
            "task_reassigned",
            "task_progress",
            "task_completed",
            "task_blocked"
        ]:

            # If this is the first time we're seeing this task,
            # we're creating a place to store its current information and history.
            if subject_key not in tasks:
                tasks[subject_key] = {
                    "subject": subject,
                    "owner": None,
                    "status": "not_started",
                    "details": None,
                    "history": []
                }

            task = tasks[subject_key]

            # Every event is added to the history so we don't lose earlier updates.
            task["history"].append(event)

            # An assignment or reassignment updates the current task owner.
            if event_type in ["task_assigned", "task_reassigned"]:
                if event.get("owner"):
                    task["owner"] = event["owner"]

                if task["status"] == "not_started":
                    task["status"] = "assigned"

            # A progress update means someone has started working on the task.
            elif event_type == "task_progress":
                task["status"] = "in_progress"

            # If the team says the task is completed, we update its status.
            elif event_type == "task_completed":
                task["status"] = "completed"

            # If someone reports a blocker, we mark the task as blocked.
            elif event_type == "task_blocked":
                task["status"] = "blocked"

            # We're keeping the latest details available for this task.
            if event.get("details"):
                task["details"] = event["details"]

        # Next, we're handling project decisions.
        elif event_type in [
            "decision_proposed",
            "decision_confirmed",
            "decision_changed"
        ]:

            if subject_key not in decisions:
                decisions[subject_key] = {
                    "subject": subject,
                    "current_decision": None,
                    "proposals": [],
                    "history": []
                }

            decision = decisions[subject_key]
            decision["history"].append(event)

            # A proposal is stored separately because it isn't necessarily final.
            if event_type == "decision_proposed":
                decision["proposals"].append(event)

            # A confirmed or changed decision becomes the current decision.
            elif event_type in [
                "decision_confirmed",
                "decision_changed"
            ]:
                decision["current_decision"] = event

        # We're keeping deadline events separately for each subject.
        elif event_type == "deadline_set":
            if subject_key not in deadlines:
                deadlines[subject_key] = {
                    "subject": subject,
                    "current_deadline": None,
                    "history": []
                }

            deadlines[subject_key]["history"].append(event)
            deadlines[subject_key]["current_deadline"] = event

        # We don't want to throw away meaningful events that don't fit
        # into the task, decision or deadline categories.
        else:
            other_events.append(event)

    return {
        "tasks": tasks,
        "decisions": decisions,
        "deadlines": deadlines,
        "other_events": other_events
    }


def print_project_state(project_state):
    # This function displays the latest project information in a readable way.
    print("\n========== CURRENT PROJECT STATE ==========\n")

    print("TASKS")

    if not project_state["tasks"]:
        print("No task information found.")

    for task in project_state["tasks"].values():
        print(f"\nTask: {task['subject']}")
        print(f"Owner: {task['owner'] or 'Not assigned'}")
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
        print(f"Latest deadline update: {current.get('details')}")

    print("\n===========================================\n")
