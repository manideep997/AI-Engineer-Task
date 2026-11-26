
from state_engine import build_project_state, print_project_state


# We're creating a few sample events to check whether the state engine works.
# These events represent changes that happened during a project.

sample_events = [

    {
        "event_type": "task_assigned",
        "subject": "Authentication",
        "owner": "Rohan",
        "details": "Rohan was assigned authentication work.",
        "evidence_message_ids": [10],
        "timestamp": "2026-10-05T10:00:00"
    },

    {
        "event_type": "task_reassigned",
        "subject": "Authentication",
        "owner": "Kabir",
        "details": "Kabir took over authentication from Rohan.",
        "evidence_message_ids": [20],
        "timestamp": "2026-10-24T14:00:00"
    },

    {
        "event_type": "task_completed",
        "subject": "Authentication",
        "owner": "Kabir",
        "details": "Authentication is completed and merged.",
        "evidence_message_ids": [30],
        "timestamp": "2026-11-10T11:00:00"
    },

    {
        "event_type": "decision_proposed",
        "subject": "Tech stack",
        "owner": "Aarav",
        "details": "React and Firebase were proposed.",
        "evidence_message_ids": [5],
        "timestamp": "2026-10-03T10:00:00"
    },

    {
        "event_type": "decision_confirmed",
        "subject": "Tech stack",
        "owner": "Aarav",
        "details": "Django backend and React frontend were confirmed.",
        "evidence_message_ids": [12],
        "timestamp": "2026-10-03T11:00:00"
    },

    {
        "event_type": "deadline_set",
        "subject": "Final report",
        "owner": "Meera",
        "details": "The final report is due on November 23.",
        "evidence_message_ids": [40],
        "timestamp": "2026-11-09T12:00:00"
    }
]


# Now we're passing the sample events to our state engine.
project_state = build_project_state(sample_events)


# Let's display the result to see what the engine has understood.
print_project_state(project_state)


# TEST 1: Check whether the latest task owner is correct.
authentication = project_state["tasks"]["authentication"]

assert authentication["owner"] == "Kabir"

print("PASS: Task reassignment is handled correctly.")


# TEST 2: Check whether the latest task status is correct.
assert authentication["status"] == "completed"

print("PASS: Task completion is handled correctly.")


# TEST 3: Check whether the earlier task history is preserved.
assert len(authentication["history"]) == 3

print("PASS: Earlier task history is preserved.")


# TEST 4: Check whether the confirmed decision is selected.
tech_stack = project_state["decisions"]["tech stack"]

assert (
    tech_stack["current_decision"]["event_type"]
    == "decision_confirmed"
)

assert "Django" in tech_stack["current_decision"]["details"]

print("PASS: Confirmed decisions are identified correctly.")


# TEST 5: Check whether the original proposal is still available.
assert len(tech_stack["proposals"]) == 1

print("PASS: Earlier proposals are preserved.")


# TEST 6: Check whether the deadline is stored correctly.
report_deadline = project_state["deadlines"]["final report"]

assert "November 23" in report_deadline["current_deadline"]["details"]

print("PASS: Project deadlines are stored correctly.")


print("\nAll state engine tests passed!")
