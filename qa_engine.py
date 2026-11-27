import os
import re
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load API settings from .env
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


# Common words that do not help much with message retrieval.
STOP_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "me",
    "my",
    "we",
    "our",
    "they",
    "their",
    "this",
    "that",
    "it",
    "be",
    "been",
    "being",
    "about",
    "tell",
    "please",
    "can",
    "could",
    "should",
    "would",
    "any",
    "before",
}


# Instructions for the question-answering model.
SYSTEM_PROMPT = """
You are a WhatsApp project chat analysis assistant.

Your job is to answer questions using the supplied project state
and relevant original WhatsApp messages.

Follow these rules carefully:

1. Only use information supported by the supplied chat messages
   and project state.

2. Never invent owners, deadlines, decisions, or task statuses.

3. Distinguish between proposed, confirmed, completed,
   pending, and blocked work.

4. Consider earlier messages when understanding ownership
   changes and task reassignment.

5. For broad questions about pending issues, identify all
   relevant unfinished or blocked tasks mentioned in the evidence.

6. If information is missing, say:
   "The chat doesn't say."

7. Answer in the same language as the user's question.
   English and Hinglish are both supported.

8. Include the message IDs that support your answer.

9. Do not treat a proposed decision as a confirmed decision.

10. Do not claim that a task is completed unless the chat
    provides evidence of completion.

Return ONLY valid JSON in this format:

{
    "answer": "Your answer here",
    "evidence_message_ids": [12, 15],
    "confidence": "HIGH"
}

Confidence must be HIGH, MEDIUM, or LOW.
"""


def create_qa_client():
    """
    Create the NVIDIA API client.
    """

    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is missing. " "Please check your .env file.")

    return OpenAI(
        base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=180.0, max_retries=1
    )


def tokenize(text):
    """
    Convert text into lowercase keywords.
    """

    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return set(words) - STOP_WORDS


def expand_query_terms(words):
    """
    Expand question keywords with related terms.

    This helps retrieve messages even when the user
    and chat use different words for the same topic.
    """

    synonyms = {
        # Authentication
        "authentication": {"authentication", "auth", "login", "signup"},
        "auth": {"auth", "authentication", "login", "signup"},
        # Payments
        "payments": {"payments", "payment", "razorpay", "upi"},
        "payment": {"payments", "payment", "razorpay", "upi"},
        # Deployment
        "deployment": {"deployment", "deploy", "render", "hosting"},
        # Database
        "database": {"database", "db", "postgres", "postgresql", "sqlite"},
        # Pending issues
        "issues": {
            "issues",
            "issue",
            "problems",
            "problem",
            "pending",
            "blocked",
            "unfinished",
        },
        "pending": {
            "pending",
            "remaining",
            "unfinished",
            "blocked",
            "tests",
            "webhook",
            "search",
            "chat",
            "deployment",
        },
        # Demo and deadlines
        "demo": {
            "demo",
            "deadline",
            "tests",
            "webhook",
            "deployment",
            "search",
            "report",
        },
        # Tasks
        "tasks": {"tasks", "task", "work", "assigned", "assignment", "responsibility"},
        "responsible": {"responsible", "owner", "owns", "assigned", "handling"},
        # Progress
        "progress": {
            "progress",
            "working",
            "completed",
            "done",
            "pending",
            "unfinished",
        },
    }

    expanded_words = set(words)

    for word in words:
        expanded_words.update(synonyms.get(word, set()))

    return expanded_words


def retrieve_relevant_messages(question, messages, top_k=12):
    """
    Retrieve the most relevant messages using keyword overlap.

    Messages are returned in chronological order.
    """

    question_words = expand_query_terms(tokenize(question))

    if not question_words:
        return []

    scored_messages = []

    for message in messages:
        message_text = message["message"].lower()

        message_words = tokenize(message_text)

        overlap = question_words.intersection(message_words)

        score = len(overlap)

        # Give extra weight to exact phrase matches.
        if question.lower() in message_text:
            score += 5

        if score > 0:
            scored_messages.append((score, message))

    # Sort by relevance.
    scored_messages.sort(key=lambda item: item[0], reverse=True)

    # Select the highest-scoring messages.
    results = [message for _, message in scored_messages[:top_k]]

    # Restore chronological order.
    results.sort(key=lambda message: message["timestamp"])

    return results


def extract_json_response(response_text):
    """
    Extract JSON from the model response.
    """

    response_text = response_text.strip()

    # Remove Markdown code fences if present.
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text)

        response_text = re.sub(r"\s*```$", "", response_text)

    try:
        return json.loads(response_text)

    except json.JSONDecodeError:
        # Try extracting the JSON object.
        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if match:
            return json.loads(match.group())

        raise ValueError("The model did not return valid JSON.")


def format_messages(messages):
    """
    Format retrieved messages for the LLM.
    """

    formatted = []

    for message in messages:
        formatted.append(
            {
                "message_id": message["id"],
                "timestamp": message["timestamp"],
                "sender": message["sender"],
                "message": message["message"],
            }
        )

    return json.dumps(formatted, indent=2, ensure_ascii=False)


def answer_question(question, messages, project_state, top_k=20):
    """
    Answer a project-related question using the chat
    messages and reconstructed project state.
    """

    # Step 1: Retrieve relevant messages.
    relevant_messages = retrieve_relevant_messages(question, messages, top_k=top_k)

    if not relevant_messages:
        return {
            "question": question,
            "answer": "The chat doesn't say.",
            "confidence": "LOW",
            "evidence": [],
        }

    # Step 2: Prepare evidence lookup.
    evidence_lookup = {message["id"]: message for message in relevant_messages}

    # Step 3: Prepare the model input.
    user_prompt = f"""
QUESTION:
{question}

CURRENT PROJECT STATE:
{json.dumps(project_state, indent=2, ensure_ascii=False, default=str)}

RELEVANT WHATSAPP MESSAGES:
{format_messages(relevant_messages)}

Answer the question using the supplied information.

Remember:
- Include only relevant evidence message IDs.
- Do not invent missing information.
- For questions about pending issues, combine all relevant
  unfinished or blocked work supported by the messages.
- Return valid JSON only.
"""

    # Step 4: Call NVIDIA.
    client = create_qa_client()

    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("The model returned an empty response.")

    # Step 5: Parse model response.
    result = extract_json_response(response_text)

    answer = result.get("answer", "The chat doesn't say.")

    confidence = result.get("confidence", "LOW")

    confidence = str(confidence).upper()

    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        confidence = "LOW"

    # Step 6: Validate evidence IDs.
    raw_evidence_ids = result.get("evidence_message_ids", [])

    valid_evidence_ids = []

    for evidence_id in raw_evidence_ids:
        try:
            evidence_id = int(evidence_id)
        except (TypeError, ValueError):
            continue

        if evidence_id in evidence_lookup:
            if evidence_id not in valid_evidence_ids:
                valid_evidence_ids.append(evidence_id)

    # Do not return unsupported answers.
    if not valid_evidence_ids:
        answer = "The chat doesn't say."
        confidence = "LOW"

    # Step 7: Prepare evidence details.
    evidence = []

    for evidence_id in valid_evidence_ids:
        message = evidence_lookup[evidence_id]

        evidence.append(
            {
                "message_id": message["id"],
                "timestamp": message["timestamp"],
                "sender": message["sender"],
                "message": message["message"],
            }
        )

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
    }


def print_answer(result):
    """
    Display the answer and its supporting evidence.
    """

    print("\n" + "=" * 65)
    print("QUESTION")
    print("=" * 65)

    print(result["question"])

    print("\n" + "=" * 65)
    print("ANSWER")
    print("=" * 65)

    print(result["answer"])

    print("\nCONFIDENCE: " + result["confidence"])

    print("\n" + "=" * 65)
    print("EVIDENCE")
    print("=" * 65)

    if not result["evidence"]:
        print("No supporting messages found.")
        return

    for item in result["evidence"]:
        print(f"\nMessage ID: {item['message_id']}")

        print(f"Timestamp: {item['timestamp']}")

        print(f"Sender: {item['sender']}")

        print(f"Message: {item['message']}")
