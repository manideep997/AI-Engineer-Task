# Final Notes

## 1. How to run it

### Requirements
- Python 3.10+
- NVIDIA API key

### Setup

Clone the repository:
git clone https://github.com/manideep997/AI-Engineer-Task.git
cd AI-Engineer-Task

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

Install the dependencies:
pip install -r requirements.txt

Create a `.env` file in the project folder:

NVIDIA_API_KEY=your_api_key
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b

Run the project:
python run.py --chat data/team_chat.txt

To try another chat, provide its file path:

python run.py --chat path/to/chat.txt

The API key is required because the project uses an LLM for extracting events and answering questions.

## 2. How it works

I didn't want to send the entire WhatsApp conversation to an LLM and simply ask it to summarize everything. Instead, I split the problem into a few steps. First, the parser reads the chat and separates individual messages, including their timestamps and senders. Then I send messages in batches to the model to identify things like task assignments, decisions, deadlines and progress updates. These events are stored in a structured format. The state engine uses them to figure out what is currently happening in the project. Finally, the QA part retrieves relevant messages and uses them to answer questions. I also wanted the system to show where an answer came from, rather than just giving a statement without any evidence.

## 3. My test set and results

I prepared these questions using data/team_chat.txt. I worked out the expected answers by reading the conversation myself.

| # | Question | Correct answer (by hand) | System's answer | Correct? |
|---|---|---|---|---|
| 1 | Which tech stack did the team decide to use? | Django, React and PostgreSQL. | The team decided to use Django for the backend, React for the frontend and PostgreSQL as the database. | [Yes] |
| 2 | Who was initially assigned authentication? | Rohan. | Rohan was initially assigned the authentication task. | [Yes] |
| 3 | Who was handling the listings UI? | Priya. | Priya was assigned the listings UI and later shared updates about the page. | [Yes] |
| 4 | Who eventually took authentication from Rohan? | Kabir. | Kabir took over authentication from Rohan. | [Yes] |
| 5 | What happened to the UPI QR payment idea? | It was dropped after Razorpay test mode worked without KYC. | The team decided to use a UPI QR code because of the Razorpay KYC issue. | [No] |
| 6 | Which deployment platform was selected? | Render instead of AWS. | The system could not clearly identify the final deployment platform from the extracted events. | [No] |
| 7 | What was the latest database schema update? | Rohan said schema v2 was ready, including an orders table. | The system identified that Rohan was working on the database schema but missed the update about schema v2 and the orders table. | [No] |
| 8 | When was the demo scheduled? | November 26, 2026. | The demo was scheduled for November 26, 2026. | [Yes] |
| 9 | Who was preparing the report and presentation? | Meera. | Meera was responsible for the report and presentation. | [Yes] |
| 10 | Was someone finally assigned the search feature? | No confirmed reassignment is present in the latest discussion. | The system could not determine whether the search feature had a final owner. | [Yes] |
| 11 | How much of the chat feature was completed according to Kabir? | Around 70%. | The system did not retrieve the message containing Kabir's progress update. | [No] |
| 12 | Was the project actually deployed? | The chat mentions Render and a deployment plan, but doesn't confirm successful deployment. | The system treated the deployment plan as confirmation that deployment was completed. | [No] |

**Final score:** 7 / 12

I've counted an answer as correct only if it gets the important information right. If the chat doesn't confirm something, saying so is also a correct answer.
## 4. What it still gets wrong, and why

There are a few things I noticed while building and testing this.

1. Sometimes the model doesn't finish its response.

I faced this several times while extracting events. The API would return a response with finish_reason="length", which meant the JSON response was incomplete. I added retries, but they don't guarantee that every batch will succeed.

2. The model can misunderstand task ownership.

For example, someone discussing a task or giving an update doesn't necessarily mean that they own it. The system needs to distinguish between an initial assignment, a reassignment and a progress update.

3. Multiple tasks in one message are difficult.

A message can mention authentication, chat and payments together. There is a chance that the model combines them or extracts only some of them.

4. The final project state depends on the extracted events.

The state engine works with the events it receives. If the model misses an important update, the final task status can also be wrong.

5. Finding the right evidence isn't always easy.

The QA component uses retrieved messages to answer questions. Sometimes a question uses different words from the original conversation, so the relevant message might not be retrieved.

These are the main issues I would work on next.

## 5. What I'd do with two more days

1. Make extraction more reliable.

I would add checkpointing so that if an API request fails halfway through a conversation, the application can continue from the last successful batch instead of starting over.

2. Improve task tracking.

I would spend time improving how the system identifies tasks and handles ownership changes. I would also make sure that a task being discussed is not automatically treated as a newly assigned task.

3. Improve question answering.

I would improve evidence retrieval and test more questions, especially questions involving older decisions, changed deadlines and information that isn't explicitly mentioned.

I chose this order because extraction comes first. If the events are wrong, improving the question-answering part won't solve the underlying problem.

## 6. What I learned that I didn't know on day one

Before starting this, I hadn't really worked with a pipeline that turns normal conversations into structured project information.

Some things I learned during this project:

- Parsing WhatsApp exports is more complicated than splitting text line by line because messages can span multiple lines.
- An LLM response needs validation. Just because it returns JSON-looking text doesn't mean it is valid JSON.
- Processing messages in batches helps manage large conversations.
- The order of events matters when calculating the current state of a task.
- A task assignment and a task progress update are two different things.
- API failures and rate limits need to be considered when building an application around an external model.
- Showing evidence is important, especially when the system is answering questions about what people said.

I also learned that getting a response from an LLM is probably the easy part. Getting a consistent and trustworthy response is much harder.

 7. What I understand deeply vs. what I accepted from AI

I used AI quite a bit during development, especially when I got stuck with API errors and structured output.

What I understand and worked on:
How the WhatsApp parser processes messages.
Why the conversation needs to be converted into structured events.
How the different Python files connect to each other.
How the state engine processes events chronologically.
How the CLI takes a chat file as input.
How to configure the API and debug failures.
How to test the individual parts of the application.
Where I relied on AI assistance:
Improving the event extraction prompts.
Some of the Pydantic schema design
Retry handling and response validation.
Debugging NVIDIA and OpenRouter API differences.
Some parts of the question-answering implementation.
I don't want to claim that I wrote every line independently. I used AI as a development assistant, but I also wrote the code, examined the outputs and fixed issues during testing. There are implementation details that I still need to understand better, particularly some of the API handling and the prompts which i gave in the event extractor file.
8. Demo video
Link: https://go.screenpal.com/watch/cOXtFJn48Rv
Outputs i got from the agent:
PS C:\Users\smani\AI-Engineer-Task> python test_project_state.py
Messages parsed: 102
Extracting events from batch 1...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 2...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 3...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 4...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 5...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 6...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 7...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 8...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 9...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 10...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 11...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 12...

LLM response details:
Finish reason: stop
Received response successfully.
Extracting events from batch 13...

LLM response details:
Finish reason: stop
Received response successfully.
Events extracted: 39

========== CURRENT PROJECT STATE ==========

TASKS

Task: Authentication
Initial owner: Rohan
Current owner: Kabir
Status: in_progress
Latest update: Authentication work completed and merged. Chat progress is 70%.

Task: Listings UI
Initial owner: Priya
Current owner: Priya
Status: assigned
Latest update: Listings UI task assigned to Priya.

Task: Payments and Report/PPT
Initial owner: Meera
Current owner: Meera
Status: assigned
Latest update: Payments and report/PPT task assigned to Meera.

Task: Search and chat
Initial owner: Kabir
Current owner: Kabir
Status: assigned
Latest update: Search and chat between buyer/seller task assigned to Kabir.

Task: Deployment
Initial owner: Aarav
Current owner: Aarav
Status: assigned
Latest update: Deployment task assigned to Aarav.

Task: Payments
Initial owner: Meera
Current owner: Kabir
Status: assigned
Latest update: Payments responsibility transferred to Kabir. Meera assigned to reports.

Task: Listing page
Initial owner: Priya
Current owner: Priya
Status: in_progress
Latest update: Priya pushed the first version of the listing page.

Task: Domain purchase
Initial owner: Aarav
Current owner: Aarav
Status: assigned
Latest update: Domain was purchased from Namecheap for Rs 650.

Task: Search
Initial owner: Priya
Current owner: Rohan
Status: in_progress
Latest update: Rohan working on schema and tests. Planning to discuss search on Saturday.

Task: Razorpay Integration
Initial owner: Meera
Current owner: Meera
Status: in_progress
Latest update: Razorpay integration blocked due to missing business KYC. Account cannot be activated.

Task: Fake Payment Page
Initial owner: Rohan
Current owner: Rohan
Status: assigned
Latest update: Task to create fake payment page with UPI QR and manual confirmation assigned to Rohan.

Task: DB schema and tests
Initial owner: Rohan
Current owner: Rohan
Status: assigned
Latest update: Rohan assigned DB schema and tests work. Rohan confirmed he would handle schema and tests.

Task: UI listing + filters
Initial owner: Priya
Current owner: Priya
Status: in_progress
Latest update: Priya completed UI listing and filters work and was now doing seller dashboard.

Task: Seller dashboard
Initial owner: Priya
Current owner: Priya
Status: in_progress
Latest update: Priya was working on seller dashboard after completing UI listing and filters.

Task: Report + PPT
Initial owner: Meera
Current owner: Meera
Status: completed
Latest update: Meera completed the report and PPT.

Task: Chat, Auth, Payments
Initial owner: Kabir
Current owner: Kabir
Status: assigned
Latest update: Chat, authentication, and payments reassigned to Kabir.

Task: Database schema v2
Initial owner: Rohan
Current owner: Rohan
Status: in_progress
Latest update: Schema v2 ready with 'orders' table added.

Task: Search task
Initial owner: Not identified
Current owner: Not assigned
Status: not_started
Latest update: No details available

DECISIONS

Topic: Tech Stack
Current decision: Not confirmed yet

Topic: Frontend Framework
Current decision: Not confirmed yet

Topic: Database
Current decision: Postgres confirmed as the database for deployment.

Topic: Payment Method
Current decision: Final decision confirmed: UPI QR code + manual payment confirmation instead of Razorpay.

Topic: Payment provider
Current decision: Razorpay test mode finalized as payment provider. QR code integration scrapped.

DEADLINES

Topic: Report submission
Latest confirmed deadline: Report due on 23 Nov

Topic: Demo Day
Latest confirmed deadline: Demo day changed from 27 Nov to 26 Nov. Notice had a typo.

Topic: Report deadline
No confirmed deadline found.

Topic: Live deployment
Latest confirmed deadline: Live deployed link required for demo. Deployment target: Render, target date: 15 Nov

===========================================

PS C:\Users\smani\AI-Engineer-Task>
