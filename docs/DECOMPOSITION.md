# Project Decomposition

## 1. The Problem in My Own Words

The goal of this project is to build an agent that can read WhatsApp group chat exports and answer questions about the project being discussed.

The agent should understand how conversations develop over time. For example, if a task is initially assigned to one person and later transferred to someone else, it should identify the latest responsibility instead of returning outdated information.

It should also identify project decisions, deadlines, completed tasks, pending work and unassigned responsibilities. Every answer should be supported by actual messages from the conversation. If the chat does not contain enough information, the agent should clearly say so instead of making assumptions.

## 2. What I Noticed in the Data

After going through the provided `team_chat.txt`, I noticed a few challenges:

->Changing decisions: The team initially decided to use UPI QR with manual confirmation, but later switched to Razorpay test mode.
->Changing responsibilities: Authentication was initially assigned to Rohan but was later taken over by Kabir.
->Changing deadlines: The demo date was initially December 4, later changed to November 27, and finally corrected to November 26.
->Incomplete work: Rohan promised to complete the tests but later mentioned that they were still pending.
->Different progress levels: Authentication was completed, while chat functionality was only 70% complete.
->Unassigned tasks: Search functionality still did not have a confirmed owner.
->Project requirements: The team was informed that a live deployed link was required for the demo.

These examples show that the agent needs to understand the context and history of conversations rather than simply picking information from individual messages.

## 3. The Simplest Approach and Its Limitations

The simplest approach would be to send the entire chat file to an LLM and ask it to answer questions or summarize the project.

However, this approach may not always capture the complete context. For example, it needs to understand that Rohan's original authentication assignment was transferred to Kabir, that the UPI QR decision was later changed to Razorpay test mode, and that the final demo date is November 26.

The main challenge is that not every message represents a confirmed decision. Some messages are suggestions, promises, questions or updates.

Therefore, I plan to extract important information, organize it into structured events and maintain an updated project state. This will help the agent provide reliable answers with supporting evidence.

## 4. Breaking the Problem into Smaller Parts

I plan to divide the system into the following components:

WhatsApp Parser: Reads the TXT file and extracts individual messages with timestamps, sender names and message content.

Message Understanding:Identifies useful information such as decisions, assignments, deadlines, progress updates and corrections. I plan to use an LLM for understanding informal English and Hinglish conversations.

Event Extraction: Converts important messages into structured events containing details such as the event type, task, person involved, timestamp and original message reference.

Project State Reconstruction: Processes events chronologically to identify the latest decisions, task owners, deadlines and progress. It should also handle situations where earlier decisions or responsibilities change.

Evidence Retrieval: Finds the original messages that support a particular answer. I will initially try keyword-based retrieval and explore semantic search if required.

Question Answering: Uses the reconstructed project state and relevant messages to answer questions in English or Hinglish, while providing supporting evidence.

Project Status Generation: Produces an overall summary of current decisions, responsibilities, completed work, pending tasks and unassigned responsibilities.

## 5. My Current Timeline

For the past few days, I have been occupied with my college examinations, so I couldn't dedicate as much time to the assignment as I initially planned.

I plan to complete the agent by the morning of **November 27**. On the same day, I will evaluate it using different questions, verify its responses against the original chat messages and check whether all the components are working as expected.

Since the project involves multiple stages, there might be some issues during integration or testing. I will continuously debug, make the necessary changes and improve the agent based on the evaluation results. My goal is to make sure the final implementation is reliable and works as expected.

## 6. Build Plan

I plan to develop the agent in stages:

1. Build the WhatsApp parser to extract messages correctly.
2. Connect an LLM and implement basic question answering.
3. Introduce structured event extraction for decisions, assignments and progress updates.
4. Build the project state engine to track changes chronologically.
5. Add evidence retrieval so that answers can be traced back to original messages.
6. Generate a project status report and test the complete system.

I will start with a basic working version and gradually improve its accuracy and reliability.

## 7. Where I Will Use AI and Where I Won't

I'm planning to use an LLM for understanding natural language, for some poweshell commands, identifying important events and generating answers.

However, I will use normal Python logic for reading the TXT file, extracting timestamps and sender names, maintaining message order, tracking event history and validating structured information.

I also want to keep the project state and evidence references organized so that the agent does not depend entirely on the LLM to remember previous conversations.

My overall approach is to use AI for language understanding and state management and verification of agent. This should make the system easier to debug, test and explain during the interview.