
import os
from dotenv import load_dotenv
from openai import OpenAI


# Here we're loading the API key from our .env file.
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# Before making a request, let's check whether the API key is available.
if not api_key:
    print("API key not found. Please check your .env file.")
    raise SystemExit


# We're connecting the OpenAI client to OpenRouter instead of OpenAI.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


try:
    # Let's send a simple message to check whether the API is responding.
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": "Say hello and confirm that the API is working."
            }
        ]
    )

    # If we get a response, our API connection is working.
    print("\nAPI connection successful!")
    print("Model response:")
    print(response.choices[0].message.content)

except Exception as error:
    # If something goes wrong, we'll print the error to understand why.
    print("\nAPI request failed.")
    print("Error:", error)
