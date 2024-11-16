import os
import json
from openai_chat import call_api


def test_chat():
    # Test case 1: Simple string prompt
    options = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
    }

    result = call_api("Say hello!", options)
    print("Test 1 - Simple prompt:")
    print(json.dumps(result, indent=2))
    print("\n")

    # Test case 2: Messages array
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's 2+2?"},
    ]
    result = call_api(json.dumps(messages), options)
    print("Test 2 - Messages array:")
    print(json.dumps(result, indent=2))
    print("\n")

    # Test case 3: Error handling (invalid API key)
    bad_options = {"api_key": "invalid_key", "model": "gpt-4o-mini"}
    result = call_api("This should fail", bad_options)
    print("Test 3 - Error handling:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_chat()
