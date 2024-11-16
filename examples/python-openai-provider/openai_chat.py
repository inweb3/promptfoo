import argparse
import json
import sys
from openai import OpenAI, OpenAIError, RateLimitError, APIError


def validate_messages(messages):
    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("Each message must be a dictionary")
        if "role" not in msg or "content" not in msg:
            raise ValueError("Each message must have 'role' and 'content' fields")
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError("Message role must be 'system', 'user', or 'assistant'")


def call_api(prompt, options, context):
    print("Custom Python OpenAI provider called!", file=sys.stderr)
    try:
        # Initialize OpenAI client
        client = OpenAI(
            api_key=options.get("api_key"),
            organization=options.get("organization"),
            base_url=options.get("base_url"),
        )

        # Parse the messages from the prompt
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = json.loads(prompt)
            validate_messages(messages)

        # Make the API call
        response = client.chat.completions.create(
            model=options.get("model", "gpt-4o-mini"),
            messages=messages,
            temperature=options.get("temperature", 0.7),
            max_tokens=options.get("max_tokens"),
            top_p=options.get("top_p"),
            frequency_penalty=options.get("frequency_penalty"),
            presence_penalty=options.get("presence_penalty"),
            stream=options.get("stream", False),
        )

        # Format the response
        result = {
            "output": response.choices[0].message.content,
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "cached": False,
        }

        return result

    except RateLimitError as e:
        return {"error": f"Rate limit exceeded: {str(e)}", "error_type": "rate_limit"}
    except APIError as e:
        return {"error": f"OpenAI API error: {str(e)}", "error_type": "api_error"}
    except OpenAIError as e:
        return {"error": f"OpenAI error: {str(e)}", "error_type": "openai_error"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "error_type": "unknown"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--options", type=json.loads, default={})
    parser.add_argument("--context", type=json.loads, default={})

    args = parser.parse_args()
    result = call_api(args.prompt, args.options, args.context)
    print(json.dumps(result))
