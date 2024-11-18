import argparse
import json
import sys
import os
from openai import OpenAI


def log_debug(msg, obj=None):
    """Conditional debug logging based on environment"""
    if "PROMPTFOO_ENABLE_DATABASE_LOGS" in os.environ:
        print(f"DEBUG - {msg}", file=sys.stderr)
        if obj is not None:
            print(f"DEBUG - {json.dumps(obj, indent=2, default=str)}", file=sys.stderr)


def call_api(prompt, options=None, context=None):
    """Main API handler"""
    try:
        # Parse Configuration
        options = options or {}
        config = options.get("config", {})

        # Parse model from provider ID
        provider_id = options.get("id", "")
        model = provider_id.split(":")[-1] if ":" in provider_id else "gpt-4o-mini"
        log_debug("Using model:", model)

        # Initialize Client
        client = OpenAI(
            api_key=options.get("api_key"),
            organization=options.get("organization"),
            base_url=options.get("base_url"),
        )

        # Parse Messages
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else json.loads(prompt) if isinstance(prompt, str) else prompt
        )

        # Build API Parameters
        api_params = {
            "model": model,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens"),
            "top_p": config.get("top_p", 1),
            "presence_penalty": config.get("presence_penalty", 0),
            "frequency_penalty": config.get("frequency_penalty", 0),
            "messages": messages,
        }

        # Add response format if provided
        if response_format := config.get("response_format"):
            api_params["response_format"] = response_format

        # Make API Call
        response = client.chat.completions.create(**api_params)
        content = response.choices[0].message.content

        # Try to parse as JSON, fallback to raw content if not valid JSON
        try:
            return {"output": json.loads(content)}
        except json.JSONDecodeError:
            return {"output": content}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--options", type=json.loads, default={})
    parser.add_argument("--context", type=json.loads, default={})

    args = parser.parse_args()
    result = call_api(args.prompt, args.options, args.context)
    print(json.dumps(result))
