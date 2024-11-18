# openai_chat_debug.py - Debug version of the OpenAI Chat API script
# openai_chat_debug.py
import argparse
import json
import sys
import traceback
from openai import OpenAI


def debug_log(msg, obj=None, level="INFO"):
    """Enhanced debug logging with levels"""
    print(f"DEBUG[{level}] - {msg}", file=sys.stderr)
    if obj is not None:
        if isinstance(obj, str):
            print(f"DEBUG[{level}] - (string length: {len(obj)}):", file=sys.stderr)
            print(
                f"DEBUG[{level}] - {obj[:1000]}{'...' if len(obj) > 1000 else ''}",
                file=sys.stderr,
            )
        else:
            print(
                f"DEBUG[{level}] - {json.dumps(obj, indent=2, default=str)}",
                file=sys.stderr,
            )


def call_api(prompt, options=None, context=None):
    """Main API handler with enhanced debugging"""
    try:
        debug_log("Starting API call with prompt:", prompt, "START")
        debug_log("Raw options received:", options, "START")
        debug_log("Raw context received:", context, "START")
        
        # Add logging for test context if present
        if context and 'test' in context:
            debug_log("Test configuration:", context['test'], "TEST-CONFIG")

        options = options or {}
        config = options.get("config", {})

        # Initialize OpenAI client
        client = OpenAI(
            api_key=options.get("api_key"),
            organization=options.get("organization"),
            base_url=options.get("base_url"),
        )

        # Parse messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = json.loads(prompt) if isinstance(prompt, str) else prompt
        debug_log("Parsed messages:", messages, "INIT")

        # Set model version explicitly
        model = "gpt-4o-mini"
        debug_log("Using model:", model, "CONFIG")

        # Basic API parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens"),
            "top_p": config.get("top_p", 1),
            "presence_penalty": config.get("presence_penalty", 0),
            "frequency_penalty": config.get("frequency_penalty", 0),
        }

        # Handle response format
        response_format = config.get("response_format")
        if response_format:
            debug_log("Response format from config:", response_format, "CONFIG")
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": response_format.get("json_schema", {}),
            }
            debug_log("Final response format:", api_params["response_format"], "CONFIG")

        debug_log("Final API parameters:", api_params, "API")

        # Make API call
        response = client.chat.completions.create(**api_params)
        debug_log("Raw API response:", response.model_dump(), "RESPONSE")

        # Extract and parse content
        content = response.choices[0].message.content
        debug_log("Response content:", content, "CONTENT")

        # Parse the JSON response
        parsed_content = json.loads(content)
        debug_log("Parsed content:", parsed_content, "RESULT")

        # Return with the required "output" key
        return {
            "output": parsed_content
        }

    except Exception as e:
        debug_log(f"Error: {str(e)}", None, "ERROR")
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--options", type=json.loads, default={})
    parser.add_argument("--context", type=json.loads, default={})

    args = parser.parse_args()
    debug_log("Command line args:", vars(args), "TEST")
    debug_log("Context from args:", args.context, "TEST")

    result = call_api(args.prompt, args.options, args.context)
    debug_log("Final result type:", type(result), "TEST")
    debug_log("Final result structure:", result, "TEST")
    print(json.dumps(result))
