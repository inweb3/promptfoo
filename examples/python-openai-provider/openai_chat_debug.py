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
        # 1. Initial Debug Logging
        debug_log("Starting API call with prompt:", prompt, "START")
        debug_log("Raw options received:", options, "START")
        debug_log("Raw context received:", context, "START")

        # 2. Parse Configuration
        options = options or {}
        config = options.get("config", {})

        # 3. Parse model from provider ID
        provider_id = options.get("id", "")
        model = provider_id.split(":")[-1] if ":" in provider_id else "gpt-4o-mini"
        debug_log("Model from provider ID:", model, "CONFIG")

        # 4. Log All Configuration Options
        debug_log("=== CONFIGURATION DETAILS ===", None, "CONFIG")

        # 4.1 Client Configuration
        client_config = {
            "api_key": options.get("api_key"),
            "organization": options.get("organization"),
            "base_url": options.get("base_url"),
        }
        debug_log("OpenAI Client Configuration:", client_config, "CONFIG")

        # 4.2 Model Configuration
        model_config = {
            "model": model,  # Use model from provider ID
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens"),
            "top_p": config.get("top_p", 1),
            "presence_penalty": config.get("presence_penalty", 0),
            "frequency_penalty": config.get("frequency_penalty", 0),
        }
        debug_log("Model Configuration:", model_config, "CONFIG")

        # 3.3 Response Format Configuration
        response_format = config.get("response_format")
        debug_log("Response Format Configuration:", response_format, "CONFIG")

        # 3.4 Provider-Specific Configuration
        provider_config = {
            "id": options.get("id"),
            "base_path": config.get("basePath"),
            "prompt_prefix": config.get("prompt", {}).get("prefix"),
            "prompt_suffix": config.get("prompt", {}).get("suffix"),
        }
        debug_log("Provider Configuration:", provider_config, "CONFIG")

        # 5. Initialize Client
        client = OpenAI(**client_config)

        # 6. Parse Messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = json.loads(prompt) if isinstance(prompt, str) else prompt
        debug_log("Parsed Messages:", messages, "INIT")

        # 7. Build API Parameters
        api_params = {
            **model_config,
            "messages": messages,
        }

        # Add response format if provided
        if response_format:
            api_params["response_format"] = response_format
            debug_log("Added Response Format:", api_params["response_format"], "CONFIG")

        debug_log("Final API Parameters:", api_params, "API")

        # 8. Make API Call
        response = client.chat.completions.create(**api_params)
        debug_log("Raw API Response:", response.model_dump(), "RESPONSE")

        # 9. Process Response
        content = response.choices[0].message.content
        debug_log("Response Content:", content, "CONTENT")

        # 10. Parse JSON Response
        parsed_content = json.loads(content)
        debug_log("Parsed Content:", parsed_content, "RESULT")

        return {"output": parsed_content}

    except Exception as e:
        debug_log(f"Error: {str(e)}", None, "ERROR")
        debug_log("Error Traceback:", traceback.format_exc(), "ERROR")
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
