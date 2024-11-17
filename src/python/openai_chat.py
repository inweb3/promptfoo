import argparse
import json
import sys
from openai import OpenAI, OpenAIError, RateLimitError, APIError

# List of models that require legacy JSON mode instead of structured output
LEGACY_JSON_MODELS = [
    'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'
]

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


def call_api(prompt, options=None, context=None):
    options = options or {}
    config = options.get('config', {})
    
    # Debug logging
    print(f"Debug - Input prompt type: {type(prompt)}", file=sys.stderr)
    print(f"Debug - Input prompt: {prompt}", file=sys.stderr)
    print(f"Debug - Options: {json.dumps(options, default=str)}", file=sys.stderr)
    
    # Get model and check if it needs legacy JSON mode
    model = config.get('model', 'gpt-4o-mini')
    needs_legacy_json = any(model.startswith(legacy_model) for legacy_model in LEGACY_JSON_MODELS)
    
    # Get all configuration parameters with defaults
    temperature = config.get('temperature', 0.7)
    max_tokens = config.get('max_tokens')
    top_p = config.get('top_p')
    frequency_penalty = config.get('frequency_penalty')
    presence_penalty = config.get('presence_penalty')
    stop = config.get('stop')
    response_format = config.get('response_format')
    
    # Log configuration
    print(f"""Custom Python OpenAI provider called with:
- model: {model}
- temperature: {temperature}
- max_tokens: {max_tokens}
- top_p: {top_p}
- frequency_penalty: {frequency_penalty}
- presence_penalty: {presence_penalty}
- stop: {stop}
- response_format: {json.dumps(response_format) if response_format else None}""", file=sys.stderr)

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
            messages = json.loads(prompt) if isinstance(prompt, str) else prompt
            validate_messages(messages)

        # Debug logging
        print(f"Debug - Processed messages: {json.dumps(messages)}", file=sys.stderr)

        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "stream": config.get("stream", False),
        }
        
        # Handle response format for structured/JSON output
        if response_format:
            if needs_legacy_json:
                # Fall back to JSON mode for older models
                api_params["response_format"] = {"type": "json_object"}
                
                # Add system message for JSON formatting if not present
                if not any("JSON" in msg.get("content", "") for msg in messages):
                    schema_msg = {
                        "role": "system",
                        "content": "You must provide output in JSON format" + 
                            (f" following this schema: {json.dumps(response_format.get('schema', {}))}" 
                             if 'schema' in response_format else "")
                    }
                    messages.insert(0, schema_msg)
                    api_params["messages"] = messages
                print(f"Debug - Using legacy JSON mode with system message", file=sys.stderr)
            else:
                # Use native structured output (default)
                api_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": response_format.get('schema', {})  # Pass the schema directly
                }
                print(f"Debug - Using structured output with schema: {json.dumps(response_format.get('schema', {}))}", file=sys.stderr)

        print(f"Debug - Final API params: {json.dumps(api_params, indent=2)}", file=sys.stderr)

        # Make the API call
        response = client.chat.completions.create(**api_params)

        # Check for completion
        if response.choices[0].finish_reason == 'length':
            print(f"Warning: Response was truncated due to length", file=sys.stderr)

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
        print(f"Debug - RateLimitError: {str(e)}", file=sys.stderr)
        return {"error": f"Rate limit exceeded: {str(e)}", "error_type": "rate_limit"}
    except APIError as e:
        print(f"Debug - APIError: {str(e)}", file=sys.stderr)
        return {"error": f"OpenAI API error: {str(e)}", "error_type": "api_error"}
    except OpenAIError as e:
        print(f"Debug - OpenAIError: {str(e)}", file=sys.stderr)
        return {"error": f"OpenAI error: {str(e)}", "error_type": "openai_error"}
    except Exception as e:
        print(f"Debug - Unexpected error: {str(e)}", file=sys.stderr)
        return {"error": f"Unexpected error: {str(e)}", "error_type": "unknown"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--options", type=json.loads, default={})
    parser.add_argument("--context", type=json.loads, default={})

    args = parser.parse_args()
    result = call_api(args.prompt, args.options, args.context)
    print(json.dumps(result))
