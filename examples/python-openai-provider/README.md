# Python OpenAI Provider Example

This example demonstrates how to use a custom Python-based OpenAI provider with promptfoo.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key
```

## Usage

1. Evaluate a prompt:

```bash
npx promptfoo@latest eval -c pf_m.yaml
```

2. View the results:

```bash
npx promptfoo@latest view
```

3. Enable debug logging (optional):

```bash
export PROMPTFOO_ENABLE_DATABASE_LOGS=true
export LOG_LEVEL=debug
npx promptfoo@latest eval -c pf_m.yaml
```

## Create `requirements.txt`

```txt
openai>=1.0.0
```

## Configuration File (`pf_m.yaml`)

```yaml
providers:
  - id: file://openai_chat.py:call_api:gpt-4o-mini
    config:
      temperature: 0.5
      response_format:
        json_schema:
          name: standup_notes_object
          schema:
            # ... schema details ...
```

## Project Structure

Your project directory should look like this:

```bash
examples/python-openai-provider/
├── README.md
├── requirements.txt
├── pf_m.yaml         # Configuration with JSON schema
├── openai_chat.py    # Production provider
└── openai_chat_debug.py  # Debug version with detailed logging
```

## Provider ID Format

The provider ID follows this format:
```
file://<script_path>:call_api:<model_name>
```

For example:
```yaml
id: file://openai_chat.py:call_api:gpt-4o-mini
```

## Example Commands

1. Run with configuration:
```bash
npx promptfoo@latest eval -c pf_m.yaml
```

2. Run with debug logging:
```bash
export PROMPTFOO_ENABLE_DATABASE_LOGS=true
npx promptfoo@latest eval -c pf_m.yaml --verbose --no-cache
```

## Notes

- The provider supports both simple text responses and structured JSON output
- Debug logging can be enabled via environment variables
- The debug version (`openai_chat_debug.py`) provides detailed logging for development
- Model name is specified in the provider ID rather than config
