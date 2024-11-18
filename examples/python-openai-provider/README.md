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
promptfoo eval -c promptfooconfig_json.yaml
```

2. View the results:

```bash
npx promptfoo@latest view
```

3. Enable debug logging (optional):

```bash
export PROMPTFOO_ENABLE_DATABASE_LOGS=true
export LOG_LEVEL=debug
promptfoo eval -c promptfooconfig_json.yaml
```

## Create `requirements.txt`

```txt
openai>=1.0.0
```

## Configuration File (`promptfooconfig_json.yaml`)

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
├── promptfooconfig_json.yaml         # Configuration with JSON schema
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

## Running Tests

You can run tests using either the global `promptfoo` command or `npx`. Here's the difference:

### Using Global Command
If you have promptfoo installed globally:
```bash
# Install globally (one-time setup)
npm install -g promptfoo

# Run commands
promptfoo eval -c promptfooconfig.yaml
promptfoo eval -c promptfooconfig.yaml --verbose --no-cache
```

### Using NPX
If you prefer not to install globally, use npx to run the latest version:
```bash
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest eval -c promptfooconfig.yaml --verbose --no-cache
```

Key differences:
- Global command: Faster execution, uses your installed version
- NPX: Always uses latest version, no global installation needed

## Example Commands

1. Run with JSON schema configuration:
```bash
promptfoo eval -c promptfooconfig_json.yaml
# or
npx promptfoo@latest eval -c promptfooconfig_json.yaml
```

2. Run with debug logging:
```bash
export PROMPTFOO_ENABLE_DATABASE_LOGS=true
promptfoo eval -c promptfooconfig_json.yaml --verbose --no-cache
# or
npx promptfoo@latest eval -c promptfooconfig_json.yaml --verbose --no-cache
```

3. Run with text-based assertions:
```bash
promptfoo eval -c promptfooconfig.yaml
# or
npx promptfoo@latest eval -c promptfooconfig.yaml
```

## Configuration Examples

1. JSON Schema Configuration (`promptfooconfig_json.yaml`):
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

2. Text-Based Configuration (`promptfooconfig.yaml`):
```yaml
providers:
  - id: file://openai_chat.py:call_api:gpt-4o-mini
    label: 'Custom Python OpenAI Provider'
    config:
      temperature: 0.5
      max_tokens: 2048
      top_p: 0.9
      frequency_penalty: 0.5
      presence_penalty: 0.5
      stop: ['Human:', 'AI:']
```

## Notes

- The provider supports both structured JSON output and plain text responses
- Debug logging can be enabled via environment variables
- The debug version (`openai_chat_debug.py`) provides detailed logging for development
- Model name is specified in the provider ID rather than config
