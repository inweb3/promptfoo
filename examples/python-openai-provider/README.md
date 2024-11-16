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
npx promptfoo@latest eval
```

2. View the results:

```bash
npx promptfoo@latest view
```

## Create `requirements.txt`

```txt
openai>=1.0.0
```

## Create `promptfooconfig.yaml`

```yaml
prompts:
"What is 2+2?"
"Write a haiku about programming"
"Explain quantum computing"
providers:
id: file://openai_chat.py
config:
model: gpt-4o-mini
temperature: 0.7
max_tokens: 150
tests:
vars: {}
assert:
type: contains
value: "4" # For the first prompt
```

## Copy the OpenAI provider to the current directory

```bash
cp src/python/openai_chat.py .
```

Your project directory should now look like this:

```bash
examples/python-openai-provider/
├── README.md
├── requirements.txt
├── promptfooconfig.yaml
└── openai_chat.py
```
