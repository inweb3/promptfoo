import os
import json
import pytest
from unittest.mock import patch, MagicMock
from openai import OpenAI, OpenAIError, APIError, RateLimitError
from openai_chat import call_api, validate_messages, LEGACY_JSON_MODELS


def test_validate_messages():
    # Valid messages
    valid_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    validate_messages(valid_messages)  # Should not raise

    # Test invalid cases
    with pytest.raises(ValueError, match="Messages must be a list"):
        validate_messages("not a list")

    with pytest.raises(ValueError, match="Each message must be a dictionary"):
        validate_messages([["not", "a", "dict"]])

    with pytest.raises(
        ValueError, match="Each message must have 'role' and 'content' fields"
    ):
        validate_messages([{"role": "user"}])  # Missing content

    with pytest.raises(
        ValueError, match="Message role must be 'system', 'user', or 'assistant'"
    ):
        validate_messages([{"role": "invalid", "content": "test"}])


@patch("openai_chat.OpenAI")
def test_simple_prompt_with_config(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    # Test with simple prompt and config
    result = call_api(
        "Say hello!",
        {
            "api_key": "test-key",
            "config": {"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 100},
        },
    )

    # Verify the result
    assert result["output"] == "Hello!"
    assert result["token_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert result.get("error") is None

    # Verify the API was called with correct config
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
        temperature=0.5,
        max_tokens=100,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop=None,
        stream=False,
    )


@patch("openai_chat.OpenAI")
def test_configuration_options(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    # Test with all configuration options
    options = {
        "api_key": "test-key",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.8,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "stop": ["\n", "Human:", "AI:"],
            "stream": False,
        },
    }

    result = call_api("Test prompt", options)

    # Verify the API was called with all config options
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.5,
        max_tokens=100,
        top_p=0.8,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stop=["\n", "Human:", "AI:"],
        stream=False,
    )


@patch("openai_chat.OpenAI")
def test_default_configuration(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    # Test with no configuration (should use defaults)
    result = call_api("Test prompt", {"api_key": "test-key"})

    # Verify defaults were used
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.7,
        max_tokens=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop=None,
        stream=False,
    )


@patch("openai_chat.OpenAI")
def test_message_array(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="4"))]
    mock_response.usage.prompt_tokens = 43
    mock_response.usage.completion_tokens = 1
    mock_response.usage.total_tokens = 44
    mock_client.chat.completions.create.return_value = mock_response

    # Test with message array
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's 2+2?"},
    ]
    result = call_api(
        json.dumps(messages), {"api_key": "test-key", "model": "gpt-4o-mini"}
    )

    assert result["output"] == "4"
    assert result["token_usage"] == {
        "prompt_tokens": 43,
        "completion_tokens": 1,
        "total_tokens": 44,
    }


@patch("openai_chat.OpenAI")
def test_error_handling(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Add debugging information about the error classes
    print("\nDebugging error classes:")
    print(f"RateLimitError: {dir(RateLimitError)}")
    print(f"APIError: {dir(APIError)}")
    print(f"OpenAIError: {dir(OpenAIError)}")

    # Test rate limit error
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

    try:
        error = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )
        print("\nSuccessfully created RateLimitError")
    except Exception as e:
        print(f"\nFailed to create RateLimitError: {str(e)}")
        raise

    mock_client.chat.completions.create.side_effect = error
    result = call_api("Test prompt", {"api_key": "test-key"})
    assert result["error"].startswith("Rate limit exceeded")
    assert result["error_type"] == "rate_limit"

    # Test API error
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": {"message": "Invalid request"}}
    mock_request = MagicMock()

    try:
        error = APIError(
            message="Invalid request",
            request=mock_request,
            body={"error": {"message": "Invalid request"}},
        )
        print("\nSuccessfully created APIError")
    except Exception as e:
        print(f"\nFailed to create APIError: {str(e)}")
        raise

    mock_client.chat.completions.create.side_effect = error
    result = call_api("Test prompt", {"api_key": "test-key"})
    assert result["error"].startswith("OpenAI API error")
    assert result["error_type"] == "api_error"

    # Test generic OpenAI error
    mock_client.chat.completions.create.side_effect = OpenAIError("Unknown error")
    result = call_api("Test prompt", {"api_key": "test-key"})
    assert result["error"].startswith("OpenAI error")
    assert result["error_type"] == "openai_error"


@patch("openai_chat.OpenAI")
def test_json_response_format(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"project": "Search", "status": "active"}'),
            finish_reason="stop"
        )
    ]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    schema = {
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "status": {"type": "string"}
        }
    }

    # Test default behavior (should use structured output)
    result = call_api(
        "Extract project info",
        {
            "api_key": "test-key",
            "config": {
                "model": "gpt-4o-mini",
                "response_format": {
                    "type": "json_schema",
                    "schema": schema
                }
            },
        },
    )

    # Verify API was called with correct parameters for structured output
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["response_format"] == {
        "type": "json_schema",
        "json_schema": schema
    }
    # No JSON system message needed for structured output
    assert not any("JSON" in msg.get("content", "") 
                  for msg in call_args["messages"])

    # Test with legacy model (should use json_object)
    result = call_api(
        "Extract project info",
        {
            "api_key": "test-key",
            "config": {
                "model": "gpt-3.5-turbo",  # Using specific legacy model
                "response_format": {
                    "type": "json_object",
                    "schema": schema
                }
            },
        },
    )

    # Verify API was called with correct parameters for legacy model
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["response_format"] == {"type": "json_object"}
    # Should have JSON instruction in system message for legacy models
    assert any("JSON" in msg.get("content", "") 
              for msg in call_args["messages"])


@patch("openai_chat.OpenAI")
def test_json_response_truncation(mock_openai):
    # Setup mock response with length truncation
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"partial": "data"'),
            finish_reason="length"  # Indicates truncation
        )
    ]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    # Test with JSON response format
    result = call_api(
        "Test prompt",
        {
            "api_key": "test-key",
            "config": {
                "model": "gpt-4o-mini",
                "response_format": {"type": "json_schema"},
                "max_tokens": 5  # Small limit to force truncation
            },
        },
    )

    # Verify warning about truncation
    assert "error" not in result
    assert isinstance(result["output"], str)
    # The actual handling of truncated JSON would depend on your implementation


@patch("openai_chat.OpenAI")
def test_system_message_injection(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"test": "data"}'),
            finish_reason="stop"
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # # Test with older model (should inject system message)
    # result = call_api(
    #     "Regular prompt without JSON mention",
    #     {
    #         "api_key": "test-key",
    #         "config": {
    #             "model": "gpt-3.5-turbo",
    #             "response_format": {"type": "json_object"}
    #         },
    #     },
    # )

    # # Verify system message was injected for older model
    # call_args = mock_client.chat.completions.create.call_args[1]
    # assert any(
    #     msg["role"] == "system" and "JSON" in msg["content"]
    #     for msg in call_args["messages"]
    # )

    # Test with newer model (should not inject system message)
    result = call_api(
        "Regular prompt without JSON mention",
        {
            "api_key": "test-key",
            "config": {
                "model": "gpt-4o-mini",
                "response_format": {
                    "type": "json_schema",
                    "schema": {"type": "object"}
                }
            },
        },
    )

    # Verify no system message was injected for newer model
    call_args = mock_client.chat.completions.create.call_args[1]
    assert not any(
        msg["role"] == "system" and "JSON" in msg["content"]
        for msg in call_args["messages"]
    )


@patch("openai_chat.OpenAI")
def test_complex_schema_handling(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='''
                {
                    "project_name": "Search Functionality",
                    "status": "In Progress",
                    "team_members": ["Alice", "Bob"],
                    "metrics": {
                        "completion": 75,
                        "issues": 2
                    }
                }
            '''.strip()),
            finish_reason="stop"
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Test with complex nested schema
    complex_schema = {
        "type": "object",
        "properties": {
            "project_name": {"type": "string"},
            "status": {"type": "string"},
            "team_members": {
                "type": "array",
                "items": {"type": "string"}
            },
            "metrics": {
                "type": "object",
                "properties": {
                    "completion": {"type": "number"},
                    "issues": {"type": "number"}
                }
            }
        }
    }

    result = call_api(
        "Get project details",
        {
            "api_key": "test-key",
            "config": {
                "model": "gpt-4o-mini",
                "response_format": {
                    "type": "json_schema",
                    "schema": complex_schema
                }
            },
        },
    )

    # Verify the result
    assert "error" not in result
    output_json = json.loads(result["output"])
    assert isinstance(output_json["team_members"], list)
    assert isinstance(output_json["metrics"], dict)
    assert isinstance(output_json["metrics"]["completion"], (int, float))


@patch("openai_chat.OpenAI")
def test_legacy_model_detection(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"test": "data"}'),
            finish_reason="stop"
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Test each legacy model
    for model in LEGACY_JSON_MODELS:
        result = call_api(
            "Test prompt",
            {
                "api_key": "test-key",
                "config": {
                    "model": model,
                    "response_format": {
                        "type": "json_object",
                        "schema": {"type": "object"}
                    }
                },
            },
        )
        
        # Verify legacy JSON mode was used
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["response_format"] == {"type": "json_object"}
        assert any("JSON" in msg.get("content", "") 
                  for msg in call_args["messages"])


if __name__ == "__main__":
    pytest.main([__file__])
