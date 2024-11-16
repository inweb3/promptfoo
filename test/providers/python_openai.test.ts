import fs from 'fs';
import path from 'path';
import { getCache, isCacheEnabled } from '../../src/cache';
import { PythonOpenAiChatProvider } from '../../src/providers/python_openai';
import { runPython } from '../../src/python/pythonUtils';
import { parsePathOrGlob } from '../../src/util';

// Mock the dependencies
jest.mock('../../src/python/pythonUtils');
jest.mock('fs');
jest.mock('path');
jest.mock('../../src/cache');
jest.mock('../../src/util', () => {
  const actual = jest.requireActual('../../src/util');
  return {
    ...actual,
    parsePathOrGlob: jest.fn(() => ({
      extension: 'py',
      functionName: 'call_api',
      isPathPattern: false,
      filePath: '/absolute/path/to/openai_chat.py',
    })),
  };
});

describe('PythonOpenAiChatProvider', () => {
  const mockRunPython = jest.mocked(runPython);
  const mockGetCache = jest.mocked(getCache);
  const mockIsCacheEnabled = jest.mocked(isCacheEnabled);
  const mockReadFileSync = jest.mocked(fs.readFileSync);
  const mockResolve = jest.mocked(path.resolve);
  const mockJoin = jest.mocked(path.join);
  const mockParsePathOrGlob = jest.mocked(parsePathOrGlob);

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetCache.mockResolvedValue({
      get: jest.fn(),
      set: jest.fn(),
    } as never);
    mockIsCacheEnabled.mockReturnValue(false);
    mockReadFileSync.mockReturnValue('mock file content');
    mockResolve.mockReturnValue('/absolute/path/to/openai_chat.py');
    mockJoin.mockImplementation((...args) => args.join('/'));
    mockParsePathOrGlob.mockReturnValue({
      extension: 'py',
      functionName: 'call_api',
      isPathPattern: false,
      filePath: '/absolute/path/to/openai_chat.py',
    });
  });

  it('should handle simple prompts', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini', {
      env: {
        OPENAI_API_KEY: 'test-key',
      },
      config: {
        basePath: '/test/path',
      },
      scriptPath: 'file://openai_chat.py:call_api',
    });

    mockRunPython.mockResolvedValue({
      output: 'Hello!',
      token_usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
      },
    });

    const response = await provider.callApi('Say hello!');

    expect(response.error).toBeUndefined();
    expect(response.output).toBe('Hello!');
    expect(response.tokenUsage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
  });

  it('should handle message arrays', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini', {
      env: {
        OPENAI_API_KEY: 'test-key',
      },
      config: {
        basePath: '/test/path',
      },
      scriptPath: 'file://openai_chat.py:call_api',
    });

    mockRunPython.mockResolvedValue({
      output: '4',
      token_usage: {
        prompt_tokens: 43,
        completion_tokens: 1,
        total_tokens: 44,
      },
    });

    const messages = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: "What's 2+2?" },
    ];

    const response = await provider.callApi(JSON.stringify(messages));

    expect(response.error).toBeUndefined();
    expect(response.output).toBe('4');
    expect(response.tokenUsage).toEqual({
      prompt_tokens: 43,
      completion_tokens: 1,
      total_tokens: 44,
    });
  });

  it('should handle API errors', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini', {
      env: {
        OPENAI_API_KEY: 'invalid-key',
      },
      config: {
        basePath: '/test/path',
      },
      scriptPath: 'file://openai_chat.py:call_api',
    });

    mockRunPython.mockResolvedValue({
      error: 'Invalid API key',
    });

    const response = await provider.callApi('This should fail');

    expect(response.error).toBe('Invalid API key');
    expect(response.output).toBeUndefined();
  });

  // Test configuration options
  it('should properly pass OpenAI configuration options', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini', {
      env: {
        OPENAI_API_KEY: 'test-key',
      },
      config: {
        temperature: 0.5,
        max_tokens: 100,
        top_p: 0.8,
        frequency_penalty: 0.2,
        presence_penalty: 0.1,
      },
    });

    mockRunPython.mockResolvedValue({
      output: 'Test response',
      token_usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    });

    await provider.callApi('Test prompt');

    // Add debugging logs
    const mockCalls = mockRunPython.mock.calls;
    console.log('Mock calls:', JSON.stringify(mockCalls, null, 2));
    console.log('First call args:', JSON.stringify(mockCalls[0], null, 2));

    expect(mockRunPython).toHaveBeenCalledWith(
      expect.any(String),
      'call_api',
      [
        'Test prompt',
        expect.any(Object),
        {
          options: expect.objectContaining({
            temperature: 0.5,
            max_tokens: 100,
            top_p: 0.8,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            model: 'gpt-4o-mini',
          }),
        },
      ],
      expect.any(Object)
    );
  });

  // Test different error types
  it('should handle rate limit errors', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini');
    mockRunPython.mockResolvedValue({
      error: 'Rate limit exceeded: Please try again later',
      error_type: 'rate_limit',
    });

    const response = await provider.callApi('Test prompt');

    expect(response.error).toContain('Rate limit exceeded');
    expect(response.error_type).toBe('rate_limit');
  });

  it('should handle API errors', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini');
    mockRunPython.mockResolvedValue({
      error: 'OpenAI API error: Invalid request',
      error_type: 'api_error',
    });

    const response = await provider.callApi('Test prompt');

    expect(response.error).toContain('OpenAI API error');
    expect(response.error_type).toBe('api_error');
  });

  // Test environment variable handling
  it('should prioritize env options over process.env', async () => {
    process.env.OPENAI_API_KEY = 'process-key';
    process.env.OPENAI_ORGANIZATION = 'process-org';
    process.env.OPENAI_API_URL = 'process-url';

    const provider = new PythonOpenAiChatProvider('gpt-4o-mini', {
      env: {
        OPENAI_API_KEY: 'env-key',
        OPENAI_ORGANIZATION: 'env-org',
        OPENAI_API_URL: 'env-url',
      },
    });

    mockRunPython.mockResolvedValue({ output: 'Test response' });

    await provider.callApi('Test prompt');

    // Add debugging logs
    const mockCalls = mockRunPython.mock.calls;
    console.log('Mock calls for env test:', JSON.stringify(mockCalls, null, 2));
    console.log('First call args for env test:', JSON.stringify(mockCalls[0], null, 2));

    expect(mockRunPython).toHaveBeenCalledWith(
      expect.any(String),
      'call_api',
      [
        'Test prompt',
        expect.any(Object),
        {
          options: expect.objectContaining({
            api_key: 'env-key',
            organization: 'env-org',
            base_url: 'env-url',
            model: 'gpt-4o-mini',
          }),
        },
      ],
      expect.any(Object)
    );
  });

  // Test message validation
  it('should handle invalid message format', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini');
    mockRunPython.mockResolvedValue({
      error: 'Each message must have \'role\' and \'content\' fields',
      error_type: 'unknown',
    });

    const invalidMessages = JSON.stringify([
      { role: 'system' }, // missing content
      { content: 'Hello' }, // missing role
    ]);

    const response = await provider.callApi(invalidMessages);

    expect(response.error).toContain('must have \'role\' and \'content\' fields');
    expect(response.error_type).toBe('unknown');
  });

  // Test token usage handling
  it('should properly transform token_usage to tokenUsage', async () => {
    const provider = new PythonOpenAiChatProvider('gpt-4o-mini');
    const tokenUsage = {
      prompt_tokens: 10,
      completion_tokens: 20,
      total_tokens: 30,
    };

    mockRunPython.mockResolvedValue({
      output: 'Test response',
      token_usage: tokenUsage,
    });

    const response = await provider.callApi('Test prompt');

    expect(response.token_usage).toBeUndefined();
    expect(response.tokenUsage).toEqual(tokenUsage);
  });
});
