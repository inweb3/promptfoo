import path from 'path';
import { PythonProvider } from './pythonCompletion';
import type { CallApiContextParams, ProviderOptions, EnvOverrides, ProviderResponse } from '../types';

interface OpenAIOptions extends ProviderOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stream?: boolean;
}

export class PythonOpenAiChatProvider extends PythonProvider {
  private modelName: string;
  private env?: EnvOverrides;
  private config: OpenAIOptions;

  constructor(
    modelName: string,
    options: { 
      config?: OpenAIOptions; 
      id?: string; 
      env?: EnvOverrides;
    } = {},
  ) {
    const scriptPath = 'openai_chat.py';
    super(scriptPath, {
      ...options,
      functionName: 'call_api',
      config: {
        ...options.config,
        basePath: path.join(__dirname, '../python'),
      },
    });
    
    this.modelName = modelName;
    this.env = options.env;
    this.config = options.config || {};
  }

  id(): string {
    return this.options?.id || `openai:python:${this.modelName}`;
  }

  protected getApiKey(): string {
    return this.env?.OPENAI_API_KEY || process.env.OPENAI_API_KEY || '';
  }

  protected getOrganization(): string {
    return this.env?.OPENAI_ORGANIZATION || process.env.OPENAI_ORGANIZATION || '';
  }

  protected getApiUrl(): string {
    return this.env?.OPENAI_API_URL || process.env.OPENAI_API_URL || 'https://api.openai.com/v1';
  }

  async callApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse> {
    const options = {
      api_key: this.getApiKey(),
      organization: this.getOrganization(),
      base_url: this.getApiUrl(),
      model: this.modelName,
      ...this.options?.config,
    };

    const response = await super.callApi(prompt, {
      ...context,
      options,
    });

    if (response.token_usage) {
      response.tokenUsage = response.token_usage;
      delete response.token_usage;
    }

    return response;
  }
}