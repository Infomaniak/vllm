# Force tool call parsing

Model: `Qwen3.5 all models are touch`

## Thinking + force tool

### Input

```json
{
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false,
  "chat_template_kwargs": {
    "enable_thinking": true
  },
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "web_search"
    }
  },
  "messages": [
    {
      "role": "user",
      "content": "quelle sont les derrnier résultata en f1"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Web search tool",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query. This is the main input for the web search"
            }
          },
          "required": [
            "query"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ]
}
```

### output

```json
{
  "id": "chatcmpl-aded088f1a9ae601",
  "object": "chat.completion",
  "created": 1776775239,
  "model": "Qwen/Qwen3.5-0.8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [
          {
            "id": "chatcmpl-tool-8f967564534d1f2d",
            "type": "function",
            "function": {
              "name": "web_search",
              "arguments": "\n\n<tool_call>\n<function=web_search>\n<parameter=query>\nF1 résultats derniers résultats 2025 2024 2023 2022\n</parameter>\n</function>\n</tool_call>"
            }
          }
        ],
        "reasoning": "The user is asking about the \"dernier résultat en F1\" in French. This seems to be asking about the final results in Formula 1. However, F1 results are not typically \"final results\" in the sense of being the absolute final score at the end of the season.\n\nLet me search for information about F1 results to provide helpful information to the user.\n"
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 303,
    "total_tokens": 433,
    "completion_tokens": 130,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

## No thinking + force tool stream

### input

```
{
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": true,
  "chat_template_kwargs": {
    "enable_thinking": false
  },
  "tool_choice": {
    "type": "function",
    "function": {
      "name": "web_search"
    }
  },
  "messages": [
    {
      "role": "user",
      "content": "quelle sont les derrnier résultata en f1"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Web search tool",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query. This is the main input for the web search"
            }
          },
          "required": [
            "query"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ]
}
```

### output

```
data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"{"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"\n  \""},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"query\": \""},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"résultats derniers"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":" Gr"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"ands Prix"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":" Form"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"ule 1"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":" 20"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"24\""},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-13ce8a4b-1f12-4e1f-aa29-050e11482706","object":"chat.completion.chunk","created":1776779614,"model":"Qwen/Qwen3.5-35B-A3B-FP8","choices":[{"index":0,"delta":{"content":"\n}"},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}]}

data: [DONE]


```
