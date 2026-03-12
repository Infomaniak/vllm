# Streaming Tool Call IndexError — Root Cause Analysis

## The Error

```
IndexError: list index out of range
  File "serving.py", line 1262, in chat_completion_stream_generator
    actual_call = tool_parser.streamed_args_for_tool[index]
```

This error occurs in the **stream finalization** block of `chat_completion_stream_generator`
(`serving.py:1211-1281`) when vLLM tries to check for unstreamed tool argument tokens.

---

## Two Key Data Structures

The `Qwen3CoderToolParser` maintains two parallel lists:

| List | Purpose | Populated by |
|------|---------|-------------|
| `prev_tool_call_arr` | Stores `{"name": ..., "arguments": ...}` for each tool call | Streaming (header send) + non-streaming |
| `streamed_args_for_tool` | Tracks what argument text was already streamed to the client | Streaming only |

**The invariant**: `len(streamed_args_for_tool)` must always equal `len(prev_tool_call_arr)`
during streaming, because `serving.py` uses the same `index` to access both.

---

## The Code Path (serving.py:1211-1281)

This block runs when `output.finish_reason is not None` (the model is done generating).

### Step 1 — Compute the index (lines 1221-1228)

```python
if tool_parser:
    auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
    index = len(tool_parser.prev_tool_call_arr) - 1 if auto_tools_called else 0
```

If the parser detected tool calls during streaming, `index` = last tool call position.

### Step 2 — Gate check (lines 1232-1236)

```python
if self._should_check_for_unstreamed_tool_arg_tokens(delta_message, output) and tool_parser:
```

This returns `True` only when **ALL** of these are true simultaneously:

```python
output.finish_reason is not None        # (1) final output chunk
and self.enable_auto_tools              # (2) --enable-auto-tool-choice flag
and self.tool_parser                    # (3) --tool-call-parser is set
and delta_message                       # (4) parser returned something
and delta_message.tool_calls            # (5) that something has tool_calls
and delta_message.tool_calls[0].function              # (6)
and delta_message.tool_calls[0].function.arguments is not None  # (7)
```

### Step 3 — The crash (line 1272)

```python
actual_call = tool_parser.streamed_args_for_tool[index]
```

If `streamed_args_for_tool` has fewer entries than `prev_tool_call_arr`, this raises
`IndexError: list index out of range`.

---

## Why It's Hard to Reproduce: The Timing Problem

Under **normal token-by-token generation**, the tool call tokens arrive in separate chunks:

```
Chunk 1:  <tool_call>           → parser returns None
Chunk 2:  <function=get_weather> → parser returns DeltaToolCall(name=...)
Chunk 3:  <parameter=location>  → parser returns DeltaToolCall(arguments="{")
Chunk 4:  Paris                 → parser returns DeltaToolCall(arguments='"location": "Paris"')
Chunk 5:  </parameter>          → consumed internally
Chunk 6:  </function>           → parser returns DeltaToolCall(arguments="}")   ← has tool_calls
Chunk 7:  </tool_call>          → parser returns None                          ← no tool_calls
Chunk 8:  <EOS>                 → finish_reason="stop", parser returns None    ← no tool_calls
```

At chunk 8 (the final one), `delta_message` has **no tool_calls**.
So condition (5) of the gate check is `False`, and the block is **skipped entirely**.
The crash line is never reached. Everything works.

---

## How to Trigger the Bug

The gate check passes when `finish_reason` and `tool_calls` are present on the **same chunk**.
This happens when **multiple tokens are delivered in a single output batch**.

### Scenario A: Speculative Decoding

With speculative decoding, a draft model proposes N tokens at once. If the final batch
contains both the closing arguments and the EOS token:

```
Single batched output:  "</function></tool_call><EOS>"

  → parser processes </function>  → returns DeltaToolCall(arguments="}")
  → finish_reason is set (EOS was in this batch)
  → Gate check: finish_reason ✓ AND tool_calls ✓ → ENTERS THE BLOCK
  → actual_call = tool_parser.streamed_args_for_tool[index]  → IndexError!
```

Server config that enables this:
```bash
vllm serve model_name \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-model [draft-model] \
  --num-speculative-tokens 5
```

### Scenario B: High Load / Token Batching

Under heavy concurrent load, the engine may batch multiple output tokens into a single
`RequestOutput`. The scheduler groups tokens that were generated between two consecutive
reads of the output queue. High GPU utilization + many concurrent requests increases
the chance that the closing tags and EOS land in the same batch.

This is why the original bug appeared in production under load but not in simple testing.

### Scenario C: Very Short Tool Call Arguments

When arguments are very short (e.g., `{"q": "x"}`), fewer tokens are generated.
This increases the probability that the closing tags and EOS are batched together,
especially with any form of batching or speculative decoding.

---

## The Bug (Old Code)

In the old `Qwen3CoderToolParser`, when the streaming header was sent, only
`prev_tool_call_arr` was populated:

```python
# OLD CODE (buggy)
self.prev_tool_call_arr.append({
    "name": self.current_function_name,
    "arguments": "{}",
})
# streamed_args_for_tool was NEVER appended to!
```

So after streaming a tool call:
- `prev_tool_call_arr` = `[{"name": "get_weather", "arguments": "..."}]` (length 1)
- `streamed_args_for_tool` = `[]` (length 0)
- `index` = `len(prev_tool_call_arr) - 1` = `0`
- `streamed_args_for_tool[0]` → **IndexError**

---

## The Fix (Current Code)

In `qwen3coder_tool_parser.py:485-497`, both lists are now populated together:

```python
# FIXED CODE
self.prev_tool_call_arr.append({
    "name": self.current_function_name,
    "arguments": "{}",
})

# Initialize streamed args tracking for this tool.
# Without this, IndexError occurs when the serving layer
# accesses streamed_args_for_tool[index].
self.streamed_args_for_tool.append("")
```

Additionally, defensive bounds checks were added at lines 627-634 and 680-687:

```python
if self.current_tool_index < len(self.streamed_args_for_tool):
    self.streamed_args_for_tool[self.current_tool_index] += combined
else:
    logger.warning("streamed_args_for_tool out of sync: index=%d len=%d", ...)
```

---

## How to Reproduce (Step by Step)

### Prerequisites
- vLLM version **before** the fix (without the `streamed_args_for_tool.append("")` line)
- Server started with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- One of: speculative decoding enabled, OR high concurrent load

### Method 1: Speculative Decoding (Most Reliable)

1. Start the server with a draft model for speculative decoding
2. Send a single streaming request with tools that forces a tool call:
   ```json
   {
     "stream": true,
     "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
     "tool_choice": "auto",
     "tools": [{"type": "function", "function": {
       "name": "get_weather",
       "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
     }}]
   }
   ```
3. The speculative decoding will batch the final tokens together
4. The gate check passes, and the IndexError is raised

### Method 2: Load Testing (Production-Like)

1. Start the server normally (no speculative decoding)
2. Run `crash_streaming.py` which sends many concurrent requests:
   - 17 conversations per round (12 single-turn + 5 multi-turn with tool history)
   - 10 concurrent threads
   - 5 rounds = 85 total requests
3. Under load, the scheduler batches tokens, increasing the chance
   that closing tags + EOS land in one output
4. Monitor server logs for `list index out of range`

### Method 3: Artificial Delay (Debug Only)

Patch `serving.py` to artificially batch tokens by adding a brief sleep
in the output loop. This simulates what happens under load when the consumer
reads the output queue less frequently:

```python
# TEMPORARY DEBUG PATCH in chat_completion_stream_generator
async for res in result_generator:
    import asyncio
    await asyncio.sleep(0.05)  # force token batching
    ...
```

This makes multiple tokens accumulate between reads, reproducing the batching effect.

---

## Remaining Risk

Even with the fix in the parser, `serving.py:1272` has **no bounds check**:

```python
actual_call = tool_parser.streamed_args_for_tool[index]  # no guard!
```

If any future edge case causes the two lists to desync, the crash returns.
A defensive fix in `serving.py` would be:

```python
if index < len(tool_parser.streamed_args_for_tool):
    actual_call = tool_parser.streamed_args_for_tool[index]
else:
    actual_call = ""
    logger.warning("streamed_args_for_tool missing index %d (len=%d)",
                    index, len(tool_parser.streamed_args_for_tool))
```