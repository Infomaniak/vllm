# Speculative Decoding Config in vLLM

## CLI Usage

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --tensor-parallel-size 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --reasoning-parser qwen3
```

## Key Files

- **Config class:** `vllm/config/speculative.py` — `SpeculativeConfig` dataclass (line 58)
- **CLI parsing:** `vllm/engine/arg_utils.py` — argument defined ~line 1205, processed by `create_speculative_config()` ~line 1372
- **Integration:** `vllm/config/vllm.py` line 251

## All Available Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | str | None | Speculative method (see below) |
| `num_speculative_tokens` | int (>0) | None | Number of draft tokens to generate |
| `model` | str | None | Draft model name/path |
| `enforce_eager` | bool | None | Override eager execution |
| `draft_tensor_parallel_size` | int (≥1) | None | TP size for draft model |
| `quantization` | str | None | Quantization for draft model |
| `max_model_len` | int (≥1) | None | Max length for draft model |
| `revision` | str | None | HuggingFace revision for draft model |
| `code_revision` | str | None | HuggingFace code revision |
| `disable_padded_drafter_batch` | bool | False | Allow variable-length speculative batches |
| `use_local_argmax_reduction` | bool | False | Reduce TP communication overhead |
| `parallel_drafting` | bool | False | Generate all draft tokens in parallel (EAGLE/draft model only) |
| `prompt_lookup_max` | int (≥1) | None | Max ngram window (ngram method only) |
| `prompt_lookup_min` | int (≥1) | None | Min ngram window (ngram method only) |
| `speculative_token_tree` | str | None | Tree structure for token generation |
| `suffix_decoding_max_tree_depth` | int | 24 | Max suffix tree depth |
| `suffix_decoding_max_cached_requests` | int | 10000 | Max cached requests in suffix tree |
| `suffix_decoding_max_spec_factor` | float | 1.0 | Controls speculation length scaling |
| `suffix_decoding_min_token_prob` | float | 0.1 | Min token probability threshold |

## Supported Methods

```
ngram, medusa, mlp_speculator, draft_model, suffix, eagle, eagle3
```

MTP model types:
```
mtp, deepseek_mtp, qwen3_next_mtp, qwen3_5_mtp, mimo_mtp,
glm4_moe_mtp, glm4_moe_lite_mtp, glm_ocr_mtp, ernie_mtp,
exaone_moe_mtp, longcat_flash_mtp, pangu_ultra_moe_mtp, step3p5_mtp
```

## Increasing num_speculative_tokens

Yes, you can increase it. **Constraint for MTP:** value must be divisible by the model's internal `n_predict`.

For [Qwen3.5 MTP](vllm/model_executor/models/qwen3_5.py:768), `n_predict=1`, so any positive integer works:

```bash
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

**Typical sweet spot: 2–5 tokens.**
- Higher values → more throughput potential but lower acceptance rate
- Beyond 5, gains plateau or reverse

### Validation rules (speculative.py)

```python
# Must be > 0
num_speculative_tokens: int = Field(default=None, gt=0)

# For MTP: must be divisible by n_predict
if num_speculative_tokens > n_predict and num_speculative_tokens % n_predict != 0:
    raise ValueError(...)
```
