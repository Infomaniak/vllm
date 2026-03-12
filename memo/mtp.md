# Multi-Token Prediction (MTP) in vLLM

## What is MTP?

MTP is a **speculative decoding** method where the model predicts multiple future tokens at once using a small, co-trained prediction head. Unlike traditional speculative decoding (which requires a separate smaller draft model), MTP uses a tiny module **built into the same checkpoint** as the main model.

The core idea: instead of generating 1 token per expensive forward pass through the full model, use a cheap MTP head to **draft** several candidate tokens, then **verify** them all in a single forward pass.

## How MTP Works (General Flow)

```
Target Model (full)                          MTP Head (tiny)
═══════════════════                          ═══════════════

User prompt --> [embed_tokens]
                    |
                    v
              [Layer 0..N]
                    |
                    v
              [final norm]
                    |
                +---+---+
                v       v
           [lm_head]  hidden_states ----------------+
                |                                    |
                v                                    v
          token[N] <-- verified             +--- MTP Head ---+
                                            |                |
                                            | embed(token[N])|
                                            |      |         |
                                            |      v         |
                                            | [pre_fc_norm]  |
                                            | [pre_fc_norm]  |
                                            |      |         |
                                            | cat([emb, hs]) |
                                            |      |         |
                                            |   [fc layer]   |
                                            |      |         |
                                            | [decoder layer]|
                                            |      |         |
                                            |   [norm]       |
                                            |      |         |
                                            |  [lm_head]     |
                                            |      |         |
                                            +------+---------+
                                                   v
                                             token[N+1] <-- draft
                                                   |
                                           (feed back in)
                                                   |
                                            +--- MTP Head ---+
                                            |                |
                                            | embed(token[N+1])
                                            | + new hidden_st|
                                            |     ...        |
                                            +------+---------+
                                                   v
                                             token[N+2] <-- draft
```

### Step-by-step

1. **Target model forward**: The full model processes the input, producing `token[N]` (via `lm_head`) and `hidden_states` (from the last layer).

2. **MTP drafts token[N+1]**: The MTP head takes `hidden_states` + `embed(token[N])`, fuses them, runs one decoder layer, and projects to vocab size to predict `token[N+1]`.

3. **MTP drafts token[N+2]**: The same MTP layer runs again with the previous MTP output hidden states + `embed(token[N+1])` to predict `token[N+2]`.

4. **Verification**: The target model runs one batched forward pass on all draft tokens. Accepted tokens are kept; the first rejected token and all after it are discarded. You always get at least 1 correct token per cycle.

## MTP Head Architecture

The MTP head is very small relative to the full model:

```
2x RMSNorm -> concat -> Linear(hidden_size*2 -> hidden_size) -> 1 transformer layer -> RMSNorm -> Linear(hidden_size -> vocab_size)
```

Components:
- **`embed_tokens`**: Shared vocabulary embedding (same as the target model, no extra memory)
- **`pre_fc_norm_embedding`** + **`pre_fc_norm_hidden`**: RMSNorm layers that normalize inputs before fusion
- **`fc`**: Dense linear projection (`hidden_size * 2 -> hidden_size`, no bias) that fuses the embedding and hidden states
- **`layers`**: One (or more) standard transformer decoder layers (attention + MLP/MoE)
- **`norm`**: Final RMSNorm
- **`lm_head`**: Dense linear projection (`hidden_size -> vocab_size`). Often tied to `embed_tokens` (same weight matrix used in reverse)

The `lm_head` is a simple dense layer with `vocab_size` output (e.g., 151,936 for Qwen3.5). No bias, no activation -- just a matrix multiply.

## MTP in vLLM: Code Structure

### Key files

| File | Purpose |
|------|---------|
| `vllm/config/speculative.py` | Configuration: method detection, weight path, validation |
| `vllm/model_executor/models/qwen3_5_mtp.py` | Model implementation: `Qwen3_5MultiTokenPredictor`, `Qwen3_5MTP`, `Qwen3_5MoeMTP` |

### Configuration flow (`speculative.py`)

1. **Detection**: When model type is `qwen3_5` or `qwen3_5_moe`, the config is remapped to `qwen3_5_mtp` and `n_predict` is set from `mtp_num_hidden_layers`.

2. **Self-drafting**: MTP uses the same checkpoint as the target model. The config sets `self.model = self.target_model_config.model` -- the draft model path points to the same model, but the architecture is overridden to `Qwen3_5MTP` or `Qwen3_5MoeMTP` (which only instantiates the tiny MTP head, not the full model).

3. **Method normalization**: All MTP model types (`deepseek_mtp`, `qwen3_5_mtp`, `mimo_mtp`, etc.) are unified to `method="mtp"`. MTP is treated as an Eagle-style method internally.

4. **`num_speculative_tokens`**: Defaults to `n_predict` (typically 1-2). If set higher, must be divisible by `n_predict` since MTP layers are reused cyclically: `layer_idx = spec_step_idx % num_mtp_layers`.

### Weight loading (`qwen3_5_mtp.py`)

The MTP weights live in the **same checkpoint** as the target model under the `mtp.*` prefix:

| Checkpoint weight name | Remapped to | Module |
|------------------------|-------------|--------|
| `mtp.layers.0.self_attn.q_proj.weight` | `model.layers.0.self_attn.q_proj.weight` | Decoder layer |
| `mtp.fc.weight` | `model.fc.weight` | Fusion projection |
| `mtp.norm.weight` | `model.norm.weight` | Final norm |
| `mtp.pre_fc_norm_hidden.weight` | `model.pre_fc_norm_hidden.weight` | Hidden norm |
| `mtp.pre_fc_norm_embedding.weight` | `model.pre_fc_norm_embedding.weight` | Embedding norm |
| `language_model.embed_tokens.weight` | `embed_tokens.weight` | Shared embedding |

The weight loader filters: only `mtp.*` and `embed_tokens` weights are loaded. All main model layer weights are skipped.

## Qwen3.5-397B-A17B-FP8 with MTP

### Memory overhead

The full model is ~397B parameters. The MTP head adds approximately:
- 1 MoE decoder layer (~a few GB)
- `embed_tokens` is shared (no extra cost)
- `lm_head` is tied to `embed_tokens` (no extra cost)

**Total overhead: ~0.1-0.3% of the full model.**

### Verifying MTP weights exist

```python
from safetensors import safe_open
import glob

files = glob.glob("/path/to/Qwen3.5-397B-A17B-FP8/*.safetensors")
for f in files:
    with safe_open(f, framework="pt") as st:
        for key in st.keys():
            if key.startswith("mtp."):
                print(key)
                break
```

Example output:
```
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.mlp.experts.12.down_proj.weight
mtp.layers.0.input_layernorm.weight
mtp.fc.weight
```

### Serving command

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

### Performance impact

| Without MTP | With MTP (num_speculative_tokens=2) |
|-------------|-------------------------------------|
| 1 full forward -> 1 token | 1 full forward + 2 tiny MTP forwards -> 1-3 tokens |

The MTP forwards are nearly free (~0.1% of model size). With typical acceptance rates of 70-85% (since MTP was co-trained with the target), you get significantly more tokens per expensive target forward pass.

### `num_speculative_tokens` usage summary

| Where | Purpose |
|-------|---------|
| Tree shape | Defines the speculation topology (chain of N tokens) |
| MTP forward loop | Number of sequential draft forward passes |
| Layer cycling | `spec_step_idx % n_predict` selects which MTP layer to use |
| Mamba/GatedDeltaNet state cache | Extra state slots for rollback on rejection |
| Scheduler | Extra KV cache slots reserved per request |