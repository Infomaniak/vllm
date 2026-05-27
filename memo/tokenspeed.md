# Implementation Details of TokenSpeed MLA Backend

## Overview
The `TokenspeedMLABackend` is an attention backend for Multi-Head Latent Attention (MLA) specifically optimized for the **DeepSeek R1** architecture running on **Blackwell** GPUs. It acts as a bridge to the external `tokenspeed_mla` package, which provides a highly optimized CuTe DSL-based decode kernel (`BlackwellMultiHeadLatentAttentionForwardFP8` and `FP16`).

## Key Requirements & Limitations
- **Hardware Compatibility**: Exclusively supports NVIDIA Blackwell architectures (Compute Capability 10.0).
- **Data Types**: The KV cache *must* be quantized to FP8 (`fp8` or `fp8_e4m3`). While the model's supported compute dtypes can be FP16 or BF16, the queries fed into the core kernel are expected to be pre-quantized to FP8 upstream (`torch.float8_e4m3fn`).
- **Fixed Dimensions**: The backend is strictly shape-specialized for DeepSeek R1. It explicitly rejects initialization unless the dimensions match: `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, and `v_head_dim=128`.
- **Supported Phases**: Only supports the **decode** phase (`AttentionType.DECODER`).
- **Unsupported Features**: Does not support alibi slopes, sliding window attention, or logits soft-capping.

## Implementation Deep Dive with Code Examples

### 1. Zero-Copy Tensor Slicing
The latent and RoPE components of both Query and KV-cache must be separated before being fed to the underlying CuTe kernel. Since these components are concatenated along the last dimension, slicing them creates non-contiguous tensors. 
The backend avoids expensive data copying by slicing with standard PyTorch operations and then defining dynamic `cute.runtime.make_fake_tensor` abstractions with runtime-inferred strides.

```python
# 1. Slice using PyTorch (these resulting tensors are non-contiguous)
q_latent_k = query[..., :kv_lora_rank]
q_rope_k = query[..., kv_lora_rank:]

c_latent_k = kv_cache[:, :, :kv_lora_rank]
c_rope_k = kv_cache[:, :, kv_lora_rank:]

# 2. Tell CuTe DSL to expect fully dynamic strides for non-contiguous tensors
q_latent_fake = cute.runtime.make_fake_tensor(
    cutlass_dtype,
    (sym_batch, sym_seq_q, sym_heads, sym_latent),
    stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
    assumed_align=16,
)
```

### 2. JIT Compilation and Caching
The core kernel is JIT-compiled through `cute.compile`. Because compiling CuTe DSL bindings can be slow, the function `_get_compiled_mla_kernel` is heavily cached using Python's `@functools.cache`. This means the compilation overhead is only paid on the very first forward pass for a given configuration.

```python
@functools.cache
def _get_compiled_mla_kernel(
    torch_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    # ... other args ...
) -> Callable:
    # Kernel configuration...
    kernel_obj = KernelClass(**kernel_kwargs)
    
    # ... Fake tensor definitions ...
    
    compiled_kernel = cute.compile(
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        # ... other fake tensors ...
        options="--enable-tvm-ffi --opt-level 2",
    )
    return compiled_kernel
```

### 3. Workspace Memory Management
The module handles workspace memory allocation lazily on a per-device basis using a global dictionary (`_g_workspace`). 
The workspace size formula is matched to the kernel's requirements: `num_sms * num_heads * MAX_Q_LEN * (kv_lora_rank + 1) * sizeof(float32)`.

```python
# vllm/v1/attention/backends/mla/tokenspeed_mla.py
def _get_workspace(
    device: torch.device, num_heads: int, kv_lora_rank: int
) -> torch.Tensor:
    from tokenspeed_mla import get_num_sm
    
    needed = (
        get_num_sm(device) * num_heads * _TOKENSPEED_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    )
    existing = _g_workspace.get(device)
    if existing is None or existing.numel() < needed:
        _g_workspace[device] = torch.empty(needed, dtype=torch.int8, device=device)
    return _g_workspace[device]
```

This required size is deterministically retrieved and cached at the CuTe wrapper level to minimize calculation overhead on each forward pass:

```python
# tokenspeed_mla/mla_decode.py
@functools.cache
def _get_split_kv_and_workspace_size(
    B: int, q_len: int, H: int, kv_lora_rank: int, max_active_blocks: int,
) -> Tuple[int, int]:
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified(
        B, q_len, max_active_blocks
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        H, q_len, kv_lora_rank, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size
```

### 4. Precision and Scaling
When the kernel is executing the FP8 variant (`torch.float8_e4m3fn`), it is configured to explicitly return `bfloat16` outputs. This allows the backend to perform highly optimized math in FP8 while preventing precision loss in downstream layers.

```python
# FP8 kernel writes BF16 output for better downstream precision.
out_dtype = torch.bfloat16 if q_dtype == torch.float8_e4m3fn else q_dtype
o_k = torch.empty(
    (B, q_len, H, kv_lora_rank), dtype=out_dtype, device=query.device
)
```

Additionally, since the values (V) are stored in the FP8 cache divided by `k_scale`, the backend must pass explicit scales down to the kernel to un-quantize the output properly:

```python
# From tokenspeed_mla.py
self.softmax_scale = (
    self.scale * layer._q_scale_float * layer._k_scale_float
)
self.output_scale = layer._k_scale_float
```

### 5. Stream Synchronization
The kernel relies on TVM FFI for execution. PyTorch issues commands onto a specific CUDA stream, but TVM might launch kernels onto a default stream if not told otherwise. To prevent race conditions, the backend wraps the kernel call inside `tvm_ffi.use_torch_stream()`, binding TVM's execution explicitly to PyTorch's active CUDA stream.

```python
import tvm_ffi

with tvm_ffi.use_torch_stream():
    compiled_kernel(
        q_latent_k,
        q_rope_k,
        c_latent_k,
        c_rope_k,
        page_table_k,
        o_k,
        None,  # lse (disabled)
        workspace_bytes,
        Int32(split_kv),
        cache_seqs,
        block_split_kvs,
        Float32(softmax_scale),
        Float32(output_scale),
    )
```
