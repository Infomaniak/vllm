# random-mm Benchmark — How Random Text Is Generated

## Entry point

`--dataset-name random-mm` → `RandomMultiModalDataset` (`vllm/benchmarks/datasets.py:802`)
Inherits text generation from `RandomDataset` (`datasets.py:438`).

---

## Text generation pipeline

### 1. Allowed tokens (`datasets.py:1139`)
```python
prohibited_tokens = [tok_id for tok_id, token in tokenizer.added_tokens_decoder.items()
                     if token.special]
allowed_tokens = np.arange(vocab_size) minus prohibited_tokens
```
Special/placeholder tokens (image placeholders, BOS, EOS…) are excluded.
For plain `RandomDataset`, `tokenizer.all_special_ids` is used instead.

### 2. Token sequence (`generate_token_sequence`, `datasets.py:639`)
```python
inner_seq = allowed_tokens[(offset + index + np.arange(input_len)) % len(allowed_tokens)]
token_sequence = prefix_token_ids + inner_seq
```
- `offset` = random int in `[0, vocab_size)` sampled per request
- `index` = request index (0, 1, 2, …)
- Result: a **deterministic arithmetic walk** through the allowed vocabulary (not random noise)

### 3. Decode/re-encode correction loop (`gen_prompt_decode_to_target_len`, `datasets.py:376`)
```
token_ids → tokenizer.decode() → string → tokenizer.encode() → truncate or pad
```
Retries up to 10 times to match the target token length.
Needed because some tokenizers (e.g. GPT-2) do not guarantee N tokens → N tokens round-trip.

### 4. Length sampling (`get_sampling_params`, `datasets.py:590`)
```
input_len  ∈ [floor(L * (1 - r)), ceil(L * (1 + r))]
output_len ∈ [floor(O * (1 - r)), ceil(O * (1 + r))]
```
Where `L` = `--random-input-len`, `O` = `--random-output-len`, `r` = `--random-range-ratio`.

### 5. Optional shared prefix (`get_prefix`, `datasets.py:557`)
- Generated once from `allowed_tokens` via the RNG
- Prepended to every request (useful for prefix-caching benchmarks)
- Controlled by `--random-prefix-len` (default 0)

---

## Multimodal content generation

### Images (`generate_synthetic_image`, `datasets.py:847`)
```python
pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
Image.fromarray(pixels)  # pure i.i.d. RGB noise
```
Worst-case for compression — very unlike real photos.

### Videos (`generate_synthetic_video`, `datasets.py:863`)
```python
pixels = rng.integers(0, 256, (num_frames, height, width, 3), dtype=np.uint8)
# Written to MP4 via cv2.VideoWriter at 30 fps
```

### Item count and bucket sampling (`get_mm_item_iterator`, `datasets.py:1036`)
- Count per request sampled uniformly from `[floor(n*(1-r)), ceil(n*(1+r))]`
- Each item type sampled from `bucket_config` (height, width, num_frames) → probability
- `num_frames == 1` → image; `num_frames > 1` → video
- Per-modality hard caps enforced via `limit_mm_per_prompt`

**Defaults:**
```python
bucket_config = {(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}
limit_mm_per_prompt = {"image": 255, "video": 1}
base_items_per_request = 1
```

---

## RNG

All sampling uses `numpy.default_rng(random_seed)` — isolated from global state, fully reproducible.

---

## Summary table

| Aspect | Detail |
|---|---|
| Text content | Arithmetic sequence through vocab — nonsense but length-accurate |
| Special tokens | Excluded (image placeholders, BOS/EOS, etc.) |
| Image pixels | Pure i.i.d. random RGB — max I/O stress, not photo-realistic |
| Video | Random frames encoded as MP4 via OpenCV |
| Reproducibility | Seeded numpy RNG, deterministic for same `--seed` |
| Prefix | Shared across all requests, generated once |
| Source file | `vllm/benchmarks/datasets.py` |