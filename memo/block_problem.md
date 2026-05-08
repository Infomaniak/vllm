# kimik2.6

Initial problem on 8B300 GPU of KimiK2.6:

````
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 1234, in forward
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     def forward(
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/caching.py", line 215, in __call__
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     return self.optimized_call(*args, **kwargs)
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "<string>", line 503, in execution_fn
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "<string>", line 9, in __vllm_inlined_submods__2
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1269, in __call__
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     return self._op(*args, **kwargs)
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/kv_transfer_utils.py", line 40, in wrapper
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py", line 1067, in unified_mla_attention_with_output
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     layer.forward_impl(
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py", line 755, in forward_impl
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla.py", line 190, in forward_mqa
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     o = trtllm_batch_decode_with_kv_cache_mla(
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py", line 741, in trtllm_batch_decode_with_kv_cache_mla
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     kv_cache = _check_trtllm_gen_mla_shape(
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py", line 187, in _check_trtllm_gen_mla_shape
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962]     raise ValueError(
(Worker_TP5_EP5 pid=1935) ERROR 05-05 11:48:07 [multiproc_executor.py:962] ValueError: Expected block_num % (128 / block_size) == 0, got block_num=7813 and block_size=32
````

Until you can update to a vLLM version that includes PR #39324 (which will automatically pad these block tables behind the scenes so the backend is happy), the best practice is to manually set --max-model-len to a multiple of 128.

Because 128 / 32 = 4, making your max context length a perfect multiple of 128 mathematically guarantees the resulting block number will always satisfy the block_num % 4 == 0 requirement without triggering the bug.

To fix your exact issue, pass one of these aligned values to your vLLM startup command:

Option A (Round down slightly): --max-model-len 249984
(Math: 249,984 / 32 = 7812 blocks. 7812 % 4 = 0) ✔️

Option B (Round up slightly): --max-model-len 250112
(Math: 250,112 / 32 = 7816 blocks. 7816 % 4 = 0) ✔️

Option C (Standard 256K Context): --max-model-len 262144
(Math: 262,144 / 32 = 8192 blocks. 8192 % 4 = 0) ✔️
