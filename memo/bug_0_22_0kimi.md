````

(Worker_TP0 pid=1911)
(Worker_TP0 pid=1911) INFO 06-01 17:31:06 [default_loader.py:397] Loading weights took 838.42 seconds
(Worker_TP0 pid=1911) WARNING 06-01 17:31:07 [kv_cache.py:109] Checkpoint does not provide a q scaling factor. Setting it to k_scale. This only matters for FP8 Attention backends (flash-attn or flashinfer).
(Worker_TP0 pid=1911) WARNING 06-01 17:31:07 [kv_cache.py:123] Using KV cache scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that k/v_scale scaling factors are properly set in the checkpoint.
(Worker_TP0 pid=1911) INFO 06-01 17:31:12 [nvfp4.py:537] Using MoEPrepareAndFinalizeNoDPEPMonolithic
(Worker_TP1 pid=1912) INFO 06-01 17:31:18 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(Worker_TP1 pid=1912) INFO 06-01 17:31:18 [selector.py:130] Using HND KV cache layout for FLASHINFER_MLA backend.
(Worker_TP1 pid=1912) INFO 06-01 17:31:18 [selector.py:126] Using TOKENSPEED_MLA MLA prefill backend.
(Worker_TP0 pid=1911) INFO 06-01 17:31:18 [gpu_model_runner.py:5061] Loading drafter model...
(Worker_TP0 pid=1911) INFO 06-01 17:31:18 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(Worker_TP2 pid=1913) INFO 06-01 17:31:18 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(Worker_TP0 pid=1911) INFO 06-01 17:31:18 [cuda.py:378] Using FLASHINFER_MLA attention backend out of potential backends: ['FLASHINFER_MLA', 'TOKENSPEED_MLA', 'TRITON_MLA'].
(Worker_TP0 pid=1911) INFO 06-01 17:31:18 [selector.py:130] Using HND KV cache layout for FLASHINFER_MLA backend.
(Worker_TP0 pid=1911) INFO 06-01 17:31:18 [selector.py:126] Using TOKENSPEED_MLA MLA prefill backend.
(Worker_TP2 pid=1913) INFO 06-01 17:31:18 [selector.py:130] Using HND KV cache layout for FLASHINFER_MLA backend.
(Worker_TP2 pid=1913) INFO 06-01 17:31:18 [selector.py:126] Using TOKENSPEED_MLA MLA prefill backend.
(Worker_TP3 pid=1914) INFO 06-01 17:31:18 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
(Worker_TP3 pid=1914) INFO 06-01 17:31:18 [selector.py:130] Using HND KV cache layout for FLASHINFER_MLA backend.
(Worker_TP3 pid=1914) INFO 06-01 17:31:18 [selector.py:126] Using TOKENSPEED_MLA MLA prefill backend.
(Worker_TP1 pid=1912) INFO 06-01 17:31:27 [weight_utils.py:603] Time spent downloading weights for lightseekorg/kimi-k2.6-eagle3.1-mla: 9.181237 seconds
(Worker_TP1 pid=1912) INFO 06-01 17:31:27 [weight_utils.py:647] No model.safetensors.index.json found in remote.
(Worker_TP2 pid=1913) INFO 06-01 17:31:28 [weight_utils.py:647] No model.safetensors.index.json found in remote.
(Worker_TP0 pid=1911) INFO 06-01 17:31:28 [weight_utils.py:647] No model.safetensors.index.json found in remote.
(Worker_TP0 pid=1911) INFO 06-01 17:31:28 [weight_utils.py:922] Filesystem type for checkpoints: OVERLAY. Checkpoint size: 5.62 GiB. Available RAM: 1365.26 GiB.
(Worker_TP0 pid=1911) INFO 06-01 17:31:28 [weight_utils.py:945] Auto-prefetch is disabled because the filesystem (OVERLAY) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 1371.14it/s]
(Worker_TP0 pid=1911)
(Worker_TP3 pid=1914) INFO 06-01 17:31:28 [weight_utils.py:647] No model.safetensors.index.json found in remote.
(Worker_TP0 pid=1911) INFO 06-01 17:31:29 [default_loader.py:397] Loading weights took 1.13 seconds
(Worker_TP0 pid=1911) INFO 06-01 17:31:39 [llm_base_proposer.py:1314] Detected EAGLE model with embed_tokens identical to the target model. Sharing target model embedding weights with the draft model.
(Worker_TP2 pid=1913) INFO 06-01 17:31:39 [llm_base_proposer.py:1314] Detected EAGLE model with embed_tokens identical to the target model. Sharing target model embedding weights with the draft model.
(Worker_TP1 pid=1912) INFO 06-01 17:31:39 [llm_base_proposer.py:1314] Detected EAGLE model with embed_tokens identical to the target model. Sharing target model embedding weights with the draft model.
(Worker_TP3 pid=1914) INFO 06-01 17:31:39 [llm_base_proposer.py:1314] Detected EAGLE model with embed_tokens identical to the target model. Sharing target model embedding weights with the draft model.
(Worker_TP0 pid=1911) INFO 06-01 17:31:40 [llm_base_proposer.py:1376] Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head weights from the target model.
(Worker_TP0 pid=1911) INFO 06-01 17:31:40 [gpu_model_runner.py:5217] Using auxiliary layers from speculative config: (2, 30, 58)
(Worker_TP2 pid=1913) INFO 06-01 17:31:40 [llm_base_proposer.py:1376] Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head weights from the target model.
(Worker_TP2 pid=1913) INFO 06-01 17:31:40 [gpu_model_runner.py:5217] Using auxiliary layers from speculative config: (2, 30, 58)
(Worker_TP3 pid=1914) INFO 06-01 17:31:40 [llm_base_proposer.py:1376] Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head weights from the target model.
(Worker_TP3 pid=1914) INFO 06-01 17:31:40 [gpu_model_runner.py:5217] Using auxiliary layers from speculative config: (2, 30, 58)
(Worker_TP1 pid=1912) INFO 06-01 17:31:40 [llm_base_proposer.py:1376] Detected EAGLE model with distinct lm_head weights. Keeping separate lm_head weights from the target model.
(Worker_TP1 pid=1912) INFO 06-01 17:31:40 [gpu_model_runner.py:5217] Using auxiliary layers from speculative config: (2, 30, 58)
(Worker_TP0 pid=1911) INFO 06-01 17:31:43 [gpu_model_runner.py:5132] Model loading took 142.31 GiB memory and 912.235622 seconds
(Worker_TP0 pid=1911) INFO 06-01 17:31:43 [gpu_model_runner.py:6136] Encoder cache will be initialized with a budget of 32768 tokens, and profiled with 7 vision_chunk items of the maximum feature size.
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962] WorkerProc hit an exception.
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962] Traceback (most recent call last):
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 957, in worker_busy_loop
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 396, in determine_available_memory
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     self.model_runner.profile_run()
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 6152, in profile_run
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     dummy_encoder_outputs = self.model.embed_multimodal(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 431, in embed_multimodal
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vision_embeddings = self._process_media_input(media_input)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 415, in _process_media_input
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     media_features = vision_tower_forward(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                      ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 645, in vision_tower_forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vt_outputs = run_dp_sharded_mrope_vision_model(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/vision.py", line 479, in run_dp_sharded_mrope_vision_model
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     image_embeds_local = vision_model(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                          ^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 603, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.encoder(hidden_states, grid_thws)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 515, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = block(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 450, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.attention_qkvpacked(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 426, in attention_qkvpacked
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     attn_out = self.attn(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                ^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_method(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 721, in forward_cuda
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 555, in _forward_fa
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = vit_flash_attn_wrapper(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 100, in vit_flash_attn_wrapper
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1269, in __call__
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._op(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 51, in flash_attn_maxseqlen_wrapper
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = flash_attn_varlen_func(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py", line 372, in flash_attn_varlen_func
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     out, softmax_lse = _flash_attn_fwd(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                        ^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/interface.py", line 888, in _flash_attn_fwd
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     fa_fwd = flash_fwd_obj_cls(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/flash_fwd_sm100.py", line 162, in __init__
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962] AssertionError: Only SM 10.x and 11.x are supported
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962] Traceback (most recent call last):
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 957, in worker_busy_loop
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 396, in determine_available_memory
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     self.model_runner.profile_run()
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 6152, in profile_run
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     dummy_encoder_outputs = self.model.embed_multimodal(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 431, in embed_multimodal
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vision_embeddings = self._process_media_input(media_input)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 415, in _process_media_input
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     media_features = vision_tower_forward(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                      ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 645, in vision_tower_forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vt_outputs = run_dp_sharded_mrope_vision_model(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/vision.py", line 479, in run_dp_sharded_mrope_vision_model
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     image_embeds_local = vision_model(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                          ^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 603, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.encoder(hidden_states, grid_thws)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 515, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = block(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 450, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.attention_qkvpacked(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 426, in attention_qkvpacked
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     attn_out = self.attn(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                ^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_method(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 721, in forward_cuda
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962] WorkerProc hit an exception.
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962] Traceback (most recent call last):
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 555, in _forward_fa
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 957, in worker_busy_loop
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = vit_flash_attn_wrapper(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 100, in vit_flash_attn_wrapper
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1269, in __call__
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 396, in determine_available_memory
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._op(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     self.model_runner.profile_run()
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 6152, in profile_run
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 51, in flash_attn_maxseqlen_wrapper
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     dummy_encoder_outputs = self.model.embed_multimodal(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = flash_attn_varlen_func(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 431, in embed_multimodal
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py", line 372, in flash_attn_varlen_func
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vision_embeddings = self._process_media_input(media_input)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     out, softmax_lse = _flash_attn_fwd(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                        ^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 415, in _process_media_input
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/interface.py", line 888, in _flash_attn_fwd
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     media_features = vision_tower_forward(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     fa_fwd = flash_fwd_obj_cls(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                      ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/flash_fwd_sm100.py", line 162, in __init__
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 645, in vision_tower_forward
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962] AssertionError: Only SM 10.x and 11.x are supported
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vt_outputs = run_dp_sharded_mrope_vision_model(
(Worker_TP0 pid=1911) ERROR 06-01 17:32:02 [multiproc_executor.py:962]
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/vision.py", line 479, in run_dp_sharded_mrope_vision_model
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     image_embeds_local = vision_model(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                          ^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 603, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.encoder(hidden_states, grid_thws)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 515, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = block(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 450, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.attention_qkvpacked(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 426, in attention_qkvpacked
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     attn_out = self.attn(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                ^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_method(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 721, in forward_cuda
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 555, in _forward_fa
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = vit_flash_attn_wrapper(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 100, in vit_flash_attn_wrapper
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1269, in __call__
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._op(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 51, in flash_attn_maxseqlen_wrapper
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = flash_attn_varlen_func(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py", line 372, in flash_attn_varlen_func
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     out, softmax_lse = _flash_attn_fwd(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                        ^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/interface.py", line 888, in _flash_attn_fwd
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     fa_fwd = flash_fwd_obj_cls(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/flash_fwd_sm100.py", line 162, in __init__
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962] AssertionError: Only SM 10.x and 11.x are supported
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962] Traceback (most recent call last):
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 957, in worker_busy_loop
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = func(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 396, in determine_available_memory
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     self.model_runner.profile_run()
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 6152, in profile_run
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     dummy_encoder_outputs = self.model.embed_multimodal(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 431, in embed_multimodal
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vision_embeddings = self._process_media_input(media_input)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py", line 415, in _process_media_input
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     media_features = vision_tower_forward(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                      ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return func(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 645, in vision_tower_forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     vt_outputs = run_dp_sharded_mrope_vision_model(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/vision.py", line 479, in run_dp_sharded_mrope_vision_model
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     image_embeds_local = vision_model(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                          ^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 603, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.encoder(hidden_states, grid_thws)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 515, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = block(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 450, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     hidden_states = self.attention_qkvpacked(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25_vit.py", line 426, in attention_qkvpacked
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     attn_out = self.attn(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                ^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._call_impl(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return forward_call(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/custom_op.py", line 136, in forward
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_method(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 721, in forward_cuda
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py", line 555, in _forward_fa
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = vit_flash_attn_wrapper(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 100, in vit_flash_attn_wrapper
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return torch.ops.vllm.flash_attn_maxseqlen_wrapper(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1269, in __call__
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     return self._op(*args, **kwargs)
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py", line 51, in flash_attn_maxseqlen_wrapper
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     output = flash_attn_varlen_func(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py", line 372, in flash_attn_varlen_func
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     out, softmax_lse = _flash_attn_fwd(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]                        ^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/interface.py", line 888, in _flash_attn_fwd
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     fa_fwd = flash_fwd_obj_cls(
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]              ^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]   File "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/cute/flash_fwd_sm100.py", line 162, in __init__
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]     assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962] AssertionError: Only SM 10.x and 11.x are supported
(Worker_TP2 pid=1913) ERROR 06-01 17:32:02 [multiproc_executor.py:962]
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165] EngineCore failed to start.
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165] Traceback (most recent call last):
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1139, in run_engine_core
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     return func(*args, **kwargs)
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 905, in __init__
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     super().__init__(
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 131, in __init__
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     return func(*args, **kwargs)
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 253, in _initialize_kv_caches
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 147, in determine_available_memory
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     return self.collective_rpc("determine_available_memory")
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 403, in collective_rpc
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     return future if non_block else future.result()
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]                                     ^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 90, in result
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     return super().result()
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]            ^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     return self.__get_result()
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     raise self._exception
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 94, in _wait_for_response
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     response = self.aggregate(self.get_response())
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]                               ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 390, in get_response
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165]     raise RuntimeError(
(EngineCore pid=1709) ERROR 06-01 17:32:02 [core.py:1165] RuntimeError: Worker failed with error 'Only SM 10.x and 11.x are supported', please check the stack trace above for the root cause
(Worker_TP0 pid=1911) WARNING 06-01 17:32:02 [multiproc_executor.py:884] WorkerProc was terminated
(Worker_TP2 pid=1913) WARNING 06-01 17:32:02 [multiproc_executor.py:884] WorkerProc was terminated
(Worker_TP3 pid=1914) WARNING 06-01 17:32:02 [multiproc_executor.py:884] WorkerProc was terminated
(Worker_TP1 pid=1912) WARNING 06-01 17:32:02 [multiproc_executor.py:884] WorkerProc was terminated
(EngineCore pid=1709) ERROR 06-01 17:32:04 [multiproc_executor.py:283] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.
(EngineCore pid=1709) Process EngineCore:
(EngineCore pid=1709) Traceback (most recent call last):
(EngineCore pid=1709)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore pid=1709)     self.run()
(EngineCore pid=1709)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore pid=1709)     self._target(*self._args, **self._kwargs)
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1169, in run_engine_core
(EngineCore pid=1709)     raise e
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1139, in run_engine_core
(EngineCore pid=1709)     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
(EngineCore pid=1709)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1709)     return func(*args, **kwargs)
(EngineCore pid=1709)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 905, in __init__
(EngineCore pid=1709)     super().__init__(
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 131, in __init__
(EngineCore pid=1709)     kv_cache_config = self._initialize_kv_caches(vllm_config)
(EngineCore pid=1709)                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(EngineCore pid=1709)     return func(*args, **kwargs)
(EngineCore pid=1709)            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 253, in _initialize_kv_caches
(EngineCore pid=1709)     available_gpu_memory = self.model_executor.determine_available_memory()
(EngineCore pid=1709)                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 147, in determine_available_memory
(EngineCore pid=1709)     return self.collective_rpc("determine_available_memory")
(EngineCore pid=1709)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 403, in collective_rpc
(EngineCore pid=1709)     return future if non_block else future.result()
(EngineCore pid=1709)                                     ^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 90, in result
(EngineCore pid=1709)     return super().result()
(EngineCore pid=1709)            ^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(EngineCore pid=1709)     return self.__get_result()
(EngineCore pid=1709)            ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(EngineCore pid=1709)     raise self._exception
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 94, in _wait_for_response
(EngineCore pid=1709)     response = self.aggregate(self.get_response())
(EngineCore pid=1709)                               ^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1709)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 390, in get_response
(EngineCore pid=1709)     raise RuntimeError(
(EngineCore pid=1709) RuntimeError: Worker failed with error 'Only SM 10.x and 11.x are supported', please check the stack trace above for the root cause
(APIServer pid=963) Traceback (most recent call last):
(APIServer pid=963)   File "/usr/local/bin/vllm", line 10, in <module>
(APIServer pid=963)     sys.exit(main())
(APIServer pid=963)              ^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/main.py", line 92, in main
(APIServer pid=963)     args.dispatch_function(args)
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/serve.py", line 148, in cmd
(APIServer pid=963)     uvloop.run(run_server(args))
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 96, in run
(APIServer pid=963)     return __asyncio.run(
(APIServer pid=963)            ^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
(APIServer pid=963)     return runner.run(main)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
(APIServer pid=963)     return self._loop.run_until_complete(task)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 48, in wrapper
(APIServer pid=963)     return await main
(APIServer pid=963)            ^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 678, in run_server
(APIServer pid=963)     await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 692, in run_server_worker
(APIServer pid=963)     async with build_async_engine_client(
(APIServer pid=963)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=963)     return await anext(self.gen)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 100, in build_async_engine_client
(APIServer pid=963)     async with build_async_engine_client_from_engine_args(
(APIServer pid=963)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=963)     return await anext(self.gen)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 136, in build_async_engine_client_from_engine_args
(APIServer pid=963)     async_llm = AsyncLLM.from_vllm_config(
(APIServer pid=963)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 217, in from_vllm_config
(APIServer pid=963)     return cls(
(APIServer pid=963)            ^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 146, in __init__
(APIServer pid=963)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=963)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=963)     return func(*args, **kwargs)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 131, in make_async_mp_client
(APIServer pid=963)     return AsyncMPClient(*client_args)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=963)     return func(*args, **kwargs)
(APIServer pid=963)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 932, in __init__
(APIServer pid=963)     super().__init__(
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 567, in __init__
(APIServer pid=963)     with launch_core_engines(
(APIServer pid=963)   File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=963)     next(self.gen)
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1150, in launch_core_engines
(APIServer pid=963)     wait_for_engine_startup(
(APIServer pid=963)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1209, in wait_for_engine_startup
(APIServer pid=963)     raise RuntimeError(
(APIServer pid=963) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
root@vllm-dev-deployment-544bf76949-pj8d5:/vllm-workspace#
````