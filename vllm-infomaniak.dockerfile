FROM vllm/vllm-openai:v0.20.0-ubuntu2404

# ---------- BUILD AND RUN TEST ----------
# docker build -f vllm-infomaniak.dockerfile -t registry.infomaniak.com:443/r-and-d/ai/k8s-llm/vllm-openai:cu130-nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52 .
# docker run -it --entrypoint=/bin/bash -p 8000:8000 registry.infomaniak.com:443/r-and-d/ai/k8s-llm/vllm-openai:cu130-nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52
# vllm serve --config debug.yaml
# ----------------------
# Overlay our forked Python sources on top of the pre-built vllm install.
#
# The base image already contains a fully-built vllm at
#   /usr/local/lib/python3.12/dist-packages/vllm
# including all compiled C/CUDA extensions (_C.abi3.so, _moe_C.abi3.so,
# vllm_flash_attn/*.so, cumem_allocator.abi3.so, ...).
#
# We only want to replace the Python source files, NOT the compiled
# extensions (rebuilding them would require the full toolchain and defeat
# the purpose of starting from the pre-built image).
#
# .dockerignore strips **/*.so, **/*.pyd, **/__pycache__, build/, dist/,
# .venv/, etc. from the build context, so this COPY only brings in .py
# (and other source) files. The pre-built .so files already in the image
# are left untouched.
#
# Note: COPY overlays — it does not delete files from the destination that
# are missing in the source. That's what we want here: the compiled
# extensions stay in place.
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY examples/ /vllm-workspace/examples/
COPY debug.yaml /vllm-workspace/debug.yaml