# Apertus 1.5 Dockerfile for vLLM
# Builds on top of the official vLLM OpenAI image, integrating custom
# Apertus 1.5 branches for Hugging Face Transformers and vLLM.

ARG BASE_IMAGE=docker.io/vllm/vllm-openai:v0.27.1
FROM ${BASE_IMAGE}

ARG TRANSFORMERS_REPO=https://github.com/swiss-ai/transformers.git
ARG TRANSFORMERS_BRANCH=add-apertus1p5
ARG VLLM_REPO=https://github.com/Infomaniak/vllm.git
ARG VLLM_BRANCH=apertus1-5
ARG VLLM_VERSION=0.27.1
ARG CUDA_VERSION=13.0

# 1. Install git and Python build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
        "packaging>=24.2" \
        ninja \
        "cmake>=3.26.1" \
        "setuptools-scm>=8" \
        setuptools_rust \
        build \
        wheel \
        "jinja2>=3.1.6"

# 2. Install custom Transformers branch
RUN python3 -m pip install --no-cache-dir --no-deps \
    "git+${TRANSFORMERS_REPO}@${TRANSFORMERS_BRANCH}"

# 3. Install custom vLLM branch using precompiled C++/CUDA extensions
RUN python3 -m pip download --no-deps --no-cache-dir -d /tmp "vllm==${VLLM_VERSION}" \
    && git clone --depth 1 --branch ${VLLM_BRANCH} ${VLLM_REPO} /tmp/vllm \
    && python3 -m pip uninstall -y vllm \
    && VLLM_USE_PRECOMPILED=1 \
       VLLM_PRECOMPILED_WHEEL_LOCATION="$(ls /tmp/vllm-${VLLM_VERSION}*.whl | head -n 1)" \
       VLLM_MAIN_CUDA_VERSION=${CUDA_VERSION} \
       VLLM_PRECOMPILED_WHEEL_VARIANT=cu$(echo ${CUDA_VERSION} | tr -d .) \
       python3 -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/vllm \
    && rm -rf /tmp/vllm /tmp/vllm-*.whl \
    && python3 -c "import vllm, transformers; print(f'vLLM: {vllm.__version__} | Transformers: {transformers.__version__}')"

# Bypass minor flashinfer cubin patch/post mismatch between runtime packages
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

