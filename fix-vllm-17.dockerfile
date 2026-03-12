FROM vllm/vllm-openai:nightly-36735fd77224467e6580f3bd48eb32d4fca8c72e

# Install curl if not already available
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py \
    https://raw.githubusercontent.com/Infomaniak/vllm/f9ed45ab7d4c6ddcf4e1f0e404a0b8adfd4707de/vllm/v1/attention/backends/gdn_attn.py