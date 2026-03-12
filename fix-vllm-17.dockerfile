FROM vllm/vllm-openai:v0.17.1

# Install curl if not already available
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Download vllm/v1/core/sched/scheduler.py
RUN curl -o /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
    https://raw.githubusercontent.com/blancsw/vllm/refs/heads/fix-V0.12.0/vllm/v1/core/sched/scheduler.py \


COp