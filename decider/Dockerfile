# Strands Decider 2B served on CPU.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "strands-decider @ git+https://github.com/strands-labs/strands-decider"
ENV HF_HOME=/models
EXPOSE 8099
CMD ["strands-decider", "serve", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "--device", "cpu", "--host", "0.0.0.0", "--port", "8099"]
