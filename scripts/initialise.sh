#!/bin/bash

# curl -LsSf https://hf.co/cli/install.sh | bash
# source ~/.bashrc

wget -O ./models/embeddinggemma-300M-BF16.gguf "https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-BF16.gguf?download=true"
wget -O ./models/Qwen3.5-0.8B-Q5_K_M.gguf "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q5_K_M.gguf?download=true"

hf download wikimedia/wikipedia --repo-type=dataset --include "./20231101.en/*" --local-dir ./data/wikipedia.en