# Local Knowledge Hub

A local RAG (Retrieval-Augmented Generation) system that indexes a Wikipedia dataset into a vector database and lets you query it via a conversational chat interface — all running fully offline on your machine.

## How it works

1. **Indexing (`init` mode)**: Reads Wikipedia articles from Parquet files, splits them into text chunks, generates embeddings using a local Gemma-300M encoder model, and stores them in a binary vector database.
2. **Chat (`chat` mode)**: Takes a user query, embeds it, performs a cosine similarity search over the database, injects the top-k results as context, and generates a response using a local Gemma chat model.

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | b9012 | LLM inference + embeddings (Metal GPU on macOS) |
| [FTXUI](https://github.com/arthursonzogni/ftxui) | v5.0.0 | Terminal UI |
| [Apache Arrow](https://arrow.apache.org/) | 24.0.0 | Parquet file reading |

Dependencies are fetched automatically via CMake `FetchContent` (except Arrow, which requires Homebrew).

## Requirements

- CMake 3.22+
- Clang with C++17
- Apache Arrow via Homebrew: `brew install apache-arrow`
- Chat model: `gemma-4-E2B-it-Q4_K_M.gguf` in `models/` or anyone you'd like
- Embedding model: `embeddinggemma-300M-BF16.gguf` in `models/` or anyone you'd like
- Wikipedia Parquet dataset in `data/wikipedia.en/20231101.en/` or anyone you'd like

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Usage

### Index the database

Reads Parquet files, embeds all text chunks, and writes `vector-database.db`:

```bash
./build/lkh init
```

### Chat

Starts an interactive chat session with RAG context from the database:

```bash
./build/lkh chat
```

Type `exit` to quit.

## Project structure

```
src/
  main.cpp        — entry point, init/chat mode dispatch, inference loop
  database.hpp/cpp — binary vector database (save/load/search/add)
  embedding.hpp/cpp — text → float[] via Gemma-300M encoder (thread-safe)
  data.hpp/cpp    — Parquet reader (Apache Arrow), WikipediaDataRow
models/           — GGUF model files (not included)
data/             — Wikipedia Parquet dataset (not included)
```

## Known limitations & possible improvements

This is a hobby project — it works but there are plenty of rough edges:

- **Linear search**: cosine similarity is computed over every entry on each query. An ANN index (e.g. HNSW via hnswlib) would make search viable at millions of entries.
- **Full prompt re-encode**: the entire conversation history is re-tokenized and decoded from scratch on every turn. Proper KV-cache reuse would be much faster.
- **No incremental database updates**: re-running `init` overwrites the database. Appending new entries without a full rebuild would be useful.
- **Binary database format**: the custom `.db` format is fragile. Migrating to SQLite or a proper vector store would improve reliability.
- **Single-file context window**: with `n_ctx = 4096` the conversation length is limited. Larger models or sliding-window attention would help.
- **Hardcoded paths**: model and data paths are compile-time constants in `main.cpp` — they should be CLI arguments or a config file.
- **No UI yet**: the `ui` mode is a stub. A proper FTXUI interface was the original plan.

