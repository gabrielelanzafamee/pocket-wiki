#include "embedding.hpp"

Embedding::Embedding(const std::string& model_path)
{
    llama_model_params params = llama_model_default_params();
    this->model = llama_load_model_from_file(model_path.c_str(), params);
    if (!this->model) {
        std::cerr << "Error loading model from path: " << model_path << std::endl;
        return;
    }

    this->vocab = (llama_vocab*)llama_model_get_vocab(this->model);
    this->num_max_seq = llama_model_n_embd_out(this->model);

    // create context once and reuse it across all get_embedding calls
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx        = EMBEDDING_MAX_CTX;
    ctx_params.n_batch      = EMBEDDING_MAX_CTX;
    ctx_params.n_ubatch     = EMBEDDING_MAX_CTX;
    ctx_params.n_threads    = 4;
    ctx_params.embeddings   = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    this->ctx = llama_init_from_model(this->model, ctx_params);
    if (!this->ctx) {
        std::cerr << "Error creating llama context" << std::endl;
    }
}

Embedding::~Embedding() {
    if (this->ctx) {
        llama_free(this->ctx);
    }
    if (this->model) {
        llama_free_model(this->model);
    }
}

std::vector<float> Embedding::get_embedding(const std::string& input)
{
    if (!this->ctx) return {};

    std::lock_guard<std::mutex> lock(this->mtx);

    // tokenize
    int32_t n_tokens = -llama_tokenize(this->vocab, input.c_str(), input.size(), nullptr, 0, true, true);
    if (n_tokens <= 0) {
        std::cerr << "Error tokenizing input" << std::endl;
        return {};
    }

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(this->vocab, input.c_str(), input.size(), tokens.data(), n_tokens, true, true);
    if (n_tokens < 0) {
        std::cerr << "Error tokenizing input" << std::endl;
        return {};
    }

    // truncate to max context to avoid overflow warning
    if (n_tokens > EMBEDDING_MAX_CTX)
        n_tokens = EMBEDDING_MAX_CTX;
    tokens.resize(n_tokens);

    // clear memory from previous call before encoding
    llama_memory_clear(llama_get_memory(this->ctx), true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_encode(this->ctx, batch) != 0) {
        std::cerr << "Error encoding input batch" << std::endl;
        return {};
    }

    float* ctx_emb = llama_get_embeddings_seq(this->ctx, 0);
    if (!ctx_emb) {
        std::cerr << "Error getting embeddings from context" << std::endl;
        return {};
    }

    std::vector<float> embedding(this->num_max_seq);
    std::memcpy(embedding.data(), ctx_emb, this->num_max_seq * sizeof(float));
    return embedding;
}