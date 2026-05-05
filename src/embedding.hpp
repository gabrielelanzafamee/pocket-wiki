#pragma once

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <mutex>

#include <llama-cpp.h>

static constexpr int32_t EMBEDDING_MAX_CTX = 2048;

class Embedding
{
public:
    Embedding(const std::string& model_path);
    ~Embedding();

    std::vector<float> get_embedding(const std::string& input);
private:
    llama_model* model;
    llama_vocab* vocab;
    llama_context* ctx;
    int32_t num_max_seq;
    std::mutex mtx; // Metal GPU context is not thread-safe; serialize calls
};