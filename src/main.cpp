#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <thread>
#include <mutex>

#include <llama-cpp.h>

#include "data.hpp"
#include "embedding.hpp"
#include "database.hpp"

#define MAX_BUFFER_SIZE 4096

const std::string MODEL_PATH = "./models/gemma-4-E2B-it-Q4_K_M.gguf";
const std::string TEMPLATE_NAME = "gemma";
const std::string DATABASE_PATH = "./vector-database.db";
const std::string EMB_MODEL_PATH = "./models/embeddinggemma-300M-BF16.gguf";
const std::string DATA_PARQUET_FOLDER = "./data/wikipedia.en/20231101.en/";

llama_model* init_llama()
{
    llama_model_params params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(MODEL_PATH.c_str(), params);
    return model;
}

llama_context* init_llama_context(llama_model* model)
{
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 4096;  // KV-cache / context window size
    ctx_params.n_batch   = 4096;  // must be >= n_ctx when encoding full prompt at once
    ctx_params.n_threads = 4;     // CPU threads for prompt processing

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    return ctx;
}

llama_vocab* get_vocab(llama_model* model)
{
    return (llama_vocab*)llama_model_get_vocab(model);
}

std::vector<llama_token> tokenize(llama_vocab* vocab, const std::string& text)
{
    int32_t n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n_tokens <= 0) {
        std::cerr << "Error tokenizing input" << std::endl;
        return {};
    }

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_tokens, true, true);
    if (n_tokens < 0) {
        std::cerr << "Error tokenizing input" << std::endl;
        return {};
    }

    return tokens;
}

std::string inference(
    llama_model *model, llama_context* ctx, llama_vocab* vocab, llama_sampler* smpl,
    llama_chat_message *messages, int n_messages)
{
    // --- 1. Apply chat template (add_ass=true so prompt ends with the assistant turn opener) ---
    char* prompt = (char*)malloc(MAX_BUFFER_SIZE);
    if (!prompt) {
        std::cerr << "Error allocating prompt buffer" << std::endl;
        return nullptr;
    }

    int32_t prompt_len = llama_chat_apply_template(
        TEMPLATE_NAME.c_str(),
        messages,
        n_messages,
        true,   // add_ass: append the assistant turn start token
        prompt,
        MAX_BUFFER_SIZE
    );

    if (prompt_len < 0) {
        std::cerr << "Error applying chat template: " << prompt_len << std::endl;
        free(prompt);
        return nullptr;
    }
    prompt[prompt_len] = '\0';
  
    std::vector<llama_token> tokens = tokenize(vocab, prompt);
    free(prompt);

    int n_tokens = tokens.size();

    if (n_tokens < 0) {
        std::cerr << "Error tokenizing prompt" << std::endl;
        llama_free(ctx);
        return nullptr;
    }

    {   // clear KV cache: we re-encode the full history on every call
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Error decoding prompt batch" << std::endl;
            llama_free(ctx);
            return nullptr;
        }
    }

    std::vector<llama_token> response_tokens;
    const int max_new_tokens = 512;

    for (int i = 0; i < max_new_tokens; ++i) {
        llama_token token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) break;

        response_tokens.push_back(token);

        // feed the new token back for the next step
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Error decoding generated token" << std::endl;
            break;
        }
    }

    // detokenize with special tokens suppressed
    int32_t text_len = llama_detokenize(
        vocab,
        response_tokens.data(),
        (int32_t)response_tokens.size(),
        nullptr, 0,
        /*remove_special=*/true,
        /*unparse_special=*/false);
    if (text_len < 0) text_len = -text_len; // negative = required size
    std::string response(text_len, '\0');
    llama_detokenize(
        vocab,
        response_tokens.data(),
        (int32_t)response_tokens.size(),
        response.data(), text_len,
        /*remove_special=*/true,
        /*unparse_special=*/false);

    while (!response.empty() && (response.back() == '\n' || response.back() == ' ')) {
        response.pop_back();
    }
    return response;
}

void ui_handler()
{
    ftxui::Element document = ftxui::hbox({
        ftxui::text("left") | ftxui::border,
        ftxui::text("middle") | ftxui::border | ftxui::flex,
        ftxui::text("right") | ftxui::border,
    });

    // Create a screen with full width and height fitting the document.
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Full(),       // Width
        ftxui::Dimension::Fit(document) // Height
    );

    // Render the document onto the screen.
    ftxui::Render(screen, document);

    // Print the screen to the console.
    screen.Print();
}

void initialisation_database()
{
    // scan folder data for parquet files and read them using the DataReader class, then print the first 10 rows of the first file for testing
    // create a Database based on VectorDatabase 

    // list files in the data folder
    std::vector<std::string> parquet_files;
    for (const auto& entry : std::filesystem::directory_iterator(DATA_PARQUET_FOLDER))
        if (entry.path().extension() == ".parquet")
            parquet_files.push_back(entry.path().string());

    if (parquet_files.empty())
    {
        std::cerr << "No parquet files found in data folder" << std::endl;
        return;
    }

    int32_t total_rows = 0;
    int32_t chunk_rows_size = 1024;
    int32_t num_workers = 1;
    int32_t text_chunk_size = 1024; // chunk text into 512 tokens for embedding to avoid OOM, will need to experiment with this value to find the optimal one for performance and memory usage

    Database db(DATABASE_PATH);
    Embedding embedding(EMB_MODEL_PATH);
    std::mutex db_mutex;
    std::mutex total_rows_mutex;

    auto process_file = [&](const std::string& file)
    {
        DataReader reader(file);
        std::shared_ptr<arrow::Table> table = reader.read_data();

        if (!table) {
            std::cerr << "Failed to read data from file: " << file << std::endl;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(total_rows_mutex);
            total_rows += table->num_rows();
        }

        for (int32_t i = 0; i < table->num_rows(); i += chunk_rows_size)
        {
            int32_t end = std::min(i + chunk_rows_size, (int32_t)table->num_rows());
            std::vector<WikipediaDataRow> rows = reader.get_rows(i, end);

            for (const auto& row : rows)
            {
                std::vector<std::string> text_chunks;
                for (size_t start = 0; start < row.text.size(); start += text_chunk_size) {
                    text_chunks.push_back(row.text.substr(start, text_chunk_size));
                }

                for (const auto& chunk : text_chunks)
                {
                    Entry entry;
                    entry.document_id = row.id; 
                    entry.embedding = std::move(embedding.get_embedding(chunk));
                    entry.title = row.title;
                    entry.url = row.url;
                    entry.value = chunk;
    
                    std::lock_guard<std::mutex> lock(db_mutex);
                    db.add_entry(entry);

                    std::cout << "Processing chunk ID: " << entry.id << std::endl;
                    std::cout << "Document ID: " << entry.document_id << std::endl;
                    std::cout << "Title: " << entry.title << std::endl;
                    std::cout << "URL: " << entry.url << std::endl;
                    std::cout << "Value: " << entry.value.substr(0, 200) << "..." << std::endl;
                    std::cout << "Embedding sample: " << (entry.embedding.empty() ? "N/A" : std::to_string(entry.embedding[0])) << "..." << std::endl;
                    std::cout << "-----------------------------" << std::endl;

                    // print progress
                    {
                        std::lock_guard<std::mutex> lock(total_rows_mutex);
                        std::cout << "Processed " << std::min(end, (int32_t)table->num_rows()) << "/" << table->num_rows() << " rows from file: " << file << std::endl;
                    }
                }

            }
        }
    };

    // process all files in batches of num_workers
    for (size_t batch_start = 0; batch_start < parquet_files.size(); batch_start += num_workers)
    {
        size_t batch_end = std::min(batch_start + (size_t)num_workers, parquet_files.size());
        std::vector<std::thread> workers_threads;
        workers_threads.reserve(batch_end - batch_start);

        for (size_t i = batch_start; i < batch_end; i++)
            workers_threads.emplace_back(process_file, parquet_files[i]);

        for (auto& t : workers_threads)
            t.join();
    }

    // for (const auto& file : parquet_files)
    // {
    //     DataReader reader(file);
    //     std::shared_ptr<arrow::Table> table = reader.read_data();

    //     if (!table) {
    //         std::cerr << "Failed to read data from file: " << file << std::endl;
    //         continue;
    //     }

    //     total_rows += table->num_rows();

    //     // preprocess and store it
    //     for (int32_t i = 0; i < table->num_rows(); i += chunk_rows_size)
    //     {
    //         int32_t end = std::min(i + chunk_rows_size, (int32_t)table->num_rows());
    //         std::vector<WikipediaDataRow> rows = reader.get_rows(i, end);

    //         // add to database
    //         for (const auto& row : rows)
    //         {
    //             db.add_entry(
    //                 {
    //                     .id = row.id,
    //                     .title = row.title,
    //                     .url = row.url,
    //                     .value = row.text,
    //                     .embedding = embedding.get_embedding(row.text)
    //                 }
    //             );
    //         }
    //     }
    // }

    db.save();
    std::cout << "Database initialised with " << total_rows << " rows!" << std::endl;
}

void llama_chat_handler()
{
    llama_model* model = init_llama();
    llama_context* ctx = init_llama_context(model);
    llama_vocab* vocab = get_vocab(model);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    Database db(DATABASE_PATH);
    db.load();
    Embedding embedding(EMB_MODEL_PATH);

    // Store message strings separately so the char* pointers in llama_chat_message remain valid
    std::vector<std::pair<std::string, std::string>> message_store = {
        {"system", "You are a helpful assistant that answers questions based on the provided database of information and your knowladge base."},
    };

    bool is_running = true;

    while (is_running) {
        std::string user_input;
        std::cout << "User: ";
        std::getline(std::cin, user_input);

        if (user_input == "exit") {
            is_running = false;
            continue;
        }

        // search the database for relevant information based on the user input and add it to the messages as context for the assistant
        std::vector<float> query_embedding = embedding.get_embedding(user_input);
        std::vector<Entry> search_results = db.search(query_embedding, 5); // get top 5 results

        std::string context_info = "Relevant information from the database:\n";
        std::cout << "\n[Context fetched (" << search_results.size() << " results)]" << std::endl;
        for (const auto& entry : search_results) {
            std::cout << "  [" << entry.id << "] " << entry.title << " (score entry): " << entry.value.substr(0, 100) << "..." << std::endl;
            context_info += "- " + entry.title + ": " + entry.value.substr(0, 200) + "...\n";
        }

        std::string prompt = context_info + "\nUser question: " + user_input;
        message_store.emplace_back("user", std::move(prompt));

        // Build llama_chat_message view from stored strings (pointers stay valid)
        std::vector<llama_chat_message> messages;
        messages.reserve(message_store.size());

        for (const auto& [role, content] : message_store)
            messages.push_back({role.c_str(), content.c_str()});

        // only get the last 2 messages for context to avoid exceeding the context window, in the future we can implement a more sophisticated strategy to select which messages to include based on relevance and recency
        // but keep system prompt always included
        if (messages.size() > 3) {
            messages.erase(messages.begin() + 1, messages.end() - 2);
        }

        std::string response = inference(model, ctx, vocab, smpl, messages.data(), messages.size());
        if (response.empty()) {
            std::cerr << "Inference failed" << std::endl;
            continue;
        }

        std::cout << "Assistant: " << response << std::endl;
        message_store.emplace_back("assistant", std::move(response));
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
}

// argc; [init|chat|ui - database_path]
int main(int argc, char *argv[])
{
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode>" << std::endl;
        std::cerr << "Modes: init, chat, ui" << std::endl;
        return 1;
    }
    std::string mode = argv[1];

    if (mode == "init") {
        initialisation_database();
    } else if (mode == "chat") {
        llama_chat_handler();
    } else {
        std::cerr << "Invalid mode: " << mode << std::endl;
    }

    // llama_execution_test();
    // llama_backend_free();

    return 0;

    // DataReader reader("/Users/gabrielelanzafame/Documents/Codes/local-knowledge-hub/data/wikipedia.en/20231101.en/train-00000-of-00041.parquet");
    // std::vector<WikipediaDataRow> rows = reader.get_rows(0, 10);

    // const char* emb_model_path = "/Users/gabrielelanzafame/Documents/Codes/local-knowledge-hub/models/embeddinggemma-300M-BF16.gguf";
    // Embedding embedding(emb_model_path);
    // std::vector<float> emb = embedding.get_embedding("What is the capital of France?");
    // std::cout << "Embedding size: " << emb.size() << std::endl;

    // Database db(DATABASE_PATH);
    // db.load();

    // std::vector<Entry> entries = db.get_all_entries();

    // for (const auto& entry : entries) {
    //     std::cout << "ID: " << entry.id << std::endl;
    //     std::cout << "Title: " << entry.title << std::endl;
    //     std::cout << "URL: " << entry.url << std::endl;
    //     std::cout << "Value: " << entry.value.substr(0, 200) << "..." << std::endl; // print first 200 chars
    //     std::cout << "-----------------------------" << std::endl;
    // }

    // // add the first 10 rows of the parquet file to the database with their embedding

    // for (const auto& row : rows) {
    //     Entry entry;
    //     entry.id = row.id;
    //     entry.title = row.title;
    //     entry.url = row.url;
    //     entry.value = row.text;
    //     entry.embedding = embedding.get_embedding(row.text);
    //     db.add_entry(entry);
    // }

    // // save
    // db.save();

    // // print the first 10 entries in the database for testing
    // entries = db.list(0, 10);

    // for (const auto& entry : entries) {
    //     std::cout << "ID: " << entry.id << std::endl;
    //     std::cout << "Title: " << entry.title << std::endl;
    //     std::cout << "URL: " << entry.url << std::endl;
    //     std::cout << "Value: " << entry.value.substr(0, 200) << "..." << std::endl; // print first 200 chars
    //     std::cout << "-----------------------------" << std::endl;
    // }

    // // for (const auto& row : rows) {
    // //     std::cout << "ID: " << row.id << std::endl;
    // //     std::cout << "Title: " << row.title << std::endl;
    // //     std::cout << "URL: " << row.url << std::endl;
    // //     std::cout << "Text: " << row.text.substr(0, 200) << "..." << std::endl; // print first 200 chars
    // //     std::cout << "-----------------------------" << std::endl;
    // // }


    // std::cout << "Data read successfully!" << std::endl;
}