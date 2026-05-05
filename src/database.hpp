#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

typedef struct Entry
{
    uint32_t id = -1; // default value for uninitialized entry, will be set to a unique value when added to the database
    uint32_t document_id;
    std::vector<float> embedding;
    std::string title;
    std::string url;
    std::string value;
} Entry;

/**
 * Database can read / write a file where store all the data, save in binary and use .db as extension, for simplicity I'm just using a vector in memory here but in the future I can implement a more efficient storage solution (e.g. sqlite, leveldb, etc.) and also add support for incremental updates to the database without having to rewrite the entire file each time
 * Can do search based on cosine similarity of the embeddings, for simplicity I'm just doing a linear scan of the entries in memory here but in the future I can implement a more efficient search solution (e.g. HNSW) and also add support for filtering based on metadata (e.g. title, url) if needed
 * Needs to run in 1GB RAM so need to be efficient in terms of memory usage, for simplicity I'm just storing the embeddings as std::vector<float> here but in the future I can implement a more efficient storage solution (e.g. quantization) if needed
 */

class Database
{
public:
    Database(const std::string& database_path);
    ~Database();

    void add_entry(Entry& entry);
    
    std::vector<Entry> search(const std::vector<float>& query_embedding, int top_k);
    std::vector<Entry> list(int start, int end);
    std::vector<Entry> get_all_entries();
    Entry get_entry_by_id(uint32_t id);

    void save();
    void load();
private:
    uint32_t last_id = 0;
    std::vector<Entry> entries;
    std::string database_path;

    // utils
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
    bool equal_embeddings(const std::vector<float>& a, const std::vector<float>& b);
};