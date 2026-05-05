#include "database.hpp"
#include <algorithm>

Database::Database(const std::string& database_path)
{
    this->database_path = database_path;
}

Database::~Database()
{
    this->save();
}

float Database::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
    {
        std::cerr << "Error: Vectors must be of the same size for cosine similarity" << std::endl;
        return 0.0f;
    }

    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); i++)
    {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f)
        return 0.0f; // Avoid division by zero

    return dot_product / (sqrt(norm_a) * sqrt(norm_b));
}

bool Database::equal_embeddings(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        return false;

    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            return false;

    return true;
}


void Database::add_entry(Entry& entry)
{
    for (const auto& e : this->entries)
    {
        if (equal_embeddings(e.embedding, entry.embedding))
        {
            std::cerr << "Error: Entry with the same embedding already exists in the database (ID: " << e.id << ")" << std::endl;
            return;
        }
    }

    entry.id = ++this->last_id;
    this->entries.push_back(entry);
}

void Database::save()
{
    // for simplicity I'm just saving the entries as a binary file here but in the future I can implement a more efficient storage solution (e.g. sqlite, leveldb) if needed
    std::ofstream ofs(this->database_path, std::ios::binary);

    if (!ofs)
    {
        std::cerr << "Error opening file for writing: " << this->database_path << std::endl;
        return;
    }

    size_t num_entries = this->entries.size();
    ofs.write((char*)&num_entries, sizeof(size_t));
    for (const auto& entry : this->entries)
    {
        ofs.write((char*)&entry.id, sizeof(uint32_t));
        ofs.write((char*)&entry.document_id, sizeof(uint32_t));

        size_t embedding_size = entry.embedding.size();
        ofs.write((char*)&embedding_size, sizeof(size_t));
        ofs.write((char*)entry.embedding.data(), embedding_size * sizeof(float));

        size_t title_size = entry.title.size();
        ofs.write((char*)&title_size, sizeof(size_t));
        ofs.write(entry.title.c_str(), title_size);

        size_t url_size = entry.url.size();
        ofs.write((char*)&url_size, sizeof(size_t));
        ofs.write(entry.url.c_str(), url_size);

        size_t value_size = entry.value.size();
        ofs.write((char*)&value_size, sizeof(size_t));
        ofs.write(entry.value.c_str(), value_size);
    }
}

void Database::load()
{
    std::ifstream ifs(this->database_path, std::ios::binary);

    if (!ifs)
    {
        std::cerr << "Error opening file for reading: " << this->database_path << std::endl;
        return;
    }

    size_t num_entries;
    ifs.read((char*)&num_entries, sizeof(size_t));
    this->entries.resize(num_entries);

    for (size_t i = 0; i < num_entries; i++)
    {
        Entry& entry = this->entries[i];
        ifs.read((char*)&entry.id, sizeof(uint32_t));
        ifs.read((char*)&entry.document_id, sizeof(uint32_t));

        size_t embedding_size;
        ifs.read((char*)&embedding_size, sizeof(size_t));
        entry.embedding.resize(embedding_size);
        ifs.read((char*)entry.embedding.data(), embedding_size * sizeof(float));

        size_t title_size;
        ifs.read((char*)&title_size, sizeof(size_t));
        entry.title.resize(title_size);
        ifs.read(entry.title.data(), title_size);

        size_t url_size;
        ifs.read((char*)&url_size, sizeof(size_t));
        entry.url.resize(url_size);
        ifs.read(entry.url.data(), url_size);

        size_t value_size;
        ifs.read((char*)&value_size, sizeof(size_t));
        entry.value.resize(value_size);
        ifs.read(entry.value.data(), value_size);

        if (entry.id > this->last_id)
            this->last_id = entry.id;
    }
}

std::vector<Entry> Database::search(const std::vector<float>& query_embedding, int top_k)
{
    std::vector<std::pair<float, size_t>> scores;
    scores.reserve(this->entries.size());

    for (size_t i = 0; i < this->entries.size(); i++)
    {
        float sim = cosine_similarity(query_embedding, this->entries[i].embedding);
        scores.emplace_back(sim, i);
    }

    int k = std::min(top_k, (int)scores.size());
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<Entry> results;
    results.reserve(k);
    for (int i = 0; i < k; i++)
        results.push_back(this->entries[scores[i].second]);

    return results;
}

std::vector<Entry> Database::list(int start, int end)
{
    std::vector<Entry> entries_list;
    entries_list.reserve(end - start);

    for (int i = start; i < end && i < (int)this->entries.size(); i++)
        entries_list.push_back(this->entries[i]);

    return entries_list;
}

Entry Database::get_entry_by_id(uint32_t id)
{
    for (const auto& entry : this->entries)
        if (entry.id == id)
            return entry;
    throw std::runtime_error("Entry not found with ID: " + std::to_string(id));
}

std::vector<Entry> Database::get_all_entries()
{
    return this->entries;
}