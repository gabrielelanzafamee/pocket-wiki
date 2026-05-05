/**
 * Parque Reader for wikipedia data parquet files
 * 
 */

#pragma once

#include <string>
#include <memory>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

typedef struct WikipediaDataRow
{
    u_int32_t id;
    std::string text;
    std::string title;
    std::string url;
} WikipediaDataRow;

class DataReader
{
public:
    DataReader(const std::string& file_path);
    ~DataReader();

    std::shared_ptr<arrow::Table> read_data();
    std::vector<WikipediaDataRow> get_rows(int start, int end);
private:
    std::string file_path;
    arrow::MemoryPool* pool;
    arrow::Table* table;
};