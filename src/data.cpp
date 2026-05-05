#include "data.hpp"
#include <iostream>

DataReader::DataReader(const std::string& file_path) {
    this->file_path = file_path;
    this->pool = arrow::default_memory_pool();
}

DataReader::~DataReader() {
}

std::shared_ptr<arrow::Table> DataReader::read_data() {
    // 1. Open the file
    auto input_result = arrow::io::ReadableFile::Open(this->file_path);
    if (!input_result.ok()) {
        std::cerr << "Error opening file: " << input_result.status().ToString() << std::endl;
        return nullptr;
    }
    std::shared_ptr<arrow::io::RandomAccessFile> input = *input_result;

    // 2. Open the Parquet reader
    auto reader_result = parquet::arrow::OpenFile(input, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        std::cerr << "Error opening parquet reader: " << reader_result.status().ToString() << std::endl;
        return nullptr;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*reader_result);

    // 3. Read the full table
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = reader->ReadTable(&table);
    if (!status.ok()) {
        std::cerr << "Error reading table: " << status.ToString() << std::endl;
        return nullptr;
    }

    return table;
}

std::vector<WikipediaDataRow> DataReader::get_rows(int start, int end) {
    std::vector<WikipediaDataRow> rows;

    std::shared_ptr<arrow::Table> table = this->read_data();
    if (!table) {
        std::cerr << "Failed to read data" << std::endl;
        return rows;
    }

    for (int i = start; i < end && i < table->num_rows(); i++) {
        WikipediaDataRow row;
        auto id_col = table->GetColumnByName("id");
        auto text_col = table->GetColumnByName("text");
        auto title_col = table->GetColumnByName("title");
        auto url_col = table->GetColumnByName("url");

        auto id_array = std::static_pointer_cast<arrow::StringArray>(id_col->chunk(0));
        auto text_array = std::static_pointer_cast<arrow::StringArray>(text_col->chunk(0));
        auto title_array = std::static_pointer_cast<arrow::StringArray>(title_col->chunk(0));
        auto url_array = std::static_pointer_cast<arrow::StringArray>(url_col->chunk(0));

        row.id = (u_int32_t)std::stoul(
            id_array->GetString(i)
        );
        row.text = text_array->GetString(i);
        row.title = title_array->GetString(i);
        row.url = url_array->GetString(i);

        rows.push_back(row);
    }

    return rows;
}