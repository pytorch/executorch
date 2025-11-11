//
// Statement.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "statement.hpp"

#include <string_view>

#include <sqlite_error.hpp>

namespace {

using namespace executorchcoreml::sqlite;

size_t get_column_count(sqlite3_stmt* stmt) { return static_cast<size_t>(sqlite3_column_count(stmt)); }

std::vector<std::string> get_column_names(sqlite3_stmt* stmt, size_t column_count) {
    std::vector<std::string> result;
    result.reserve(static_cast<size_t>(column_count));
    for (int i = 0; i < column_count; i++) {
        const char* name = sqlite3_column_name(stmt, i);
        if (!name) {
            return {};
        }
        result.emplace_back(name);
    }

    return result;
}

int get_parameter_index(sqlite3_stmt* stmt, const std::string& name) {
    return sqlite3_bind_parameter_index(stmt, name.c_str());
}

bool bind_unowned_string(sqlite3_stmt* stmt, size_t index, UnOwnedString value, bool copy, std::error_code& error) {
    if (error) {
        return false;
    }
    auto destructor = copy ? SQLITE_TRANSIENT : SQLITE_STATIC;
    const int status =
        sqlite3_bind_text(stmt, static_cast<int>(index), value.data, static_cast<int>(value.size), destructor);
    return process_sqlite_status(status, error);
}

bool bind_blob(sqlite3_stmt* stmt, size_t index, const UnOwnedBlob& value, bool copy, std::error_code& error) {
    if (error) {
        return false;
    }
    auto destructor = copy ? SQLITE_TRANSIENT : SQLITE_STATIC;
    const int status =
        sqlite3_bind_blob(stmt, static_cast<int>(index), value.data, static_cast<int>(value.size), destructor);
    return process_sqlite_status(status, error);
}

StorageType get_column_storage_type(sqlite3_stmt* stmt, int index) {
    switch (sqlite3_column_type(stmt, index)) {
        case SQLITE_INTEGER: {
            return StorageType::Integer;
        }
        case SQLITE_FLOAT: {
            return StorageType::Double;
        }
        case SQLITE_TEXT: {
            return StorageType::Text;
        }
        case SQLITE_BLOB: {
            return StorageType::Blob;
        }
        case SQLITE_NULL: {
            return StorageType::Null;
        }
        default: {
            return StorageType::Null;
        }
    }
}

std::vector<StorageType> get_column_storage_types(sqlite3_stmt* stmt, size_t columnCount) {
    std::vector<StorageType> result;
    result.reserve(static_cast<size_t>(columnCount));
    for (int i = 0; i < columnCount; i++) {
        result.emplace_back(get_column_storage_type(stmt, i));
    }

    return result;
}

int64_t get_int64_value(sqlite3_stmt* stmt, size_t index) {
    return sqlite3_column_int64(stmt, static_cast<int>(index));
}

int64_t get_double_value(sqlite3_stmt* stmt, size_t index) {
    return sqlite3_column_double(stmt, static_cast<int>(index));
}

std::string get_string_value(sqlite3_stmt* stmt, size_t index) {
    auto data = static_cast<const char*>(sqlite3_column_blob(stmt, static_cast<int>(index)));
    return std::string(data, sqlite3_column_bytes(stmt, static_cast<int>(index)));
}

UnOwnedString get_unowned_string_value(sqlite3_stmt* stmt, size_t index) {
    auto data = static_cast<const char*>(sqlite3_column_blob(stmt, static_cast<int>(index)));
    return UnOwnedString(data, sqlite3_column_bytes(stmt, static_cast<int>(index)));
}

std::pair<const void*, size_t> get_stored_blob_value(sqlite3_stmt* stmt, size_t index) {
    const void* data = sqlite3_column_blob(stmt, static_cast<int>(index));
    int n = sqlite3_column_bytes(stmt, static_cast<int>(index));
    return { data, static_cast<size_t>(n) };
}

Blob get_blob_value(sqlite3_stmt* stmt, size_t index) {
    const auto& pair = get_stored_blob_value(stmt, index);
    return Blob(pair.first, pair.second);
}

UnOwnedBlob get_unowned_blob_value(sqlite3_stmt* stmt, size_t index) {
    const auto& pair = get_stored_blob_value(stmt, index);
    return UnOwnedBlob(pair.first, pair.second);
}
} // namespace

namespace executorchcoreml {
namespace sqlite {

PreparedStatement::PreparedStatement(std::unique_ptr<sqlite3_stmt, StatementDeleter> prepared_statement) noexcept
    : column_count_(::get_column_count(prepared_statement.get())),
      column_names_(::get_column_names(prepared_statement.get(), column_count_)),
      prepared_statement_(std::move(prepared_statement)) { }

bool PreparedStatement::bind(size_t index, int64_t value, std::error_code& error) const noexcept {
    if (error) {
        return false;
    }
    const int status = sqlite3_bind_int64(get_underlying_statement(), static_cast<int>(index), value);
    return process_sqlite_status(status, error);
}

bool PreparedStatement::bind_name(const std::string& name, int64_t value, std::error_code& error) const noexcept {
    return bind(get_parameter_index(get_underlying_statement(), name), value, error);
}

bool PreparedStatement::bind(size_t index, double value, std::error_code& error) const noexcept {
    if (error) {
        return false;
    }
    const int status = sqlite3_bind_double(get_underlying_statement(), static_cast<int>(index), value);
    return process_sqlite_status(status, error);
}

bool PreparedStatement::bind(size_t index, UnOwnedString value, std::error_code& error) const noexcept {
    return bind_unowned_string(get_underlying_statement(), index, value, true, error);
}

bool PreparedStatement::bind_no_copy(size_t index, UnOwnedString value, std::error_code& error) const noexcept {
    return bind_unowned_string(get_underlying_statement(), index, value, false, error);
}

bool PreparedStatement::bind_name(const std::string& name, UnOwnedString value, std::error_code& error) const noexcept {
    size_t index = get_parameter_index(get_underlying_statement(), name);
    return bind_unowned_string(get_underlying_statement(), index, value, true, error);
}

bool PreparedStatement::bind_name_no_copy(const std::string& name,
                                          UnOwnedString value,
                                          std::error_code& error) const noexcept {
    size_t index = get_parameter_index(get_underlying_statement(), name);
    return bind_unowned_string(get_underlying_statement(), index, value, false, error);
}

bool PreparedStatement::bind(size_t index, const UnOwnedBlob& value, std::error_code& error) const noexcept {
    return bind_blob(get_underlying_statement(), index, value, true, error);
}

bool PreparedStatement::bind_name(const std::string& name,
                                  const UnOwnedBlob& value,
                                  std::error_code& error) const noexcept {
    size_t index = get_parameter_index(get_underlying_statement(), name);
    return bind_blob(get_underlying_statement(), index, value, true, error);
}

bool PreparedStatement::bind_no_copy(size_t index, const UnOwnedBlob& value, std::error_code& error) const noexcept {
    return bind_blob(get_underlying_statement(), index, value, false, error);
}

bool PreparedStatement::bind_name_no_copy(const std::string& name,
                                          const UnOwnedBlob& value,
                                          std::error_code& error) const noexcept {
    size_t index = get_parameter_index(get_underlying_statement(), name);
    return bind_blob(get_underlying_statement(), index, value, false, error);
}

bool PreparedStatement::reset(std::error_code& error) const noexcept {
    if (error) {
        return false;
    }
    const int status = sqlite3_reset(get_underlying_statement());
    return process_sqlite_status(status, error);
}

bool PreparedStatement::step(std::error_code& error) const noexcept {
    if (error) {
        return false;
    }
    const int status = sqlite3_step(get_underlying_statement());
    if (status == SQLITE_ROW) {
        return true;
    } else if (status == SQLITE_DONE) {
        return false;
    } else {
        return process_sqlite_status(status, error);
    }
}

const std::vector<StorageType>& PreparedStatement::get_column_storage_types() noexcept {
    if (column_storage_types_.empty()) {
        column_storage_types_ = ::get_column_storage_types(get_underlying_statement(), get_column_count());
    }
    return column_storage_types_;
}

Value PreparedStatement::get_column_value(size_t index, std::error_code& error) noexcept {
    if (error) {
        return Null();
    }
    switch (get_column_storage_type(index)) {
        case StorageType::Integer: {
            return get_int64_value(get_underlying_statement(), index);
        }
        case StorageType::Double: {
            return get_double_value(get_underlying_statement(), index);
        }
        case StorageType::Text: {
            return get_string_value(get_underlying_statement(), index);
        }
        case StorageType::Blob: {
            return get_blob_value(get_underlying_statement(), index);
        }
        case StorageType::Null: {
            return Null();
        }
    }
}

UnOwnedValue PreparedStatement::get_column_value_no_copy(size_t index, std::error_code& error) noexcept {
    if (error) {
        return Null();
    }
    switch (get_column_storage_type(index)) {
        case StorageType::Integer: {
            return get_int64_value(get_underlying_statement(), index);
        }
        case StorageType::Double: {
            return get_double_value(get_underlying_statement(), index);
        }
        case StorageType::Text: {
            return get_unowned_string_value(get_underlying_statement(), index);
        }
        case StorageType::Blob: {
            return get_unowned_blob_value(get_underlying_statement(), index);
        }
        case StorageType::Null: {
            return Null();
        }
    }
}

bool PreparedStatement::execute(std::error_code& error) const noexcept {
    if (error) {
        return false;
    }
    const int status = sqlite3_step(get_underlying_statement());
    if (status == SQLITE_DONE) {
        return true;
    } else {
        return process_sqlite_status(status, error);
    }
}

} // namespace sqlite
} // namespace executorchcoreml
