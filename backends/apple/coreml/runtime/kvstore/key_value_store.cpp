//
// KeyValueStore.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "key_value_store.hpp"

#include <iostream>
#include <sstream>

namespace {
using namespace executorchcoreml::sqlite;

constexpr std::string_view kAccessCountColumnName = "ENTRY_ACCESS_COUNT";
constexpr std::string_view kAccessTimeColumnName = "ENTRY_ACCESS_TIME";

std::string to_string(StorageType storage_type) {
    switch (storage_type) {
        case StorageType::Text: {
            return "TEXT";
        }
        case StorageType::Integer: {
            return "INTEGER";
        }
        case StorageType::Double: {
            return "REAL";
        }
        case StorageType::Blob: {
            return "BLOB";
        }
        case StorageType::Null: {
            return "NULL";
        }
    }
}

std::string
get_create_store_statement(std::string_view store_name, StorageType key_storage_type, StorageType value_storage_type) {
    std::stringstream ss;
    ss << "CREATE TABLE IF NOT EXISTS ";
    ss << store_name << " ";
    ss << "(";
    ss << "ENTRY_KEY " << to_string(key_storage_type) << " PRIMARY KEY UNIQUE, ";
    ss << "ENTRY_VALUE " << to_string(value_storage_type) << ", ";
    ss << "ENTRY_ACCESS_COUNT INTEGER NOT NULL DEFAULT 0, ";
    ss << "ENTRY_ACCESS_TIME  INTEGER NOT NULL DEFAULT 0";
    ss << ")";

    return ss.str();
}

std::string get_create_index_statement(std::string_view store_name, std::string_view column_name) {
    std::stringstream ss;
    ss << "CREATE INDEX IF NOT EXISTS " << store_name << "_" << column_name << "_INDEX" << " ON " << store_name << "("
       << column_name << ")";

    return ss.str();
}

std::string get_insert_or_replace_statement(std::string_view store_name) {
    std::stringstream ss;
    ss << "INSERT OR REPLACE INTO " << store_name
       << "(ENTRY_KEY, ENTRY_VALUE, ENTRY_ACCESS_COUNT, ENTRY_ACCESS_TIME) VALUES (?, ?, ?, ?)";

    return ss.str();
}

std::string get_remove_statement(std::string_view store_name) {
    std::stringstream ss;
    ss << "DELETE FROM " << store_name << " WHERE ENTRY_KEY = ?";

    return ss.str();
}

std::string getQueryStatement(std::string_view store_name) {
    std::stringstream ss;
    ss << "SELECT ENTRY_VALUE, ENTRY_ACCESS_COUNT FROM " << store_name << " WHERE ENTRY_KEY = ?";

    return ss.str();
}

std::string get_key_count_statement(std::string_view store_name) {
    std::stringstream ss;
    ss << "SELECT COUNT(*) FROM " << store_name << " WHERE ENTRY_KEY = ?";

    return ss.str();
}

std::string get_update_entry_access_statement(std::string_view store_name) {
    std::stringstream ss;
    ss << "UPDATE " << store_name
       << " SET ENTRY_ACCESS_COUNT = ENTRY_ACCESS_COUNT + 1, ENTRY_ACCESS_TIME = ? WHERE ENTRY_KEY = ?";
    return ss.str();
}

static std::string get_exists_statement(std::string_view store) {
    std::stringstream ss;
    ss << "SELECT 1 FROM " << store << " WHERE ENTRY_KEY = ? LIMIT 1";
    return ss.str();
}

std::string to_string(SortOrder order) {
    switch (order) {
        case SortOrder::Ascending: {
            return "ASC";
        }
        case SortOrder::Descending: {
            return "DESC";
        }
    }
}

std::string
get_keys_sorted_by_column_statement(std::string_view storeName, std::string_view columnName, SortOrder order) {
    std::stringstream ss;
    ss << "SELECT ENTRY_KEY, ENTRY_ACCESS_COUNT, ENTRY_ACCESS_TIME FROM " << storeName << " ORDER BY " << columnName
       << " ";
    ss << to_string(order);

    return ss.str();
}

bool bind_value(
    PreparedStatement* statement, StorageType type, const Value& value, size_t index, std::error_code& error) {
    switch (type) {
        case StorageType::Text: {
            return statement->bind(index, std::get<std::string>(value), error);
        }
        case StorageType::Integer: {
            return statement->bind(index, std::get<int64_t>(value), error);
        }
        case StorageType::Double: {
            return statement->bind(index, std::get<double>(value), error);
        }
        case StorageType::Blob: {
            return statement->bind(index, std::get<Blob>(value).toUnOwned(), error);
        }
        default: {
            return false;
        }
    }
}

bool execute(Database* database,
             const std::string& query,
             size_t columnIndex,
             const std::function<bool(const UnOwnedValue&)>& fn,
             std::error_code& error) {
    auto statement = database->prepare_statement(query, error);
    if (!statement) {
        return false;
    }

    while (statement->step(error)) {
        auto columnValue = statement->get_column_value_no_copy(columnIndex, error);
        if (error || !fn(columnValue)) {
            break;
        }
    }

    return !(error.operator bool());
}

int64_t get_last_access_time(Database* database, std::string_view storeName, std::error_code& error) {
    int64_t latestAccessTime = 0;
    auto statement = get_keys_sorted_by_column_statement(storeName, kAccessTimeColumnName, SortOrder::Descending);

    bool ok = execute(
        database,
        statement,
        /*columnIndex=*/2,
        [&](const UnOwnedValue& v) {
            latestAccessTime = std::get<int64_t>(v);
            return false; // stop after first row
        },
        error);

    return (ok && !error) ? latestAccessTime : 0;
}

} // namespace

namespace executorchcoreml {
namespace sqlite {

bool KeyValueStoreImpl::init(std::error_code& error) noexcept { return ensure_schema_exists(error); }

bool KeyValueStoreImpl::exists(const Value& key, std::error_code& error) noexcept {
    error = {}; // ensure "miss" can be distinguished from "error"
    if (!ensure_schema_exists(error))
        return false;
    auto q = database_->prepare_statement(get_exists_statement(name_), error);
    if (!q)
        return false;
    if (!bind_value(q.get(), get_key_storage_type(), key, 1, error))
        return false;
    bool has_row = q->step(error);
    if (error)
        return false;
    return has_row;
}

bool KeyValueStoreImpl::get(const Value& key,
                            const std::function<void(const UnOwnedValue&)>& fn,
                            std::error_code& error,
                            bool updateAccessStatistics) noexcept {
    error = {}; // ensure "miss" can be distinguished from "error"

    if (!ensure_schema_exists(error))
        return false;

    auto query = database_->prepare_statement(getQueryStatement(name_), error);
    if (!query) {
        return false;
    }

    if (!bind_value(query.get(), get_key_storage_type(), key, 1, error)) {
        return false;
    }

    bool has_row = query->step(error);
    if (error)
        return false;
    if (!has_row) {
        error = {};
        return false;
    }

    auto value = query->get_column_value_no_copy(0, error);

    if (error)
        return false;

    fn(value);

    if (updateAccessStatistics) {
        auto update = database_->prepare_statement(get_update_entry_access_statement(name_), error);
        if (!update)
            return false;

        auto next = lastAccessTime_.load(std::memory_order_acquire) + 1;
        if (!bind_value(update.get(), StorageType::Integer, next, 1, error))
            return false;
        if (!bind_value(update.get(), get_key_storage_type(), key, 2, error))
            return false;
        bool ok = update->execute(error);
        if (ok && !error) {
            lastAccessTime_.store(next, std::memory_order_release);
        }
        return ok && !error;
    }

    return true;
}

bool KeyValueStoreImpl::put(const Value& key, const Value& value, std::error_code& error) noexcept {
    error = {}; // clear error

    if (!ensure_schema_exists(error))
        return false;

    auto statement = database_->prepare_statement(get_insert_or_replace_statement(name_), error);
    if (!statement) {
        return false;
    }

    if (!bind_value(statement.get(), get_key_storage_type(), key, 1, error)) {
        return false;
    }

    if (!bind_value(statement.get(), get_value_storage_type(), value, 2, error)) {
        return false;
    }

    if (!bind_value(statement.get(), StorageType::Integer, int64_t(1), 3, error)) {
        return false;
    }

    auto next = lastAccessTime_.load(std::memory_order_acquire) + 1;
    if (!bind_value(statement.get(), StorageType::Integer, next, 4, error)) {
        return false;
    }
    bool ok = statement->execute(error);
    if (ok && !error) {
        lastAccessTime_.store(next, std::memory_order_release);
    }
    return ok && !error;
}

bool KeyValueStoreImpl::ensure_schema_exists(std::error_code& error) noexcept {
    error = {};
    if (!database_->execute(get_create_store_statement(name_, get_key_storage_type(), get_value_storage_type()), error))
        return false;
    if (!database_->execute(get_create_index_statement(name_, kAccessCountColumnName), error))
        return false;
    if (!database_->execute(get_create_index_statement(name_, kAccessTimeColumnName), error))
        return false;

    // Always recompute (cheap with the index).
    auto t = get_last_access_time(database_.get(), name_, error);
    if (error)
        return false;
    lastAccessTime_.store(t, std::memory_order_seq_cst);
    return true;
}

bool KeyValueStoreImpl::remove(const Value& key, std::error_code& error) noexcept {
    error = {}; // clear error
    if (!ensure_schema_exists(error))
        return false;
    auto statement = database_->prepare_statement(get_remove_statement(name_), error);
    if (!statement)
        return false;
    if (!bind_value(statement.get(), get_key_storage_type(), key, 1, error)) {
        return false;
    }

    return statement->execute(error);
}

bool KeyValueStoreImpl::get_keys_sorted_by_access_count(const std::function<bool(const UnOwnedValue&)>& fn,
                                                        SortOrder order,
                                                        std::error_code& error) noexcept {
    error = {}; // clear error
    if (!ensure_schema_exists(error))
        return false;
    auto statement = get_keys_sorted_by_column_statement(name(), kAccessCountColumnName, order);
    return execute(database_.get(), statement, 0, fn, error);
}

bool KeyValueStoreImpl::get_keys_sorted_by_access_time(const std::function<bool(const UnOwnedValue&)>& fn,
                                                       SortOrder order,
                                                       std::error_code& error) noexcept {
    error = {}; // clear error
    if (!ensure_schema_exists(error))
        return false;
    auto statement = get_keys_sorted_by_column_statement(name(), kAccessTimeColumnName, order);
    return execute(database_.get(), statement, 0, fn, error);
}

std::optional<size_t> KeyValueStoreImpl::size(std::error_code& error) noexcept {
    error = {}; // clear error
    if (!ensure_schema_exists(error))
        return std::nullopt;
    int64_t count = database_->get_row_count(name_, error);
    return count < 0 ? std::nullopt : std::optional<size_t>(count);
}

bool KeyValueStoreImpl::purge(std::error_code& error) noexcept {
    error = {}; // clear error
    if (!database_->drop_table(name_, error)) {
        return false;
    }
    return ensure_schema_exists(error);
}

} // namespace sqlite
} // namespace executorchcoreml
