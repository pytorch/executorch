//
//  database.cpp
//  kvstore
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include <database.hpp>

#include <sqlite_error.hpp>

namespace {
using namespace executorchcoreml::sqlite;

sqlite3_stmt* getPreparedStatement(sqlite3* database, const std::string& statement, std::error_code& error) {
    sqlite3_stmt* handle = nullptr;
    const int status = sqlite3_prepare_v2(database, statement.c_str(), -1, &handle, nullptr);
    if (!process_sqlite_status(status, error)) {
        return nullptr;
    }

    return handle;
}

/// Returns the sqlite pragma from
std::string toString(Database::SynchronousMode mode) {
    switch (mode) {
        case Database::SynchronousMode::Full: {
            return "FULL";
        }
        case Database::SynchronousMode::Extra: {
            return "EXTRA";
        }
        case Database::SynchronousMode::Normal: {
            return "NORMAL";
        }
        case Database::SynchronousMode::Off: {
            return "OFF";
        }
    }
}

/// Returns the sqlite statement for a specified transaction behavior.
std::string getTransactionStatement(Database::TransactionBehavior behavior) {
    switch (behavior) {
        case Database::TransactionBehavior::Deferred: {
            return "BEGIN DEFERRED";
        }
        case Database::TransactionBehavior::Immediate: {
            return "BEGIN IMMEDIATE";
        }
        case Database::TransactionBehavior::Exclusive: {
            return "BEGIN EXCLUSIVE";
        }
    }
}
} // namespace

namespace executorchcoreml {
namespace sqlite {

int Database::OpenOptions::get_sqlite_flags() const noexcept {
    int flags = 0;
    if (is_read_only_option_enabled()) {
        flags |= SQLITE_OPEN_READONLY;
    }

    if (is_read_write_option_enabled()) {
        flags |= SQLITE_OPEN_READWRITE;
    }

    if (is_create_option_enabled()) {
        flags |= SQLITE_OPEN_CREATE;
    }

    if (is_memory_option_enabled()) {
        flags |= SQLITE_OPEN_MEMORY;
    }

    if (is_no_mutex_option_enabled()) {
        flags |= SQLITE_OPEN_NOMUTEX;
    }

    if (is_full_mutex_option_enabled()) {
        flags |= SQLITE_OPEN_FULLMUTEX;
    }

    if (is_shared_cache_option_enabled()) {
        flags |= SQLITE_OPEN_SHAREDCACHE;
    }

    if (is_shared_cache_option_enabled()) {
        flags |= SQLITE_OPEN_SHAREDCACHE;
    }

    if (is_uri_option_enabled()) {
        flags |= SQLITE_OPEN_URI;
    }

    return flags;
}

bool Database::open(OpenOptions options, SynchronousMode mode, int busy_timeout_ms, std::error_code& error) noexcept {
    sqlite3* handle = nullptr;
    const int status = sqlite3_open_v2(file_path_.c_str(), &handle, options.get_sqlite_flags(), nullptr);
    sqlite_database_.reset(handle);
    if (!process_sqlite_status(status, error)) {
        return false;
    }

    if (!set_busy_timeout(busy_timeout_ms, error)) {
        return false;
    }

    if (!execute("pragma journal_mode = WAL", error)) {
        return false;
    }

    if (!execute("pragma auto_vacuum = FULL", error)) {
        return false;
    }

    if (!execute("pragma synchronous = " + toString(mode), error)) {
        return false;
    }

    return true;
}

bool Database::is_open() const noexcept { return sqlite_database_ != nullptr; }

bool Database::table_exists(const std::string& tableName, std::error_code& error) const noexcept {
    auto statement = prepare_statement("SELECT COUNT(*) FROM sqlite_master WHERE TYPE='table' AND NAME=?", error);
    if (!statement) {
        return false;
    }

    if (!statement->bind(1, UnOwnedString(tableName), error)) {
        return false;
    }

    if (!statement->step(error)) {
        return false;
    }

    auto value = statement->get_column_value(0, error);
    if (error) {
        return false;
    }

    return (std::get<int64_t>(value) == 1);
}

bool Database::drop_table(const std::string& tableName, std::error_code& error) const noexcept {
    std::string statement = "DROP TABLE IF EXISTS " + tableName;
    return execute(statement, error);
}

int64_t Database::get_row_count(const std::string& tableName, std::error_code& error) const noexcept {
    auto statement = prepare_statement("SELECT COUNT(*) FROM " + tableName, error);
    if (!statement) {
        return -1;
    }

    if (!statement->step(error)) {
        return -1;
    }

    auto value = statement->get_column_value(0, error);
    return std::get<int64_t>(value);
}

bool Database::set_busy_timeout(int busy_timeout_ms, std::error_code& error) const noexcept {
    const int status = sqlite3_busy_timeout(get_underlying_database(), busy_timeout_ms);
    return process_sqlite_status(status, error);
}

bool Database::execute(const std::string& statements, std::error_code& error) const noexcept {
    const int status = sqlite3_exec(get_underlying_database(), statements.c_str(), nullptr, nullptr, nullptr);
    return process_sqlite_status(status, error);
}

int Database::get_updated_row_count() const noexcept { return sqlite3_changes(get_underlying_database()); }

std::string Database::get_last_error_message() const noexcept { return sqlite3_errmsg(get_underlying_database()); }

std::unique_ptr<PreparedStatement> Database::prepare_statement(const std::string& statement,
                                                               std::error_code& error) const noexcept {
    sqlite3_stmt* handle = getPreparedStatement(get_underlying_database(), statement, error);
    return std::make_unique<PreparedStatement>(std::unique_ptr<sqlite3_stmt, StatementDeleter>(handle));
}

int64_t Database::get_last_inserted_row_id() const noexcept {
    return sqlite3_last_insert_rowid(get_underlying_database());
}

std::error_code Database::get_last_error_code() const noexcept {
    int code = sqlite3_errcode(get_underlying_database());
    return static_cast<ErrorCode>(code);
}

std::error_code Database::get_last_extended_error_code() const noexcept {
    int code = sqlite3_extended_errcode(get_underlying_database());
    return static_cast<ErrorCode>(code);
}

bool Database::begin_transaction(TransactionBehavior behavior, std::error_code& error) const noexcept {
    return execute(getTransactionStatement(behavior), error);
}

bool Database::commit_transaction(std::error_code& error) const noexcept {
    return execute("COMMIT TRANSACTION", error);
}

bool Database::rollback_transaction(std::error_code& error) const noexcept {
    return execute("ROLLBACK TRANSACTION", error);
}

bool Database::transaction(const std::function<bool(void)>& fn,
                           TransactionBehavior behavior,
                           std::error_code& error) noexcept {
    if (!begin_transaction(behavior, error)) {
        return false;
    }

    bool status = fn();
    if (status) {
        return commit_transaction(error);
    } else {
        rollback_transaction(error);
        return false;
    }
}

std::shared_ptr<Database> Database::make_inmemory(SynchronousMode mode, int busy_timeout_ms, std::error_code& error) {
    auto database = std::make_shared<Database>(":memory:");
    OpenOptions options;
    options.set_read_write_option(true);
    if (database->open(options, mode, busy_timeout_ms, error)) {
        return database;
    }

    return nullptr;
}

std::shared_ptr<Database> Database::make(const std::string& filePath,
                                         OpenOptions options,
                                         SynchronousMode mode,
                                         int busy_timeout_ms,
                                         std::error_code& error) {
    auto database = std::make_shared<Database>(filePath);
    if (database->open(options, mode, busy_timeout_ms, error)) {
        return database;
    }

    return nullptr;
}

} // namespace sqlite
} // namespace executorchcoreml
