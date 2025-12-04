//
//  database.cpp
//  kvstore
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include <database.hpp>

#include <iostream> // @nocommit
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
    return "FULL"; // safe default
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
    return "BEGIN DEFERRED"; // safe default
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

    if (is_uri_option_enabled()) {
        flags |= SQLITE_OPEN_URI;
    }

    return flags;
}

bool Database::open(std::error_code& error) const {
    sqlite3* tmp = nullptr;
    const int status = sqlite3_open_v2(file_path_.c_str(), &tmp, open_options_.get_sqlite_flags(), nullptr);

    if (!process_sqlite_status(status, error)) {
        if (tmp)
            sqlite3_close_v2(tmp); // ensure no leaked/half-open handle
        return false;
    }

    // Now we know it's good: install the connection
    sqlite_database_.reset(tmp);

    // Re-apply connection configuration
    if (!set_busy_timeout(busy_timeout_ms_, error)) {
        sqlite_database_.reset(nullptr);
        return false;
    }

    const bool ro = open_options_.is_read_only_option_enabled();
    const bool in_mem = (file_path_ == ":memory:");
    if (!ro && !in_mem) {
        if (!execute("pragma journal_mode = WAL", error)) {
            sqlite_database_.reset(nullptr);
            return false;
        }
        if (!execute("pragma auto_vacuum = FULL", error)) {
            sqlite_database_.reset(nullptr);
            return false;
        }
    }

    if (!execute(std::string("pragma synchronous = ") + toString(synchronous_mode_), error)) {
        sqlite_database_.reset(nullptr);
        return false;
    }

    error.clear(); // clear the error
    return true;
}

bool Database::is_open() const noexcept { return sqlite_database_ != nullptr; }

bool Database::table_exists(const std::string& tableName, std::error_code& error) const {
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
    if (auto p = std::get_if<int64_t>(&value)) {
        return (*p == 1);
    }
    error = make_error_code(std::errc::invalid_argument);
    return false;
}

bool Database::drop_table(const std::string& tableName, std::error_code& error) const {
    std::string statement = "DROP TABLE IF EXISTS " + tableName;
    return execute(statement, error);
}

int64_t Database::get_row_count(const std::string& tableName, std::error_code& error) const {
    auto statement = prepare_statement("SELECT COUNT(*) FROM " + tableName, error);
    if (!statement) {
        return -1;
    }

    if (!statement->step(error)) {
        return -1;
    }

    auto value = statement->get_column_value(0, error);
    if (auto p = std::get_if<int64_t>(&value))
        return *p;
    error = make_error_code(std::errc::invalid_argument);
    return -1;
}

bool Database::set_busy_timeout(int busy_timeout_ms, std::error_code& error) const noexcept {
    auto* db = get_underlying_database();
    if (!db) {
        error = std::make_error_code(std::errc::bad_file_descriptor);
        return false;
    }
    const int status = sqlite3_busy_timeout(db, busy_timeout_ms);
    return process_sqlite_status(status, error);
}

bool Database::execute(const std::string& statements, std::error_code& error) const {
    return execute_and_maybe_retry(statements, error, true);
}

bool Database::execute_and_maybe_retry(const std::string& statements, std::error_code& error, bool retry) const {

    std::cout << "Starting to execute statements: " << statements << std::endl;
    auto* db = get_underlying_database();
    if (!db) {
        error = std::make_error_code(std::errc::bad_file_descriptor);
        return false;
    }
    const int status = sqlite3_exec(db, statements.c_str(), nullptr, nullptr, nullptr);
    if (process_sqlite_status(status, error)) {
        std::cout << "Execute succeeded on first attempt" << std::endl;
        std::cout << "Returning true with error code: " << error << std::endl;
        error.clear(); // clear the error
        return true;
    }

    if (!retry) {
        std::cout << "Not retrying.  Returning false." << std::endl;
        return false;
    }

    std::cout << "Execute failed in SQL.  Trying to reopen." << std::endl;

    // Only attempt a reopen if we're not inside a transaction
    if (!in_transaction()) {
        std::cout << "Not in transaction.  Attempting to reopen." << std::endl;
        db = get_underlying_database();
        if (!db) {
            error = std::make_error_code(std::errc::bad_file_descriptor);
            return false;
        }
        const int err = sqlite3_errcode(db);
        const int xerr = sqlite3_extended_errcode(db);
        if (is_recoverable_connection_error(err, xerr)) {
            std::cout << "Recoverable error code" << std::endl;
            std::error_code reopen_ec;
            if (reopen(reopen_ec)) {
                std::cout << "Reopen succeeded.  Trying again." << std::endl;
                std::error_code retry_ec;
                db = get_underlying_database();
                if (!db) {
                    error = std::make_error_code(std::errc::bad_file_descriptor);
                    return false;
                }
                int retry_status = sqlite3_exec(db, statements.c_str(), nullptr, nullptr, nullptr);
                if (process_sqlite_status(retry_status, retry_ec)) {
                    std::cout << "Retry succeeded." << std::endl;
                    error.clear(); // clear the error
                    return true;
                }
                std::cout << "Retry failed." << std::endl;
                error = retry_ec; // surface the retry failure
            } else {
                std::cout << "Reopen failed." << std::endl;
                error = reopen_ec; // surface the reopen failure
            }
        }
    }
    std::cout << "Reopen failed.  Returning false." << std::endl;
    return false;
}

int Database::get_updated_row_count() const noexcept {
    auto db = get_underlying_database();
    if (!db) {
        return -1;
    }
    return sqlite3_changes(db);
}

std::string Database::get_last_error_message() const noexcept {
    auto db = get_underlying_database();
    if (!db) {
        return "";
    }
    return sqlite3_errmsg(db);
}

std::unique_ptr<PreparedStatement> Database::prepare_statement(const std::string& statement,
                                                               std::error_code& error) const noexcept {
    std::cout << "Preparing statement: " << statement << std::endl;
    auto db = get_underlying_database();
    if (!db) {
        error = std::make_error_code(std::errc::bad_file_descriptor);
        return nullptr;
    }
    sqlite3_stmt* handle = getPreparedStatement(db, statement, error);
    if (handle) {
        std::cout << "Returning prepared statement." << std::endl;
        error.clear();
        return std::make_unique<PreparedStatement>(std::unique_ptr<sqlite3_stmt, StatementDeleter>(handle));
    }
    std::cout << "Failed to prepare statement. Retrying.  The error code was: " << error << std::endl;
    std::cout << " xerr=" << sqlite3_extended_errcode(get_underlying_database()) << std::endl;
    std::cout << " msg=" << sqlite3_errmsg(get_underlying_database()) << "\n";
    if (!in_transaction()) {
        std::cout << "Not in transaction.  Attempting to reopen." << std::endl;
        db = get_underlying_database();
        if (!db) {
            error = std::make_error_code(std::errc::bad_file_descriptor);
            return nullptr;
        }
        const int err = sqlite3_errcode(db);
        const int xerr = sqlite3_extended_errcode(db);
        if (is_recoverable_connection_error(err, xerr)) {
            std::error_code reopen_ec;
            if (reopen(reopen_ec)) {
                // try again
                db = get_underlying_database();
                if (!db) {
                    error = std::make_error_code(std::errc::bad_file_descriptor);
                    return nullptr;
                }
                sqlite3_stmt* h2 = getPreparedStatement(db, statement, error);
                if (h2) {
                    return std::make_unique<PreparedStatement>(std::unique_ptr<sqlite3_stmt, StatementDeleter>(h2));
                }
                // else `error` is already set by getPreparedStatement
            } else {
                error = reopen_ec;
            }
        }
    }
    std::cout << "Returning nullptr." << std::endl;
    return nullptr;
}

int64_t Database::get_last_inserted_row_id() const noexcept {
    auto db = get_underlying_database();
    if (!db) {
        return -1;
    }
    return sqlite3_last_insert_rowid(db);
}

std::error_code Database::get_last_error_code() const noexcept {
    auto db = get_underlying_database();
    if (!db) {
        return std::make_error_code(std::errc::bad_file_descriptor);
    }
    int code = sqlite3_errcode(db);
    return make_error_code(static_cast<ErrorCode>(code));
}

std::error_code Database::get_last_extended_error_code() const noexcept {
    auto db = get_underlying_database();
    if (!db) {
        return std::make_error_code(std::errc::bad_file_descriptor);
    }
    int code = sqlite3_extended_errcode(db);
    return make_error_code(static_cast<ErrorCode>(code));
}

bool Database::begin_transaction(TransactionBehavior behavior, std::error_code& error) const {
    return execute(getTransactionStatement(behavior), error);
}

bool Database::commit_transaction(std::error_code& error) const { return execute("COMMIT TRANSACTION", error); }

bool Database::rollback_transaction(std::error_code& error) const {
    auto* db = get_underlying_database();
    if (!db) {
        error = std::make_error_code(std::errc::bad_file_descriptor);
        return false;
    }

    // If no txn is active, treat as success.
    if (sqlite3_get_autocommit(db) != 0) {
        error.clear();
        return true;
    }

    if (execute_and_maybe_retry("ROLLBACK TRANSACTION", error, false)) {
        error.clear();
        return true;
    }

    // Recovery: force-close (implicit rollback) and reopen
    // Optional: aggressively finalize any live statements
    db = get_underlying_database();
    if (!db) {
        error = std::make_error_code(std::errc::bad_file_descriptor);
        return false;
    }
    sqlite3_stmt* s = nullptr;
    while ((s = sqlite3_next_stmt(db, s)) != nullptr)
        sqlite3_finalize(s);

    sqlite_database_.reset(nullptr); // implicit rollback at close
    if (open(error)) {
        error.clear();
        return true;
    }

    return false; // 'error' already set
}

bool Database::transaction(const std::function<bool(void)>& fn, TransactionBehavior behavior, std::error_code& error) {
    std::cout << "Starting a new transaction." << std::endl;
    if (!begin_transaction(behavior, error)) {
        return false;
    }
    bool status = fn();
    std::cout << "Status of transaction: " << status << std::endl;
    if (status) {
        std::cout << "Committing transaction." << std::endl;
        return commit_transaction(error);
    } else {
        std::cout << "Rolling back transaction." << std::endl;
        rollback_transaction(error);
        return false;
    }
}

bool Database::in_transaction() const noexcept {
    auto* db = get_underlying_database();
    return db && sqlite3_get_autocommit(db) == 0;
}

bool Database::is_recoverable_connection_error(int err, int xerr) noexcept {
    const int primary = (err & 0xFF);

    // Extended codes that clearly indicate a stale/broken handle or path change
    switch (xerr) {
        case SQLITE_READONLY_DBMOVED:
        case SQLITE_IOERR_READ:
        case SQLITE_IOERR_WRITE:
        case SQLITE_IOERR_FSYNC:
        case SQLITE_IOERR_LOCK:
        case SQLITE_CANTOPEN_DIRTYWAL:
        case SQLITE_CANTOPEN_NOTEMPDIR:
            return true;
        case SQLITE_BUSY_RECOVERY:
        case SQLITE_BUSY_SNAPSHOT:
        case SQLITE_LOCKED:
            return false;
        default:
            break;
    }

    // Primary classes that are generally reopenable
    switch (primary) {
        case SQLITE_IOERR:
        case SQLITE_NOTADB:
        case SQLITE_CANTOPEN:
            return true;
        default:
            return false;
    }

    return false;
}

bool Database::reopen(std::error_code& error) const {
    std::cout << "Inside reopen." << std::endl;
    if (in_transaction()) {
        std::cout << "In transaction.  Returning false." << std::endl;
        error = std::make_error_code(std::errc::operation_in_progress);
        return false;
    }
    if (auto* db = get_underlying_database()) {
        // Refuse to reopen if any live statements exist.
        if (sqlite3_next_stmt(db, nullptr) != nullptr) {
            error = std::make_error_code(std::errc::device_or_resource_busy);
            return false;
        }
    }

    // Close current (best-effort)
    std::cout << "Closing current." << std::endl;
    sqlite_database_.reset(nullptr);
    std::cout << "Calling open." << std::endl;
    return open(error);
}

std::shared_ptr<Database> Database::make_inmemory(SynchronousMode mode, int busy_timeout_ms, std::error_code& error) {
    OpenOptions options;
    options.set_read_write_option(true);
    auto database = std::make_shared<Database>(":memory:", options, mode, busy_timeout_ms);
    if (database->open(error)) {
        return database;
    }

    return nullptr;
}

std::shared_ptr<Database> Database::make(const std::string& filePath,
                                         OpenOptions options,
                                         SynchronousMode mode,
                                         int busy_timeout_ms,
                                         std::error_code& error) {
    auto database = std::make_shared<Database>(filePath, options, mode, busy_timeout_ms);
    if (database->open(error)) {
        return database;
    }

    return nullptr;
}

} // namespace sqlite
} // namespace executorchcoreml
