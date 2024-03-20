//
// database.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <bitset>
#include <functional>
#include <memory>
#include <string>
#include <system_error>

#include <sqlite3.h>

#include <statement.hpp>

namespace executorchcoreml {
namespace sqlite {

/// The deleter for a sqlite database, closes the database at the time of deallocation.
struct DatabaseDeleter {
    inline void operator()(sqlite3* handle) {
        if (handle) {
            sqlite3_close(handle);
        }
    }
};

/// Database represents a sqlite database.
class Database {
public:
    /// OpenOptions lists all the options that can be used to open a sqlite database.
    ///
    /// The caller is responsible for setting and passing a valid option when opening the database.
    class OpenOptions {
    public:
        /// Corresponds to `SQLITE_OPEN_READONLY` flag, when set the database will be opened in read-only mode.
        inline void set_read_only_option(bool enable) noexcept {
            flags_[0] = enable;
        }
        
        /// Returns `true` if read-only option is enabled otherwise `false`.
        inline bool is_read_only_option_enabled() const noexcept {
            return flags_[0];
        }
        
        /// Corresponds to `SQLITE_OPEN_READWRITE` flag, when set the database will be opened in read and write mode.
        inline void set_read_write_option(bool enable) noexcept {
            flags_[1] = enable;
        }
        
        /// Returns `true` if read and write option is enabled otherwise `false`.
        inline bool is_read_write_option_enabled() const noexcept {
            return flags_[1];
        }
        
        /// Corresponds to `SQLITE_OPEN_CREATE` flag, when set the database will be created if it does not exist.
        inline void set_create_option(bool enable) noexcept {
            flags_[2] = enable;
        }
        
        /// Returns `true` if create option is enabled otherwise `false`.
        inline bool is_create_option_enabled() const noexcept {
            return flags_[2];
        }
        
        /// Corresponds to `SQLITE_OPEN_MEMORY` flag, when set the database will be opened as in-memory database.
        inline void set_memory_option(bool enable) noexcept {
            flags_[3] = enable;
        }
        
        /// Returns `true` if memory option is enabled otherwise `false`.
        inline bool is_memory_option_enabled() const noexcept {
            return flags_[3];
        }
        
        /// Corresponds to `SQLITE_OPEN_NOMUTEX` flag, when set the database connection will use the "multi-thread" threading mode.
        inline void set_no_mutex_option(bool enable) noexcept {
            flags_[4] = enable;
        }
        
        /// Returns `true` if no mutex option is enabled otherwise `false`.
        inline bool is_no_mutex_option_enabled() const noexcept {
            return flags_[4];
        }
        
        /// Corresponds to `SQLITE_OPEN_FULLMUTEX` flag, when set the database connection will use the "serialized" threading mode.
        inline void set_full_mutex_option(bool enable) noexcept {
            flags_[5] = enable;
        }
        
        /// Returns `true` if full mutex option is enabled otherwise `false`.
        inline bool is_full_mutex_option_enabled() const noexcept {
            return flags_[5];
        }
        
        /// Corresponds to `SQLITE_OPEN_SHAREDCACHE` flag, when set the database will be opened with shared cache enabled.
        inline void set_shared_cache_option(bool enable) noexcept {
            flags_[6] = enable;
        }
        
        /// Returns `true` if shared cache option is enabled otherwise `false`.
        inline bool is_shared_cache_option_enabled() const noexcept {
            return flags_[6];
        }
        
        /// Corresponds to `SQLITE_OPEN_URI` flag, when set the filename can be interpreted as a URI.
        inline void set_uri_option(bool enable) noexcept {
            flags_[7] = enable;
        }
        
        /// Returns `true` if URI option is enabled otherwise `false`.
        inline bool is_uri_option_enabled() const noexcept {
            return flags_[7];
        }
        
        /// Returns the sqlite flags that can be used to open a sqlite database from the set options.
        int get_sqlite_flags() const noexcept;
        
    private:
        std::bitset<8> flags_;
    };
    
    /// Represents sqlite synchronous flag.
    enum class SynchronousMode: uint8_t {
        Extra = 0,
        Full,
        Normal,
        Off,
    };
    
    /// Represents the behavior of a sqlite transaction
    enum class TransactionBehavior: uint8_t {
        Deferred = 0,
        Immediate,
        Exclusive,
    };
    
    /// Constructs a database from a file path.
    Database(const std::string& filePath) noexcept
    :file_path_(filePath)
    {}
    
    Database(Database const&) = delete;
    Database& operator=(Database const&) = delete;
    
    /// Opens a database
    ///
    /// @param options The options for opening the database.
    /// @param mode   The synchronous mode for the database connection.
    /// @param busy_timeout_ms   The busy timeout interval in milliseconds.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the database is opened otherwise `false`.
    bool open(OpenOptions options,
              SynchronousMode mode,
              int busy_timeout_ms,
              std::error_code& error) noexcept;
    
    /// Returns `true` is the database is opened otherwise `false`.
    bool is_open() const noexcept;
    
    /// Check if a table exists with the specified name.
    ///
    /// @param tableName The table name.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the table exists otherwise `false`.
    bool table_exists(const std::string& tableName, std::error_code& error) const noexcept;
    
    /// Drops a table with the specified name.
    ///
    /// @param tableName The table name.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the table is dropped otherwise `false`.
    bool drop_table(const std::string& tableName, std::error_code& error) const noexcept;
    
    /// Returns the number of rows in the table.
    ///
    /// @param tableName The table name.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The number of rows in the table.
    int64_t get_row_count(const std::string& tableName, std::error_code& error) const noexcept;
    
    /// Executes the provided statements.
    ///
    /// @param statements The statements to execute.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the execution succeeded otherwise `false`.
    bool execute(const std::string& statements, std::error_code& error) const noexcept;
    
    /// Returns the number of rows updated by the last statement.
    int get_updated_row_count() const noexcept;
    
    /// Returns the error message of the last failed sqlite call.
    std::string get_last_error_message() const noexcept;
    
    /// Returns the error code of the last failed sqlite call.
    std::error_code get_last_error_code() const noexcept;
    
    /// Returns the extended error code of the last failed sqlite call.
    std::error_code get_last_extended_error_code() const noexcept;
    
    /// Returns the value of the last inserted row id.
    int64_t get_last_inserted_row_id() const noexcept;
    
    /// Returns the file path that was used to create the database.
    std::string_view file_path() const noexcept {
        return file_path_;
    }
    
    /// Compiles the provided statement and returns it.
    ///
    /// @param statement The statement to be compiled.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The compiled statement.
    std::unique_ptr<PreparedStatement>
    prepare_statement(const std::string& statement, std::error_code& error) const noexcept;
    
    /// Executes the provided function inside a transaction.
    ///
    /// The transaction is committed only if the provided function returns `true` otherwise the transaction is rolled-back.
    ///
    /// @param fn The function that will  be executed inside a transaction.
    /// @param behavior   The transaction behavior.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the transaction is committed otherwise `false`.
    bool transaction(const std::function<bool(void)>& fn,
                     TransactionBehavior behavior,
                     std::error_code& error) noexcept;
    
    /// Opens an in-memory database.
    ///
    /// @param mode The synchronous mode.
    /// @param busy_timeout_ms   The total busy timeout duration, in milliseconds.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The opened in-memory database.
    static std::shared_ptr<Database> make_inmemory(SynchronousMode mode,
                                                   int busy_timeout_ms,
                                                   std::error_code& error);
    
    /// Creates and opens a  database at the specified path.
    ///
    /// @param filePath The file path of the database.
    /// @param options   The open options.
    /// @param mode   The synchronous mode.
    /// @param busy_timeout_ms The total busy timeout duration, in milliseconds.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The opened database.
    static std::shared_ptr<Database> make(const std::string& filePath,
                                          OpenOptions options,
                                          SynchronousMode mode,
                                          int busy_timeout_ms,
                                          std::error_code& error);
    
private:
    /// Returns the internal sqlite database.
    inline sqlite3 *get_underlying_database() const noexcept {
        return sqlite_database_.get();
    }
    
    /// Registers an internal busy handler that keeps attempting to acquire a busy lock until the total specified time has passed.
    bool set_busy_timeout(int busy_timeout_ms, std::error_code& error) const noexcept;
    
    /// Begins an explicit transaction with the specified behavior.
    bool begin_transaction(TransactionBehavior behavior, std::error_code& error) const noexcept;
    
    /// Commits the last open transaction.
    bool commit_transaction(std::error_code& error) const noexcept;
    
    /// Rollbacks the last open transaction.
    bool rollback_transaction(std::error_code& error) const noexcept;
    
    std::string file_path_;
    std::unique_ptr<sqlite3, DatabaseDeleter> sqlite_database_;
};

} // namespace sqlite
} // namespace executorchcoreml
