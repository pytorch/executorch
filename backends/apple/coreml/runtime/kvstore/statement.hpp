//
// database.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <string>
#include <system_error>
#include <sqlite3.h>

#include <types.hpp>

namespace executorchcoreml {
namespace sqlite {

/// The deleter for a sqlite statement, finalizes the statement at the time of deallocation.
struct StatementDeleter {
    inline void operator()(sqlite3_stmt* stmt) {
        if (stmt) {
            sqlite3_finalize(stmt);
        }
    }
};

/// A class representing a compiled sqlite statement.
class PreparedStatement {
public:
    virtual ~PreparedStatement() = default;
    
    explicit PreparedStatement(std::unique_ptr<sqlite3_stmt, StatementDeleter> preparedStatement) noexcept;
    
    PreparedStatement(PreparedStatement const&) = delete;
    PreparedStatement& operator=(PreparedStatement const&) = delete;
    
    /// Binds an int64_t value to a parameter at the specified index.
    ///
    /// @param index The column index.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind(size_t index, int64_t value, std::error_code& error) const noexcept;
    
    /// Binds an int64_t value to a parameter with the specified name.
    ///
    /// @param name The column name.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_name(const std::string& name, int64_t value, std::error_code& error) const noexcept;
    
    /// Binds a double value to a parameter at the specified index.
    ///
    /// @param index The column index.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind(size_t index, double value, std::error_code& error) const noexcept;
    
    /// Binds a double value to a parameter with the specified name.
    ///
    /// @param name The column name.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_name(const std::string& name, double value, std::error_code& error) const noexcept;
    
    /// Binds a string value to a parameter at the specified index.
    ///
    /// @param index The column index.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind(size_t index, UnOwnedString value, std::error_code& error) const noexcept;
    
    /// Binds a string value to a parameter at the specified index without copying.
    ///
    /// @param index The column index.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_no_copy(size_t index, UnOwnedString value, std::error_code& error) const noexcept;
    
    /// Binds a string value to a parameter with the specified name.
    ///
    /// @param name The column name.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_name(const std::string& name, UnOwnedString value, std::error_code& error) const noexcept;
    
    /// Binds a string value to a parameter with the specified name without copying.
    ///
    /// @param name The column name.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_name_no_copy(const std::string& name, UnOwnedString value, std::error_code& error) const noexcept;
    
    /// Binds a blob value to a parameter at the specified index.
    ///
    /// @param index The column index.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind(size_t index, const UnOwnedBlob& value, std::error_code& error) const noexcept;
    
    /// Binds a blob value to a parameter at the specified index without copying.
    ///
    /// @param index The column index.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_no_copy(size_t index, const UnOwnedBlob& value, std::error_code& error) const noexcept;
    
    /// Binds a blob value to a parameter with the specified name.
    ///
    /// @param name The column name.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_name(const std::string& name, const UnOwnedBlob& value, std::error_code& error) const noexcept;
    
    /// Binds a blob value to a parameter with the specified name without copying.
    ///
    /// @param name The column name.
    /// @param value   The column value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool bind_name_no_copy(const std::string& name, const UnOwnedBlob& value, std::error_code& error) const noexcept;
    
    /// Resets the statement. The statement can be used again after calling this method but the column values needs to be rebound.
    ///
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the value was bound otherwise `false`.
    bool reset(std::error_code& error) const noexcept;
    
    /// Returns the underlying sqlite statement.
    inline sqlite3_stmt *get_underlying_statement() const noexcept {
        return prepared_statement_.get();
    }
    
    /// Returns the column storage type.
    inline StorageType get_column_storage_type(size_t index) noexcept {
        return get_column_storage_types()[index];
    }
    
    /// Returns the column value at the specified index.
    ///
    /// @param index The column index.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The column value at the specified index.
    Value get_column_value(size_t index, std::error_code& error) noexcept;
    
    /// Returns the column value without copy at the specified index. It's the caller's responsibility to copy the value.
    ///
    /// @param index The column index.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval The column value at the specified index.
    UnOwnedValue get_column_value_no_copy(size_t index, std::error_code& error) noexcept;
    
    /// Returns the column storage types.
    const std::vector<StorageType>& get_column_storage_types() noexcept;
    
    /// Returns the column count.
    inline size_t get_column_count() const noexcept {
        return column_count_;
    }
    
    /// Returns the column names.
    inline const std::vector<std::string>& get_column_names() const noexcept {
        return column_names_;
    }
    
    /// Executes the statement.
    ///
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the statement executed successfully `false`.
    bool execute(std::error_code& error) const noexcept;
    
    /// Retrieves next row.
    ///
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if there is a next row to step to otherwise false.
    bool step(std::error_code& error) const noexcept;
    
private:
    size_t column_count_;
    std::vector<std::string> column_names_;
    std::vector<StorageType> column_storage_types_;
    std::unique_ptr<sqlite3_stmt, StatementDeleter> prepared_statement_;
};

} // namespace sqlite
} // namespace executorchcoreml
