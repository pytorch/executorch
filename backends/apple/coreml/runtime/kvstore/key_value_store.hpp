//
// key_value_store.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <functional>
#include <optional>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>

#include <database.hpp>
#include <types.hpp>

namespace executorchcoreml {
namespace sqlite {

/// A class to convert a type `T` from `sqlite::Value` and vice-versa.
///
/// This is used by the KeyValueStore to read and write values to and from the backing sqlite table.
///
template<typename T>
struct Converter {
    static constexpr StorageType storage_type = StorageType::Null;
    
    template<typename FROM>
    static sqlite::Value to_sqlite_value(FROM&& value);
    
    static T from_sqlite_value(const sqlite::UnOwnedValue& value);
};

/// Converter for `int64_t` type.
template<>
struct Converter<int64_t> {
    static constexpr StorageType storage_type = StorageType::Integer;
    
    static inline Value to_sqlite_value(int value) {
        return value;
    }
    
    static inline int64_t from_sqlite_value(const sqlite::UnOwnedValue& value) {
        return std::get<int64_t>(value);
    }
};

/// Converter for `int` type.
template<>
struct Converter<int> {
    static constexpr StorageType storage_type = StorageType::Integer;
    
    static inline Value to_sqlite_value(int value) {
        return static_cast<int>(value);
    }
    
    static  inline int from_sqlite_value(const sqlite::UnOwnedValue& value) {
        return static_cast<int>(std::get<int64_t>(value));
    }
};

/// Converter for `size_t` type.
template<>
struct Converter<size_t> {
    static constexpr StorageType storage_type = StorageType::Integer;
    
    static inline Value to_sqlite_value(size_t value) {
        return static_cast<int>(value);
    }
    
    static inline size_t from_sqlite_value(const sqlite::UnOwnedValue& value) {
        return static_cast<size_t>(std::get<int64_t>(value));
    }
};

/// Converter for `double` type.
template<>
struct Converter<double> {
    static constexpr StorageType storage_type = StorageType::Double;
    
    static inline Value to_sqlite_value(double value) {
        return value;
    }
    
    static inline int from_sqlite_value(const UnOwnedValue& value) {
        return std::get<double>(value);
    }
};

/// Converter for `std::string` type.
template<>
struct Converter<std::string> {
    static constexpr sqlite::StorageType storage_type = StorageType::Text;
    
    static inline sqlite::Value to_sqlite_value(const std::string& value) {
        return value;
    }
    
    static inline std::string from_sqlite_value(const UnOwnedValue& value) {
        return std::string(std::get<sqlite::UnOwnedString>(value).data);
    }
};

/// Represents the sort order.
enum class SortOrder: uint8_t {
    Ascending = 0,
    Descending
};

/// Represents a type-erased KeyValue store, the store is backed by a sqlite table.
class KeyValueStoreImpl {
public:
    /// Constructs a type-erased KeyValue store.
    ///
    /// The backing table is created with the key column type set to the `get_key_storage_type` and the
    /// value column type set to the `get_value_storage_type`.
    ///
    /// @param database The opened database.
    /// @param name   The name of the store.
    /// @param get_key_storage_type   The key storage type.
    /// @param get_value_storage_type  The value storage type.
    inline KeyValueStoreImpl(const std::shared_ptr<Database>& database,
                             const std::string& name,
                             StorageType get_key_storage_type,
                             StorageType get_value_storage_type) noexcept
    :name_(name),
    get_key_storage_type_(get_key_storage_type),
    get_value_storage_type_(get_value_storage_type),
    database_(std::move(database))
    {}
    
    KeyValueStoreImpl(KeyValueStoreImpl const&) noexcept = delete;
    KeyValueStoreImpl& operator=(KeyValueStoreImpl const&) noexcept = delete;
    
    /// Returns the name of the store.
    inline std::string_view name() const noexcept {
        return name_;
    }
    
    /// Returns the key storage type.
    inline StorageType get_key_storage_type() const noexcept {
        return get_key_storage_type_;
    }
    
    /// Returns the value storage type.
    inline StorageType get_value_storage_type() const noexcept {
        return get_value_storage_type_;
    }
    
    /// Returns the sqlite database.
    inline Database *database() const noexcept {
        return database_.get();
    }
    
    /// Returns the value for the specified key. If the key doesn't exists in the store or for some reason the operation failed
    /// then `nullopt` is returned.
    ///
    /// @param key The key for which the value is retrieved.
    /// @param fn   The function that will be invoked with the retrieved value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @param update_access_statistics   If it is `true` then the access statistics (access time and count) are updated otherwise not.
    /// @retval The associated value for the key.
    bool get(const Value& key,
             const std::function<void(const UnOwnedValue&)>& fn,
             std::error_code& error,
             bool update_access_statistics) noexcept;
    
    /// Returns `true` if the key exists in the store otherwise `false`.
    ///
    /// @param key The key.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the key exists in the store otherwise `false`.
    bool exists(const Value& key,
                std::error_code& error) noexcept;
    
    /// Sorts the keys by the access count and calls the `std::function` on each key value. The sort order
    /// is specified by the `order` parameter. The caller can stop the iteration by returning `false`
    /// from the lambda, to continue the iteration the caller must return `true`.
    ///
    /// @param fn The `std::function` that gets called for each key after its sorted.
    /// @param order The sort order.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    bool get_keys_sorted_by_access_count(const std::function<bool(const UnOwnedValue&)>& fn,
                                         SortOrder order,
                                         std::error_code& error) noexcept;
    
    /// Sorts the keys by the access time and calls the `std::function` on each key value. The sort order
    /// is specified by the `order` parameter. The caller can stop the iteration by returning `false`
    /// from the lambda, to continue the iteration the caller must return `true`.
    ///
    /// @param fn The `std::function` that gets called for each key after its sorted.
    /// @param order The sort order.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    bool get_keys_sorted_by_access_time(const std::function<bool(const UnOwnedValue&)>& fn,
                                        SortOrder order,
                                        std::error_code& error) noexcept;
    
    /// Stores a key and a value in the store, the old value is overwritten.
    ///
    /// @param key The key.
    /// @param value The value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    bool put(const Value& key, const Value& value, std::error_code& error) noexcept;
    
    /// Removes the specified key and the associated value from the store.
    ///
    /// @param key The key.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    bool remove(const Value& key, std::error_code& error) noexcept;
    
    /// Purges the store. The backing table is dropped and re-created.
    bool purge(std::error_code& error)  noexcept;
    
    /// Returns the size of the store.
    std::optional<size_t> size(std::error_code& error) noexcept;
    
    /// Initializes the store.
    ///
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    bool init(std::error_code& error) noexcept;
    
private:
    bool updateValueAccessCountAndTime(const Value& key,
                                       int64_t accessCount,
                                       std::error_code& error) noexcept;
    
    std::string name_;
    StorageType get_key_storage_type_;
    StorageType get_value_storage_type_;
    std::shared_ptr<Database> database_;
    std::atomic<int64_t> lastAccessTime_;
};

/// Represents a KeyValue store, the store is backed by a sqlite table.
template <typename Key, typename Value, typename ValueConverter = Converter<Value>, typename KeyConverter = Converter<Key>>
class KeyValueStore final {
public:
    template<typename T> using same_key = std::is_same<typename std::decay_t<T>, Key>;
    template<typename T> using same_value = std::is_same<typename std::decay_t<T>, Value>;
    
    virtual ~KeyValueStore() = default;
    
    KeyValueStore(KeyValueStore const&) noexcept = delete;
    KeyValueStore& operator=(KeyValueStore const&) noexcept = delete;
    
    inline KeyValueStore(std::unique_ptr<KeyValueStoreImpl> impl) noexcept
    :impl_(std::move(impl))
    {}
    
    /// Executes the provided lambda inside a transaction. The lambda must return `true` if the transaction is to
    /// be committed otherwise `false`.
    ///
    /// The transaction is only committed if the lambda returns `true` otherwise the transaction is rolled-back.
    ///
    /// @param fn The lambda that will  be executed inside a transaction.
    /// @param behavior   The transaction behavior.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the transaction is committed otherwise `false`.
    template<typename FN>
    bool transaction(FN&& fn, Database::TransactionBehavior behavior, std::error_code& error) noexcept {
        return impl_->database()->transaction([&fn](){
            return fn();
        }, behavior, error);
    }
    
    
    /// Returns the value for the specified key. If the key doesn't exists in the store or the operation failed
    /// then `nullopt` is returned.
    ///
    /// @param key The key for which the value is retrieved.
    /// @param error   On failure, error is populated with the failure reason.
    /// @param update_access_statistics   If it is `true` then the access statistics (access time and access count) are updated otherwise not.
    /// @retval The associated value for the key.
    template<typename T = Key>
    inline std::optional<Value> get(T&& key, std::error_code& error, bool update_access_statistics = true) noexcept {
        Value result;
        std::function<void(const UnOwnedValue&)> fn = [&result](const UnOwnedValue& value) {
            result = ValueConverter::from_sqlite_value(value);
        };
        
        if (!impl_->get(KeyConverter::to_sqlite_value(std::forward<T>(key)), fn, error, update_access_statistics)) {
            return std::nullopt;
        }
        
        return result;
    }
    
    /// Returns `true` if the key exists in the store otherwise `false`.
    ///
    /// @param key The key.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the key exists in the store otherwise `false`.
    template<typename T = Key>
    inline bool exists(T&& key, std::error_code& error) noexcept {
        return impl_->exists(KeyConverter::to_sqlite_value(std::forward<T>(key)), error);
    }
    
    /// Stores a key and its associated value in the store, the old value is overwritten.
    ///
    /// @param key The key.
    /// @param value The value.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    template<typename T = Key, typename U = Value>
    inline bool put(T&& key, U&& value, std::error_code& error) const noexcept {
        return impl_->put(KeyConverter::to_sqlite_value(std::forward<T>(key)),
                          ValueConverter::to_sqlite_value(std::forward<U>(value)),
                          error);
    }
    
    /// Sorts the keys by the access count and calls the lambda on each key value. The sort order
    /// is specified by the `order` parameter. The caller can stop the iteration by returning `false`
    /// from the lambda, to continue the iteration the caller must return `true`.
    ///
    /// @param fn The lambda that gets called for each key after its sorted.
    /// @param order The sort order.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    template<typename FN>
    bool get_keys_sorted_by_access_count(FN&& fn,
                                         SortOrder order,
                                         std::error_code& error) noexcept {
        std::function<bool(const UnOwnedValue&)> wrappedFn = [&fn](const UnOwnedValue& value) {
            return fn(KeyConverter::from_sqlite_value(value));
        };
        
        return impl_->get_keys_sorted_by_access_count(wrappedFn, order, error);
    }
    
    /// Sorts the keys by the access time and calls the lambda on each key value. The sort order
    /// is specified by the `order` parameter. The caller can stop the iteration by returning `false`
    /// from the lambda, to continue the iteration the caller must return `true`.
    ///
    /// @param fn The lambda that gets called for each key after its sorted.
    /// @param order The sort order.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    template<typename FN>
    bool get_keys_sorted_by_access_time(FN&& fn,
                                        SortOrder order,
                                        std::error_code& error) noexcept {
        std::function<bool(const UnOwnedValue&)> wrappedFn = [&fn](const UnOwnedValue& value) {
            return fn(KeyConverter::from_sqlite_value(value));
        };
        
        return impl_->get_keys_sorted_by_access_time(wrappedFn, order, error);
    }
    
    /// Removes the specified key and its associated value from the store.
    ///
    /// @param key The key.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval `true` if the operation succeeded otherwise `false`.
    template<typename T = Key>
    inline bool remove(T&& key, std::error_code& error) noexcept {
        return impl_->remove(Converter<Key>::to_sqlite_value(std::forward<T>(key)), error);
    }
    
    /// Returns the name of the store.
    inline std::string_view name() const noexcept {
        return impl_->name();
    }
    
    /// Returns the size of the store.
    inline std::optional<size_t> size(std::error_code& error) const noexcept {
        return impl_->size(error);
    }
    
    /// Purges the store. The backing table is dropped and re-created.
    inline bool purge(std::error_code& error) noexcept {
        return impl_->purge(error);
    }
    
    /// Creates a typed KeyValue store.
    ///
    /// The returned store's key type is `KeyType` and the value type is `ValueType`.  The store
    /// uses the `KeyConverter` type to convert a value of `KeyType` to a `sqlite::Value` and
    /// the `ValueConverter` to convert a value of `ValueType` to a `sqlite::Value`.
    ///
    /// @param database The sqlite database used for persisting.
    /// @param name  The name of the store.
    /// @param error   On failure, error is populated with the failure reason.
    /// @retval the `unique_ptr` to the created store or `nullptr` if the creation failed.
    static inline std::unique_ptr<KeyValueStore<Key, Value, ValueConverter, KeyConverter>>
    make(const std::shared_ptr<Database>& database, const std::string& name, std::error_code& error) noexcept {
        auto impl = std::make_unique<KeyValueStoreImpl>(database,
                                                        name,
                                                        KeyConverter::storage_type,
                                                        ValueConverter::storage_type);
        if (!impl->init(error)) {
            return nullptr;
        }
        
        return std::make_unique<KeyValueStore<Key, Value, ValueConverter, KeyConverter>>(std::move(impl));
    }
    
private:
    std::unique_ptr<KeyValueStoreImpl> impl_;
};

} // namespace sqlite
} // namespace executorchcoreml
