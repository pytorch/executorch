//
// KeyValueStoreTests.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import <json_key_value_store.hpp>
#import <nlohmann/json.hpp>

namespace {
using json = nlohmann::json;

struct Entry;

void to_json(json& j, const Entry& entry) noexcept;

void from_json(const json& j, Entry& entry) noexcept;

struct Entry {
    inline Entry(std::string identifier, size_t count) noexcept
    :identifier(std::move(identifier)), count(count)
    {}

    inline Entry() noexcept
    :identifier(""), count(0)
    {}

    inline std::string to_json_string() const noexcept {
        json j;
        to_json(j, *this);
        std::stringstream ss;
        ss << j;
        return ss.str();
    }

    inline void from_json_string(const std::string& json_string) noexcept {
        auto j = json::parse(json_string);
        from_json(j, *this);
    }

    std::string identifier;
    size_t count;
};

void to_json(json& j, const Entry& entry) noexcept {
    j = json{
        {"identifier", entry.identifier},
        {"count", entry.count}
    };
}

void from_json(const json& j, Entry& entry) noexcept {
    j.at("identifier").get_to(entry.identifier);
    j.at("count").get_to(entry.count);
}
}

@interface KeyValueStoreTests : XCTestCase

@end

@implementation KeyValueStoreTests

using namespace executorchcoreml::sqlite;

- (void)testKVStorePut {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<int, double>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(1, 2.0, error));
    XCTAssertTrue(store->size(error) == 1);
    XCTAssertTrue(store->put(1, 3.0, error));
    XCTAssertTrue(store->size(error) == 1);
}

- (void)testKVStoreGet {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<std::string, int>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(std::string("1"), 1, error));
    XCTAssertTrue(store->size(error) == 1);
    XCTAssertTrue(store->get(std::string("1"), error).value() == 1);
    XCTAssertFalse(store->get(std::string("2"), error).has_value());
}

- (void)testKVStoreRemove {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<std::string, int>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(std::string("1"), 1, error));
    XCTAssertTrue(store->size(error) == 1);
    XCTAssertTrue(store->remove(std::string("1"), error));
    XCTAssertTrue(store->size(error) == 0);
}

- (void)testKVStoreExists {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<std::string, int>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(std::string("1"), 1, error));
    XCTAssertTrue(store->exists(std::string("1"), error));
    XCTAssertFalse(store->exists(std::string("2"), error));
}

- (void)testJSONKeyValueStore {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = JSONKeyValueStore<int, Entry>::make(std::move(database), "test", error);

    XCTAssertTrue(store->put(1, Entry("1", 1), error));
    auto entry1 = store->get(1, error);
    XCTAssertTrue(entry1.value().count == 1);
    XCTAssertTrue(entry1.value().identifier == "1");

    XCTAssertTrue(store->put(2, Entry("2", 2), error));
    auto entry2 = store->get(2, error);
    XCTAssertTrue(entry2.value().count == 2);
    XCTAssertTrue(entry2.value().identifier == "2");
}

- (void)testKVStoreTransactionCommit {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<int, std::string>::make(std::move(database), "test", error);
    XCTAssertTrue(store->transaction([store = store.get(), &error]() {
        XCTAssertTrue(store->put(1, std::string("abc"), error));
        XCTAssertTrue(store->put(2, std::string("def"), error));
        XCTAssertTrue(store->get(2, error).value() == "def");
        XCTAssertTrue(store->size(error) == 2);
        // Commit the transaction.
        return true;
    }, Database::TransactionBehavior::Immediate, error));

    XCTAssertTrue(store->size(error) == 2);
}

- (void)testKVStoreTransactionRollback {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<int, std::string>::make(std::move(database), "test", error);
    XCTAssertFalse(store->transaction([store = store.get(), &error]() {
        XCTAssertTrue(store->put(1, std::string("abc"), error));
        XCTAssertTrue(store->put(2, std::string("def"), error));
        XCTAssertTrue(store->get(2, error).value() == "def");
        XCTAssertTrue(store->size(error) == 2);
        // Rollback the transaction.
        return false;
    }, Database::TransactionBehavior::Immediate, error));

    XCTAssertTrue(store->size(error) == 0);
}

- (void)testKVStoreGetKeysSortedByAccessTime {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<int, std::string>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(1, std::string("1"), error));
    XCTAssertTrue(store->put(2, std::string("2"), error));
    XCTAssertTrue(store->put(3, std::string("3"), error));
    XCTAssertTrue(store->get(1, error).has_value());
    XCTAssertTrue(store->get(2, error).has_value());
    XCTAssertTrue(store->get(3, error).has_value());
    {
        std::vector<int> keys;
        XCTAssertTrue(store->get_keys_sorted_by_access_time([&keys](int key) {
            keys.push_back(key);
            return true;
        }, SortOrder::Ascending, error));
        // 1 is accessed first then 2 and then 3
        XCTAssertTrue(keys == (std::vector<int>{1, 2, 3}));
    }

    {
        std::vector<int> keys;
        XCTAssertTrue(store->get_keys_sorted_by_access_time([&keys](int key) {
            keys.push_back(key);
            return true;
        }, SortOrder::Descending, error));
        // 1 is accessed first then 2 and then 3
        XCTAssertTrue(keys == (std::vector<int>{3, 2, 1}));
    }
}

- (void)testKVStoreGetKeysSortedByAccessCount {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<int, std::string>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(1, std::string("1"), error));
    XCTAssertTrue(store->put(2, std::string("2"), error));
    XCTAssertTrue(store->put(3, std::string("3"), error));
    // 1 is accessed 3 times.
    XCTAssertTrue(store->get(1, error).has_value());
    XCTAssertTrue(store->get(1, error).has_value());
    XCTAssertTrue(store->get(1, error).has_value());
    // 2 is accessed 2 times.
    XCTAssertTrue(store->get(2, error).has_value());
    XCTAssertTrue(store->get(2, error).has_value());
    // 3 is accessed 1 times.
    XCTAssertTrue(store->get(3, error).has_value());
    {
        std::vector<int> keys;
        XCTAssertTrue(store->get_keys_sorted_by_access_count([&keys](int key) {
            keys.push_back(key);
            return true;
        }, SortOrder::Ascending, error));
        // 3 is accessed 1 time, 2 is accessed 2 times, and 1 is accessed 3 times.
        XCTAssertTrue(keys == (std::vector<int>{3, 2, 1}));
    }

    {
        std::vector<int> keys;
        XCTAssertTrue(store->get_keys_sorted_by_access_count([&keys](int key) {
            keys.push_back(key);
            return true;
        }, SortOrder::Descending, error));
        // 3 is accessed 1 time, 2 is accessed 2 times, and 1 is accessed 3 times.
        XCTAssertTrue(keys == (std::vector<int>{1, 2, 3}));
    }
}

- (void)testKVStorePurge {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    auto store = KeyValueStore<int, std::string>::make(std::move(database), "test", error);
    XCTAssertTrue(store->put(1, std::string("1"), error));
    XCTAssertTrue(store->put(2, std::string("2"), error));
    XCTAssertTrue(store->size(error) == 2);
    XCTAssertTrue(store->purge(error));
    XCTAssertTrue(store->size(error) == 0);
}

@end
