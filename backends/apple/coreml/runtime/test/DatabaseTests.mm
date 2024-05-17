//
// DatabaseTests.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import <database.hpp>
#import <nlohmann/json.hpp>

@interface DatabaseTests : XCTestCase

@end

@implementation DatabaseTests

using namespace executorchcoreml::sqlite;

- (void)testDatabaseOpen {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
}

- (void)testTableCreation {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
    XCTAssertTrue(database->execute("CREATE TABLE IF NOT EXISTS TEST (id INTEGER PRIMARY KEY, value TEXT)", error));
    XCTAssertTrue(database->table_exists("TEST", error));
    XCTAssertTrue(database->drop_table("TEST", error));
    XCTAssertFalse(database->table_exists("TEST", error));
}

- (void)testDatabaseExecute {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
    XCTAssertTrue(database->execute("CREATE TABLE IF NOT EXISTS TEST (key TEXT PRIMARY KEY, value TEXT)", error));
    auto insertStatement = database->prepare_statement("INSERT INTO TEST (key, value) VALUES ($key, $value)", error);
    XCTAssertTrue(insertStatement != nullptr);
    XCTAssertTrue(insertStatement->bind_name("$key", std::string("1"), error));
    XCTAssertTrue(insertStatement->bind_name("$value", std::string("1"), error));
    XCTAssertTrue(insertStatement->execute(error));
    XCTAssertTrue(database->get_row_count("TEST", error) == 1);
}

- (void)testDatabaseQuery {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
    XCTAssertTrue(database->execute("CREATE TABLE IF NOT EXISTS TEST (key TEXT PRIMARY KEY, value TEXT)", error));
    auto insertStatement = database->prepare_statement("INSERT INTO TEST (key, value) VALUES ($key, $value)", error);
    XCTAssertTrue(insertStatement != nullptr);
    XCTAssertTrue(insertStatement->bind_name("$key", std::string("1"), error));
    XCTAssertTrue(insertStatement->bind_name("$value", std::string("1"), error));
    XCTAssertTrue(insertStatement->execute(error));
    XCTAssertTrue(database->get_row_count("TEST", error) == 1);

    auto query = database->prepare_statement("SELECT * FROM TEST", error);
    XCTAssertTrue(query != nullptr);
    XCTAssertTrue(query->step(error));
    auto key = query->get_column_value_no_copy(0, error);
    XCTAssertFalse(std::get<UnOwnedString>(key).empty());
    auto value = query->get_column_value_no_copy(1, error);
    XCTAssertFalse(std::get<UnOwnedString>(value).empty());
    XCTAssertFalse(query->step(error));
}

- (void)testDatabaseTransactionCommit {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
    XCTAssertTrue(database->execute("CREATE TABLE IF NOT EXISTS TEST (key TEXT PRIMARY KEY, value TEXT)", error));
    // Insert a row inside a transaction.
    XCTAssertTrue(database->transaction([database = database.get()]() {
        std::error_code transactionError;
        auto insertStatement = database->prepare_statement("INSERT INTO TEST (key, value) VALUES ($key, $value)", transactionError);
        XCTAssertTrue(insertStatement != nullptr);
        XCTAssertTrue(insertStatement->bind_name("$key", std::string("1"), transactionError));
        XCTAssertTrue(insertStatement->bind_name("$value", std::string("1"), transactionError));
        XCTAssertTrue(insertStatement->execute(transactionError));
        XCTAssertTrue(database->get_row_count("TEST", transactionError) == 1);
        // Returns true to commit the transaction.
        return true;
    }, Database::TransactionBehavior::Immediate, error));
    // The row must exist because the transaction was committed.
    XCTAssertTrue(database->get_row_count("TEST", error) == 1);
}

- (void)testDatabaseTransactionRollback {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
    XCTAssertTrue(database->execute("CREATE TABLE IF NOT EXISTS TEST (key TEXT PRIMARY KEY, value TEXT)", error));
    // Insert a row inside a transaction.
    XCTAssertFalse(database->transaction([database = database.get()]() {
        std::error_code transactionError;
        auto insertStatement = database->prepare_statement("INSERT INTO TEST (key, value) VALUES ($key, $value)", transactionError);
        XCTAssertTrue(insertStatement != nullptr);
        XCTAssertTrue(insertStatement->bind_name("$key", std::string("1"), transactionError));
        XCTAssertTrue(insertStatement->bind_name("$value", std::string("1"), transactionError));
        XCTAssertTrue(insertStatement->execute(transactionError));
        XCTAssertTrue(database->get_row_count("TEST", transactionError) == 1);
        // Returns false to rollback the transaction.
        return false;
    }, Database::TransactionBehavior::Immediate, error));
    // The row must exist because the transaction was rollbacked.
    XCTAssertTrue(database->get_row_count("TEST", error) == 0);
}

- (void)testDatabaseLastError {
    std::error_code error;
    auto database = Database::make_inmemory(Database::SynchronousMode::Normal, 100, error);
    XCTAssert(database != nullptr);
    XCTAssertTrue(database->execute("CREATE TABLE IF NOT EXISTS TEST (key TEXT PRIMARY KEY, value TEXT)", error));
    auto insertStatement = database->prepare_statement("INSERT INTO TEST (key, value) VALUES ($key, $value)", error);
    XCTAssertTrue(insertStatement->bind_name("$key", std::string("1"), error));
    XCTAssertTrue(insertStatement->bind_name("$value", std::string("1"), error));
    XCTAssertTrue(insertStatement->execute(error));
    // Reset the insert statement.
    XCTAssertTrue(insertStatement->reset(error));
    // Insert the same key, this is a constraint violation.
    XCTAssertTrue(insertStatement->bind_name("$key", std::string("1"), error));
    XCTAssertTrue(insertStatement->bind_name("$value", std::string("1"), error));
    XCTAssertFalse(insertStatement->execute(error));
    XCTAssertEqual(database->get_last_error_code().value(), SQLITE_CONSTRAINT);
    XCTAssertEqual(database->get_last_extended_error_code().value(), SQLITE_CONSTRAINT_PRIMARYKEY);
    XCTAssertFalse(database->get_last_error_message().empty());
}

@end
