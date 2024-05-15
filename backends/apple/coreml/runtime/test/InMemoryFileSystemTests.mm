//
// InMemoryFileSystemTests.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import <iostream>
#import <sstream>

#import <inmemory_filesystem_utils.hpp>
#import <memory_stream.hpp>
#import <nlohmann/json.hpp>
#import <json_util.hpp>

using json = nlohmann::json;

namespace {
using namespace inmemoryfs;

struct Content {
    inline Content(std::string identifier, std::string value) noexcept
    :identifier(std::move(identifier)), value(std::move(value))
    {}

    inline Content() noexcept
    :identifier(""), value("")
    {}

    std::string identifier;
    std::string value;
};

bool operator==(const Content& lhs, const Content& rhs) {
    return lhs.identifier == rhs.identifier && lhs.value == rhs.value;
}

void to_json(json& j, const Content& content) {
    j = json{
        {"identifier", content.identifier},
        {"value", content.value}
    };
}

void from_json(const json& j, Content& content) {
    j.at("identifier").get_to(content.identifier);
    j.at("value").get_to(content.value);
}

template <typename T>
std::shared_ptr<MemoryBuffer> to_memory_buffer(const T& value) {
    std::stringstream ss;
    json j;
    to_json(j, value);
    ss << j;
    auto text = ss.str();
    return MemoryBuffer::make_copy(static_cast<void *>(text.data()), text.size());
}

template <typename T>
T from_memory_buffer(const std::shared_ptr<MemoryBuffer>& buffer) {
    T result;
    MemoryIStream memstream(buffer);
    json j;
    memstream >> j;
    from_json(j, result);
    return result;
}

std::string generate_random_string(size_t length) {
    static const char chars[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += chars[rand() % (sizeof(chars) - 1)];
    }

    return result;
}

struct SerdeVerificationConfig {
    InMemoryFileSystem::FileLoadOption file_load_option;
    size_t alignment;
    size_t n_files;
    size_t file_base_length;
};
}

@interface InMemoryFileSystemTests : XCTestCase

@end

@implementation InMemoryFileSystemTests

using namespace inmemoryfs;

- (void)testCreation {
    auto fs = InMemoryFileSystem("test");
    XCTAssert(fs.root()->name() == "test");
}

- (void)testMakeFileAtPath {
    auto fs = InMemoryFileSystem("test");
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    std::error_code error;
    XCTAssertTrue(fs.make_file({"content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));
    // This must fail if we try to overwrite the file at the same path but with the `overwrite` parameter set to true.
    XCTAssertFalse(fs.make_file({"content.json"}, buffer, InMemoryFileSystem::Attributes(), false /*overwrite*/, error));
    // This must pass if we try to overwrite the file at the same path but with the `overwrite` parameter set to true.
    XCTAssertTrue(fs.make_file({"content.json"}, buffer, InMemoryFileSystem::Attributes(), true /*overwrite*/, error));
}

- (void)testMakeDirectoryAtPath {
    auto fs = InMemoryFileSystem("test");
    std::error_code error;
    XCTAssertTrue(fs.make_directory({"dir1"}, InMemoryFileSystem::Attributes(), false, error));
    // This must fail, `dir2` doesn't exists and `create_intermediate_directories` is `false`.
    XCTAssertFalse(fs.make_directory({"dir1", "dir2", "dir3"}, InMemoryFileSystem::Attributes(), false /*create_intermediate_directories*/, error));
    // This must pass, `dir2` doesn't exists but `create_intermediate_directories` is `true`.
    XCTAssertTrue(fs.make_directory({"dir1", "dir2", "dir3"}, InMemoryFileSystem::Attributes(), true /*create_intermediate_directories*/, error));
}

- (void)testIsDirectory {
    auto fs = InMemoryFileSystem("test");
    std::error_code error;
    XCTAssertTrue(fs.make_directory({"dir1"}, InMemoryFileSystem::Attributes(), false, error));
    XCTAssertTrue(fs.is_directory({"dir1"}));
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    XCTAssertTrue(fs.make_file({"dir1", "content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));
    XCTAssertFalse(fs.is_directory({"dir1", "content.json"}));
}

- (void)testIsFile {
    auto fs = InMemoryFileSystem("test");
    std::error_code error;
    XCTAssertTrue(fs.make_directory({"dir1"}, InMemoryFileSystem::Attributes(), false, error));
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    XCTAssertTrue(fs.make_file({"dir1", "content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));
    XCTAssertTrue(fs.is_file({"dir1", "content.json"}));
    XCTAssertFalse(fs.is_file({"dir1"}));
}

- (void)testFileContentAtPath {
    auto fs = InMemoryFileSystem("test");
    std::error_code error;
    XCTAssertTrue(fs.make_directory({"dir1"}, InMemoryFileSystem::Attributes(), false, error));
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    XCTAssertTrue(fs.make_file({"dir1", "content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));
    auto contents = fs.get_file_content({"dir1", "content.json"}, error);
    XCTAssert(contents != nullptr);
}

- (void)testRemoveItemAtPath {
    auto fs = InMemoryFileSystem("test");
    std::error_code error;
    XCTAssertTrue(fs.make_directory({"dir1"}, InMemoryFileSystem::Attributes(), false, error));
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    XCTAssertTrue(fs.make_file({"dir1", "content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));
    XCTAssertTrue(fs.remove_item({"dir1", "content.json"}, error));
    XCTAssertFalse(fs.exists({"dir1", "content.json"}));
    XCTAssertTrue(fs.remove_item({"dir1"}, error));
    XCTAssertFalse(fs.exists({"dir1"}));
}

- (void)testWriteItemAtPath {
    auto fs = InMemoryFileSystem("test");
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    std::error_code error;

    XCTAssertTrue(fs.make_directory({"dir1"}, InMemoryFileSystem::Attributes(), false, error));
    XCTAssertTrue(fs.make_file({"dir1", "content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));
    XCTAssertTrue(fs.make_directory({"dir1", "dir2"}, InMemoryFileSystem::Attributes(), false, error));
    XCTAssertTrue(fs.make_file({"dir1", "dir2", "content.json"}, buffer, InMemoryFileSystem::Attributes(), false  /*overwrite*/, error));

    NSURL *dirURL = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSFileManager *fm = [[NSFileManager alloc] init];
    NSError *localError = nil;
    XCTAssertTrue([fm createDirectoryAtURL:dirURL withIntermediateDirectories:NO attributes:@{} error:&localError]);
    // Only write dir2
    XCTAssertTrue(fs.write_item_to_disk({"dir1", "dir2"}, dirURL.path.UTF8String, true, error));
    {
        NSData *data = [NSData dataWithContentsOfURL:[dirURL URLByAppendingPathComponent:@"dir2/content.json"]];
        XCTAssert(data.length > 0);
    }
    // Dump the whole thing
    XCTAssertTrue(fs.write_item_to_disk({}, dirURL.path.UTF8String, true, error));
    {
        NSData *data = [NSData dataWithContentsOfURL:[dirURL URLByAppendingPathComponent:@"test/dir1/content.json"]];
        XCTAssert(data.length > 0);
        data = [NSData dataWithContentsOfURL:[dirURL URLByAppendingPathComponent:@"test/dir1/dir2/content.json"]];
        XCTAssert(data.length > 0);
    }
    XCTAssertTrue([fm removeItemAtURL:dirURL error:&localError]);
}

- (void)testCreationFromFileSystem {
    Content content("abc", "xyz");
    std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
    NSURL *dirURL = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSFileManager *fm = [[NSFileManager alloc] init];
    NSError *localError = nil;
    XCTAssertTrue([fm createDirectoryAtURL:dirURL withIntermediateDirectories:NO attributes:@{} error:&localError]);
    // create dir1
    XCTAssertTrue([fm createDirectoryAtURL:[dirURL URLByAppendingPathComponent:@"dir1"] withIntermediateDirectories:NO attributes:@{} error:&localError]);
    // create dir2
    XCTAssertTrue([fm createDirectoryAtURL:[dirURL URLByAppendingPathComponent:@"dir2"] withIntermediateDirectories:NO attributes:@{} error:&localError]);
    // write content
    NSData *data = [NSData dataWithBytesNoCopy:buffer->data() length:buffer->size() freeWhenDone:NO];
    XCTAssertTrue([data writeToURL:[dirURL URLByAppendingPathComponent:@"dir1/content.json"] atomically:YES]);
    XCTAssertTrue([data writeToURL:[dirURL URLByAppendingPathComponent:@"dir2/content.json"] atomically:YES]);

    std::filesystem::path dirPath(dirURL.path.UTF8String);
    std::error_code error;
    auto fs = InMemoryFileSystem::make_from_directory(dirPath,
                                                      InMemoryFileSystem::FileLoadOption::Malloc,
                                                      error);
    XCTAssertTrue(fs->is_directory({"dir1"}));
    XCTAssertTrue(fs->is_file({"dir1", "content.json"}));
    XCTAssertTrue(fs->is_directory({"dir2"}));
    XCTAssertTrue(fs->is_file({"dir2", "content.json"}));
    XCTAssertTrue([fm removeItemAtURL:dirURL error:&localError]);
}

- (void)_testSerdeWithConfig:(SerdeVerificationConfig)config {
    NSURL *dirURL = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSFileManager *fm = [[NSFileManager alloc] init];
    NSError *localError = nil;
    XCTAssertTrue([fm createDirectoryAtURL:dirURL withIntermediateDirectories:NO attributes:@{} error:&localError]);

    // Create content.
    std::error_code error;
    std::vector<Content> fileContents;
    fileContents.reserve(config.n_files);
    {
        auto fs = InMemoryFileSystem("test");
        XCTAssertTrue(fs.make_directory({"dir"}, InMemoryFileSystem::Attributes(), false, error));
        for (NSUInteger i = 0; i < config.n_files; i++) {
            std::string name = "file_";
            name.append(std::to_string(i));
            Content content(name, generate_random_string(config.file_base_length * (i + 1)));
            std::shared_ptr<MemoryBuffer> buffer = to_memory_buffer(content);
            XCTAssertTrue(fs.make_file({"dir", name}, buffer, InMemoryFileSystem::Attributes(), false /*overwrite*/, error));
            fileContents.emplace_back(std::move(content));
        }
        XCTAssertTrue(fs.write_item_to_disk({}, dirURL.path.UTF8String, true, error));
    }

    // Verify serialization.
    std::shared_ptr<MemoryBuffer> buffer = nullptr;
    {
        std::error_code error;
        auto fs = InMemoryFileSystem::make_from_directory(dirURL.path.UTF8String,
                                                          config.file_load_option,
                                                          error);

        XCTAssertTrue(fs != nullptr);
        size_t length = inmemoryfs::get_buffer_size_for_serialization(*fs, {}, config.alignment);
        switch (config.file_load_option) {
            case InMemoryFileSystem::FileLoadOption::LazyMMap: {
                buffer = MemoryBuffer::make_using_mmap(length);
                break;
            }

            default:
                buffer = MemoryBuffer::make_using_malloc(length);
                break;
        }

        XCTAssertTrue(inmemoryfs::serialize(*fs, {}, config.alignment, buffer->data(), error));
    }

    // Verify de-serialization.
    {
        auto fs = inmemoryfs::make_from_buffer(buffer);
        XCTAssertTrue(fs != nullptr);
        XCTAssertTrue(fs->is_directory({"test", "dir"}));
        for (const auto& content : fileContents) {
            XCTAssertEqual(from_memory_buffer<Content>(fs->get_file_content({"test", "dir", content.identifier}, error)), content);
        }
    }

    [fm removeItemAtURL:dirURL error:nil];
}

- (void)testSerde {
    std::vector<SerdeVerificationConfig> configs;
    configs.emplace_back(SerdeVerificationConfig {
        .file_load_option = InMemoryFileSystem::FileLoadOption::Malloc,
        .n_files = 5,
        .file_base_length = 100,
        .alignment = 1,
    });
    configs.emplace_back(SerdeVerificationConfig {
        .file_load_option = InMemoryFileSystem::FileLoadOption::Malloc,
        .n_files = 5,
        .file_base_length = 100,
        .alignment = 64,
    });
    configs.emplace_back(SerdeVerificationConfig {
        .file_load_option = InMemoryFileSystem::FileLoadOption::MMap,
        .n_files = 5,
        .file_base_length = 100,
        .alignment = 1,
    });
    configs.emplace_back(SerdeVerificationConfig {
        .file_load_option = InMemoryFileSystem::FileLoadOption::MMap,
        .n_files = 5,
        .file_base_length = 100,
        .alignment = 64,
    });
    configs.emplace_back(SerdeVerificationConfig {
        .file_load_option = InMemoryFileSystem::FileLoadOption::LazyMMap,
        .n_files = 5,
        .file_base_length = 100,
        .alignment = (size_t)getpagesize(),
    });
    configs.emplace_back(SerdeVerificationConfig {
        .file_load_option = InMemoryFileSystem::FileLoadOption::LazyMMap,
        .n_files = 5,
        .file_base_length = 100,
        .alignment = 2 * (size_t)getpagesize(),
    });

    for (const auto& config : configs) {
        [self _testSerdeWithConfig:config];
    }
}

- (void)testReadJSONObject {
    using json = nlohmann::json;
    {
        std::stringstream ss;
        std::string fragment("{\"x\" : 1}xyz");
        ss << fragment;
        auto object = executorchcoreml::json::read_object_from_stream(ss);
        XCTAssertTrue(object.has_value(), "There is a valid json object, `read_json_object` must not return nullopt");
        auto j = json::parse(object.value().begin(), object.value().end());
        XCTAssertEqual(j["x"], 1, "The value must match");
    }

    {
        std::stringstream ss;
        std::string fragment("{\"x\" : 1");
        ss << fragment;
        auto object = executorchcoreml::json::read_object_from_stream(ss);
        XCTAssertFalse(object.has_value(), "There is no closing brace, `read_json_object` must return nullopt");
    }


    {
        std::stringstream ss;
        std::string fragment("{\"x\" : \"\\\"1\"}xyz");
        ss << fragment;
        auto object = executorchcoreml::json::read_object_from_stream(ss);
        XCTAssertTrue(object.has_value(), "There is a valid json object, `read_json_object` must not return nullopt");
        auto j = json::parse(object.value().begin(), object.value().end());
        std::string value = j["x"];
        XCTAssertEqual(value, std::string("\"1"), "The value must match");
    }

    {
        std::stringstream ss;
        std::string fragment("{sdhalskjks}");
        ss << fragment;
        auto object = executorchcoreml::json::read_object_from_stream(ss);
        XCTAssertTrue(object.has_value(), "The json object is invalid but is correctly nested, `read_json_object` must not return nullopt");
        std::exception_ptr eptr;
        try {
            auto j = json::parse(object.value().begin(), object.value().end());
        } catch (...) {
            eptr = std::current_exception();
        }
        XCTAssertNotEqual(eptr, nullptr, "Parsing invalid json object must throw an exception");
    }

}

@end
