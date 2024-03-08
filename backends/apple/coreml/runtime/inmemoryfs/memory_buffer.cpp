//
// memory_buffer.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include <memory_buffer.hpp>

#include <functional>
#include <iostream>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace {
using namespace inmemoryfs;

class MMappedBuffer : public MemoryBuffer {
public:
    explicit MMappedBuffer(std::shared_ptr<void> buffer, size_t size) noexcept
        : MemoryBuffer(reinterpret_cast<uint8_t*>(buffer.get()), size, MemoryBuffer::Kind::MMap),
          buffer_(std::move(buffer)) { }

    MMappedBuffer(const MemoryBuffer&) = delete;
    MMappedBuffer& operator=(const MemoryBuffer&) = delete;

    ~MMappedBuffer() { }

private:
    const std::shared_ptr<void> buffer_;
};

class MallocedBuffer : public MemoryBuffer {
public:
    explicit MallocedBuffer(std::vector<uint8_t> buffer) noexcept
        : MemoryBuffer(buffer.data(), buffer.size(), MemoryBuffer::Kind::Malloc), buffer_(std::move(buffer)) { }

    MallocedBuffer(const MallocedBuffer&) = delete;
    MallocedBuffer& operator=(const MallocedBuffer&) = delete;

    ~MallocedBuffer() { }

private:
    const std::vector<uint8_t> buffer_;
};

std::unique_ptr<void, std::function<void(void*)>> mmap_file_region(FILE* file, MemoryRegion region) {
    auto ptr = mmap(nullptr, region.size, PROT_READ, MAP_PRIVATE, fileno(file), region.offset);
    return std::unique_ptr<void, std::function<void(void*)>>(
        ptr, [size = region.size](void* bufferPtr) mutable { munmap(bufferPtr, size); });
}

size_t get_file_length(const std::string& file_path, std::error_code& error) {
    struct stat fileInfo;
    if (stat(file_path.c_str(), &fileInfo) != 0) {
        error = std::error_code(errno, std::generic_category());
        return 0;
    }

    return static_cast<size_t>(fileInfo.st_size);
}

std::unique_ptr<FILE, decltype(&fclose)> open_file(const std::string& file_path, std::error_code& error) {
    std::unique_ptr<FILE, decltype(&fclose)> file(fopen(file_path.c_str(), "rb"), fclose);
    if (!file) {
        error = std::error_code(errno, std::generic_category());
    }
    return file;
}

std::unique_ptr<MemoryBuffer> mmap_buffer_from_file(FILE* file, MemoryRegion region, std::error_code& error) {
    error.clear();
    auto ptr = mmap_file_region(file, region);
    if (!ptr || (reinterpret_cast<int*>(ptr.get()) == MAP_FAILED)) {
        error = std::error_code(errno, std::generic_category());
        return {};
    }

    return std::make_unique<MMappedBuffer>(std::move(ptr), region.size);
}

std::unique_ptr<MemoryBuffer> malloced_buffer_from_file(FILE* file, MemoryRegion region, std::error_code& error) {
    error.clear();
    size_t offset = region.offset;
    if (std::fseek(file, offset, SEEK_SET) != 0) {
        error = std::error_code(errno, std::generic_category());
        return nullptr;
    }

    std::vector<uint8_t> buffer(region.size, 0);
    if (std::fread(buffer.data(), 1, buffer.size(), file) != region.size) {
        error = std::error_code(errno, std::generic_category());
        return nullptr;
    }

    return std::make_unique<MallocedBuffer>(std::move(buffer));
}

std::unique_ptr<MemoryBuffer>
read_file_content(FILE* file, MemoryRegion region, MMappedBuffer::ReadOption option, std::error_code& error) {
    switch (option) {
        case MemoryBuffer::ReadOption::MMap: {
            return mmap_buffer_from_file(file, region, error);
        }

        case MemoryBuffer::ReadOption::Malloc: {
            return malloced_buffer_from_file(file, region, error);
        }

        case MemoryBuffer::ReadOption::Any: {
            auto buffer = mmap_buffer_from_file(file, region, error);
            if (!buffer) {
                buffer = malloced_buffer_from_file(file, region, error);
            }
            return buffer;
        }
    }
}
} // namespace

namespace inmemoryfs {
std::shared_ptr<MemoryBuffer> MemoryBuffer::slice(MemoryRegion region) noexcept {
    if (region.get_length() > size()) {
        return nullptr;
    }

    auto start = static_cast<void*>(static_cast<uint8_t*>(data()) + region.offset);
    return std::make_shared<MemoryBuffer>(start, region.size, kind(), shared_from_this());
}

std::vector<std::shared_ptr<MemoryBuffer>> MemoryBuffer::read_file_content(const std::string& file_path,
                                                                           const std::vector<MemoryRegion>& regions,
                                                                           ReadOption option,
                                                                           std::error_code& error) {
    auto file = open_file(file_path, error);
    if (!file) {
        return {};
    }

    std::vector<std::shared_ptr<MemoryBuffer>> result;
    result.reserve(regions.size());
    for (const auto& region: regions) {
        auto buffer = ::read_file_content(file.get(), region, option, error);
        if (!buffer) {
            return {};
        }
        result.emplace_back(std::move(buffer));
    }

    return result;
}

std::shared_ptr<MemoryBuffer>
MemoryBuffer::read_file_content(const std::string& file_path, ReadOption option, std::error_code& error) {
    const size_t length = ::get_file_length(file_path, error);
    if (length == 0) {
        return {};
    }

    auto file = open_file(file_path, error);
    if (!file) {
        return {};
    }

    auto buffer = ::read_file_content(file.get(), MemoryRegion(0, length), option, error);
    return buffer;
}

std::shared_ptr<MemoryBuffer> MemoryBuffer::make(const void* data, size_t size) {
    std::vector<uint8_t> bytes;
    bytes.resize(size);
    std::memcpy(bytes.data(), data, size);
    return std::make_unique<MallocedBuffer>(std::move(bytes));
}

std::shared_ptr<MemoryBuffer> MemoryBuffer::make(std::vector<uint8_t> bytes) {
    return std::make_unique<MallocedBuffer>(std::move(bytes));
}

std::shared_ptr<MemoryBuffer> MemoryBuffer::make_unowned(void* data, size_t size) {
    return std::make_unique<MemoryBuffer>(data, size);
}

} // namespace inmemoryfs
