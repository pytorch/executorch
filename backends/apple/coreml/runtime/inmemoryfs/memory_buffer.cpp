//
// memory_buffer.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "memory_buffer.hpp"

#include <assert.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace inmemoryfs;

using MMAP_HANDLE = std::unique_ptr<void, std::function<void(void*)>>;

MMAP_HANDLE memory_map_file_range(FILE* file, Range range) {
    auto ptr = mmap(nullptr, range.size, PROT_READ, MAP_PRIVATE, fileno(file), range.offset);
    return MMAP_HANDLE(ptr, [size = range.size](void* bufferPtr) mutable { munmap(bufferPtr, size); });
}

MMAP_HANDLE alloc_using_mmap(size_t size) {
    auto ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return MMAP_HANDLE(ptr, [size](void* bufferPtr) mutable { munmap(bufferPtr, size); });
}

void* memory_map_file_range_at_address(FILE* file, Range range, void* dst) {
    return mmap(dst, range.size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fileno(file), range.offset);
}

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

class LazyMMappedBuffer : public MemoryBuffer {
public:
    explicit LazyMMappedBuffer(std::shared_ptr<FILE> file, Range range) noexcept
        : MemoryBuffer(nullptr, range.size, MemoryBuffer::Kind::MMap), file_(std::move(file)), range_(range),
          page_size_(getpagesize()) { }

    LazyMMappedBuffer(const LazyMMappedBuffer&) = delete;
    LazyMMappedBuffer& operator=(const LazyMMappedBuffer&) = delete;

    bool load(std::error_code& error) noexcept override {
        std::lock_guard<std::mutex> guard(mutex_);
        if (memory_mapped_data_ != nullptr) {
            return true;
        }
        auto ptr = memory_map_file_range(file_.get(), range_);
        if (!ptr || (reinterpret_cast<int*>(ptr.get()) == MAP_FAILED)) {
            error = std::error_code(errno, std::system_category());
            return false;
        }

        memory_mapped_data_ = std::move(ptr);
        return true;
    }

    void* data() noexcept override {
        std::error_code ec;
        assert(load(ec) == true);
        return memory_mapped_data_.get();
    }

    std::pair<size_t, size_t> get_offset_range(size_t proposed_offset) const noexcept override {
        return { proposed_offset, proposed_offset + page_size_ - 1 };
    }

    Range get_revised_range_for_writing(void* dst, Range proposed_range) const noexcept override {
        uint8_t* ptr = static_cast<uint8_t*>(dst) + proposed_range.offset;
        size_t alignment = page_size_;
        uintptr_t addr = (uintptr_t)ptr;
        if (addr % alignment == 0) {
            return proposed_range;
        }

        uintptr_t mask = alignment - 1;
        uintptr_t aligned_addr = (addr + mask) & ~mask;
        assert(aligned_addr >= addr);
        return Range(proposed_range.offset + (aligned_addr - addr), proposed_range.size);
    }

    bool write(void* dst, size_t offset, std::error_code& error) noexcept override {
        uint8_t* ptr = static_cast<uint8_t*>(dst) + offset;
        size_t alignment = page_size_;
        uintptr_t addr = (uintptr_t)ptr;
        if (addr % alignment != 0) {
            error = std::error_code(EFAULT, std::system_category());
            return false;
        }

        auto mmapped_ptr = memory_map_file_range_at_address(file_.get(), range_, ptr);
        if (!mmapped_ptr || (reinterpret_cast<int*>(mmapped_ptr) == MAP_FAILED)) {
            error = std::error_code(errno, std::system_category());
            return false;
        }

        return true;
    }

    std::shared_ptr<MemoryBuffer> slice(Range range) noexcept override {
        if (range.length() > size()) {
            return nullptr;
        }

        return std::make_shared<LazyMMappedBuffer>(file_, range);
    }

    ~LazyMMappedBuffer() { }

private:
    std::shared_ptr<FILE> file_;
    Range range_;
    size_t page_size_;
    std::unique_ptr<void, std::function<void(void*)>> memory_mapped_data_;
    std::mutex mutex_;
};

class MallocedBuffer : public MemoryBuffer {
public:
    enum class Ownership : uint8_t { Unowned, Owned };

    explicit MallocedBuffer(void* data, size_t size, Ownership ownership) noexcept
        : MemoryBuffer(data, size, MemoryBuffer::Kind::Malloc), ownership_(ownership) { }

    explicit MallocedBuffer(std::vector<uint8_t> buffer) noexcept
        : MemoryBuffer(buffer.data(), buffer.size(), MemoryBuffer::Kind::Malloc), buffer_(std::move(buffer)),
          ownership_(Ownership::Owned) { }

    MallocedBuffer(const MallocedBuffer&) = delete;
    MallocedBuffer& operator=(const MallocedBuffer&) = delete;

    ~MallocedBuffer() {
        if (buffer_.size() > 0) {
            return;
        }

        if (!data()) {
            return;
        }

        switch (ownership_) {
            case Ownership::Owned:
                free(data());
                break;

            default:
                break;
        }
    }

private:
    std::vector<uint8_t> buffer_;
    Ownership ownership_;
};

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
        error = std::error_code(errno, std::system_category());
    }
    return file;
}

std::unique_ptr<MemoryBuffer> mmap_buffer_from_file(FILE* file, Range range, std::error_code& error) {
    auto ptr = memory_map_file_range(file, range);
    if (!ptr || (reinterpret_cast<int*>(ptr.get()) == MAP_FAILED)) {
        error = std::error_code(errno, std::generic_category());
        return {};
    }

    return std::make_unique<MMappedBuffer>(std::move(ptr), range.size);
}

std::unique_ptr<MemoryBuffer>
mmap_buffer_lazy_from_file(std::unique_ptr<FILE, decltype(&fclose)> file, Range range, std::error_code& error) {
    return std::make_unique<LazyMMappedBuffer>(std::move(file), range);
}

std::unique_ptr<MemoryBuffer> malloced_buffer_from_file(FILE* file, Range range, std::error_code& error) {
    size_t offset = range.offset;
    if (std::fseek(file, offset, SEEK_SET) != 0) {
        error = std::error_code(errno, std::generic_category());
        return nullptr;
    }

    std::vector<uint8_t> buffer(range.size, 0);
    if (std::fread(buffer.data(), 1, buffer.size(), file) != range.size) {
        error = std::error_code(errno, std::generic_category());
        return nullptr;
    }

    return std::make_unique<MallocedBuffer>(std::move(buffer));
}

std::unique_ptr<MemoryBuffer> read_file_content(std::unique_ptr<FILE, decltype(&fclose)> file,
                                                Range range,
                                                MMappedBuffer::ReadOption option,
                                                std::error_code& error) {
    switch (option) {
        case MemoryBuffer::ReadOption::MMap: {
            return ::mmap_buffer_from_file(file.get(), range, error);
        }

        case MemoryBuffer::ReadOption::LazyMMap: {
            return ::mmap_buffer_lazy_from_file(std::move(file), range, error);
        }

        case MemoryBuffer::ReadOption::Malloc: {
            return ::malloced_buffer_from_file(file.get(), range, error);
        }
    }
}

} // namespace

namespace inmemoryfs {
bool MemoryBuffer::write(void* dst, size_t offset, std::error_code& error) noexcept {
    auto ptr = static_cast<uint8_t*>(dst);
    std::memcpy(ptr + offset, data(), size());
    return true;
}

std::shared_ptr<MemoryBuffer> MemoryBuffer::slice(Range range) noexcept {
    if (range.length() > size()) {
        return nullptr;
    }

    auto start = static_cast<void*>(static_cast<uint8_t*>(data()) + range.offset);
    return std::make_shared<MemoryBuffer>(start, range.size, kind(), shared_from_this());
}

std::vector<std::shared_ptr<MemoryBuffer>> MemoryBuffer::read_file_content(const std::string& file_path,
                                                                           const std::vector<Range>& ranges,
                                                                           ReadOption option,
                                                                           std::error_code& error) {
    auto file = open_file(file_path, error);
    if (!file) {
        return {};
    }

    std::vector<std::shared_ptr<MemoryBuffer>> result;
    result.reserve(ranges.size());
    for (const auto& range: ranges) {
        auto buffer = ::read_file_content(std::move(file), range, option, error);
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

    auto buffer = ::read_file_content(std::move(file), Range(0, length), option, error);
    return buffer;
}

std::unique_ptr<MemoryBuffer> MemoryBuffer::make_using_malloc(size_t size, size_t alignment) {
    void* data = nullptr;
    if (alignment > 1) {
        assert(size % alignment == 0);
        data = aligned_alloc(alignment, size);
    } else {
        data = malloc(size);
    }

    return std::make_unique<MallocedBuffer>(data, size, MallocedBuffer::Ownership::Owned);
}

std::unique_ptr<MemoryBuffer> MemoryBuffer::make_using_mmap(size_t size) {
    auto ptr = alloc_using_mmap(size);
    if (!ptr || (reinterpret_cast<int*>(ptr.get()) == MAP_FAILED)) {
        return nullptr;
    }

    return std::make_unique<MMappedBuffer>(std::move(ptr), size);
}

std::unique_ptr<MemoryBuffer> MemoryBuffer::make_unowned(void* data, size_t size) {
    return std::make_unique<MallocedBuffer>(data, size, MallocedBuffer::Ownership::Unowned);
}

std::unique_ptr<MemoryBuffer> MemoryBuffer::make_copy(void* data, size_t size) {
    std::vector<uint8_t> buffer;
    buffer.resize(size);
    std::memcpy(buffer.data(), data, size);
    return std::make_unique<MallocedBuffer>(std::move(buffer));
}

} // namespace inmemoryfs
