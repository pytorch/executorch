//
// types.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace executorchcoreml {
namespace sqlite {

struct Null {};

struct UnOwnedBlob {
    inline explicit UnOwnedBlob(const void *data, size_t size)
    :data(data), size(size)
    {}
    
    const void *data;
    size_t size;
};

struct UnOwnedString {
    inline UnOwnedString(const char *data, size_t size)
    :data(data), size(size)
    {}
    
    inline UnOwnedString(const std::string& string)
    :data(string.c_str()), size(string.size())
    {}
    
    inline bool empty() const noexcept {
        return size == 0;
    }
    
    inline std::string toString() const noexcept {
        return std::string(data);
    }
    
    const char *data;
    size_t size;
};

struct Blob {
    static inline void *copy_data(const void *data, size_t size) {
        void *result = ::operator new(size);
        std::memcpy(result, data, size);
        return result;
    }
    
    inline Blob(const void *data, size_t size)
    :data(copy_data(data, size)), size(size)
    {}
    
    Blob(Blob const&) noexcept = delete;
    Blob& operator=(Blob const&) noexcept = delete;
    
    inline Blob(Blob&& other) noexcept
    :data(std::exchange(other.data, nullptr)),
    size(std::exchange(other.size, 0))
    {}
    
    inline Blob& operator=(Blob&& other) noexcept {
        std::swap(data, other.data);
        std::swap(size, other.size);
        return *this;
    }
    
    inline ~Blob() {
        if (data) {
            ::operator delete(data);
        }
    }
    
    inline UnOwnedBlob toUnOwned() const noexcept {
        return UnOwnedBlob(data, size);
    }
    
    void *data = nullptr;
    size_t size = 0;
};

enum class StorageType: uint8_t {
    Null,
    Blob,
    Text,
    Double,
    Integer
};

using Value = std::variant<int64_t, double, std::string, Blob, Null>;
using UnOwnedValue = std::variant<int64_t, double, UnOwnedString, UnOwnedBlob, Null>;

} // namespace sqlite
} // namespace executorchcoreml
