//
// inmemory_filesystem_py.cpp
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <system_error>

#include "inmemory_filesystem.hpp"
#include "memory_buffer.hpp"
#include "memory_stream.hpp"

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

using namespace inmemoryfs;

namespace executorchcoreml {

/// Flattens the directory contents at the specified path.
///
/// @param path  The directory path
/// @retval The flattened directory contents.
pybind11::bytes flatten_directory_contents(const std::string& path) {
    std::filesystem::path fs_path(path);
    std::error_code error;
    auto canonical_path = std::filesystem::canonical(fs_path);

    if (error) {
        throw std::system_error(error.value(), error.category(), error.message());
    }

    auto inMemoryfs = InMemoryFileSystem::make(canonical_path, error);
    if (error) {
        throw std::system_error(error.value(), error.category(), error.message());
    }

    size_t length = inMemoryfs->get_serialization_size({}, 1);
    auto bytes = PyBytes_FromStringAndSize(NULL, length);
    void* data = static_cast<void*>(PyBytes_AsString(bytes));
    auto buffer = MemoryBuffer::make_unowned(data, length);
    auto memstream = MemoryOStream(buffer);
    inMemoryfs->serialize({}, 1, memstream);

    return pybind11::reinterpret_steal<pybind11::bytes>((PyObject*)bytes);
}

} // namespace executorchcoreml

PYBIND11_MODULE(executorchcoreml, mod) {
    mod.def("flatten_directory_contents", &executorchcoreml::flatten_directory_contents);
}
