//
// inmemory_filesystem_py.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.


#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <system_error>
#include <thread>
#include <unistd.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "inmemory_filesystem_utils.hpp"
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

namespace executorchcoreml {

void* alloc_using_mmap(size_t size) {
    return mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
}

std::once_flag external_bytes_initialization_flag;
static PyTypeObject PyExternalBytes_Type;

static void external_bytes_free(void* ptr) {
    printf("external_bytes_free called \n");
    PyBytesObject* obj = (PyBytesObject*)ptr;
    Py_ssize_t size = Py_SIZE(obj);
    munmap(obj, size);
}

void intialize_external_bytes_type() {
    std::call_once(external_bytes_initialization_flag, []() {
        PyExternalBytes_Type = PyBytes_Type;
        PyExternalBytes_Type.tp_free = external_bytes_free;
    });
}

PyBytesObject* initialize_buffer_as_bytes_object(void* buffer, Py_ssize_t size) {
    intialize_external_bytes_type();
    PyBytesObject* obj = (PyBytesObject*)buffer;
    PyObject_INIT_VAR(obj, &PyExternalBytes_Type, size);
    obj->ob_sval[size] = '\0';

    return obj;
}

/// The method allocates memory using `mmap` and then reads the contents of the all files in the directory. The file
/// content is again memory mapped at fixed addresses in the allocated memory. The approach avoids dirtying the memory.
/// The down side of this method is that it could result in a larger file when the bytes are dumped to disk.
PyBytesObject* get_bytes_from_external_memory(const std::filesystem::path& dir_path) {
    using namespace inmemoryfs;

    std::error_code error;
    std::stringstream ss;
    auto fs = InMemoryFileSystem::make_from_directory(dir_path, InMemoryFileSystem::FileLoadOption::LazyMMap, error);
    if (fs == nullptr) {
        ss << "Failed to create InMemoryFileSystem because of error=" << error.message().c_str() << "\n";
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        return nullptr;
    }

    size_t alignment = getpagesize();
    size_t serialized_buffer_length = get_buffer_size_for_serialization(*fs, {}, alignment);
    size_t py_bytes_obj_length = offsetof(PyBytesObject, ob_sval);
    size_t py_bytes_obj_total_length = py_bytes_obj_length + serialized_buffer_length + 1;
    void* backing_buffer = alloc_using_mmap(py_bytes_obj_total_length);
    if (backing_buffer == NULL || (reinterpret_cast<int*>(backing_buffer) == MAP_FAILED)) {
        ss << "Failed to allocate memory of size=" << py_bytes_obj_total_length / (1024 * 10224) << " mb.";
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        return nullptr;
    }

    if (!serialize(*fs, {}, alignment, static_cast<uint8_t*>(backing_buffer) + py_bytes_obj_length, error)) {
        ss << "Failed to serialize directory contents because of error=" << error.message().c_str() << ".";
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        return nullptr;
    }

    PyBytesObject* bytes = initialize_buffer_as_bytes_object(backing_buffer, py_bytes_obj_total_length);
    if (bytes == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create bytes object.");
        return nullptr;
    }

    return bytes;
}

/// The method writes to the memory managed by the python bytes object. The method dirties the memory and can be slow
/// but results in a relatively smaller file when the bytes are dumped to disk.
PyBytesObject* get_bytes(inmemoryfs::InMemoryFileSystem& fs, size_t length) {
    using namespace inmemoryfs;

    std::error_code error;
    PyObject* bytes = PyBytes_FromStringAndSize(NULL, length);
    void* data = static_cast<void*>(PyBytes_AsString(bytes));
    if (!serialize(fs, {}, 1, data, error)) {
        throw std::system_error(error.value(), error.category(), error.message());
    }

    return (PyBytesObject*)bytes;
}

bool is_large_model(size_t model_size_in_bytes) {
    static constexpr size_t large_model_size_threshold = 1024 * 1024 * 1024; // 1 GB
    return model_size_in_bytes > large_model_size_threshold;
}

/// Flattens the directory contents at the specified path.
///
/// @param path  The directory path
/// @retval The flattened directory contents.
pybind11::bytes flatten_directory_contents(const std::string& path) {
    using namespace inmemoryfs;

    std::filesystem::path fs_path(path);
    std::error_code error;
    auto canonical_path = std::filesystem::canonical(fs_path);
    std::stringstream ss;
    auto fs = InMemoryFileSystem::make_from_directory(canonical_path, InMemoryFileSystem::FileLoadOption::MMap, error);
    if (fs == nullptr) {
        ss << "Failed to create InMemoryFileSystem because of error=" << error.message().c_str() << ".";
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        return nullptr;
    }

    size_t model_size_in_bytes = get_buffer_size_for_serialization(*fs, {}, 1);
    PyBytesObject* bytes = nullptr;
    if (is_large_model(model_size_in_bytes)) {
        bytes = get_bytes_from_external_memory(canonical_path);
    } else {
        bytes = get_bytes(*fs, model_size_in_bytes);
    }

    return bytes == nullptr ? pybind11::none() : pybind11::reinterpret_steal<pybind11::object>((PyObject*)bytes);
}

/// Unflattens and writes the contents of the memory buffer at the specified path.
///
/// @param bytes  The bytes returned from `flatten_directory_contents`.
/// @param path  The directory path
bool unflatten_directory_contents(pybind11::bytes bytes, const std::string& path) {
    using namespace inmemoryfs;

    char* buffer = nullptr;
    ssize_t length = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bytes.ptr(), &buffer, &length)) {
        pybind11::pybind11_fail("Failed to extract contents of bytes object!");
    }
    std::shared_ptr<MemoryBuffer> memory_buffer =
        MemoryBuffer::make_unowned((void*)buffer, static_cast<size_t>(length));
    auto fs = inmemoryfs::make_from_buffer(memory_buffer);
    if (!fs) {
        pybind11::pybind11_fail("Failed to de-serialize bytes object!");
        return false;
    }
    std::error_code ec;
    std::filesystem::path fs_path(path);
    auto canonical_path = std::filesystem::canonical(fs_path);
    if (!fs->write_item_to_disk({}, canonical_path, true, ec)) {
        pybind11::pybind11_fail("Failed to write the item to disk!");
        return false;
    }

    return true;
}
} // namespace executorchcoreml

PYBIND11_MODULE(executorchcoreml, mod) {
    mod.def("flatten_directory_contents", &executorchcoreml::flatten_directory_contents);
    mod.def("unflatten_directory_contents", &executorchcoreml::unflatten_directory_contents);
}
