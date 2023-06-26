#include <executorch/util/read_file.h>

#include <executorch/core/Log.h>

#include <stdio.h>
#include <memory>

#if defined(ET_MMAP_SUPPORTED)
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace torch {
namespace executor {
namespace util {

__ET_NODISCARD Error read_file_content(
    const char* file_name,
    std::shared_ptr<char>* file_data,
    size_t* file_length) {
  FILE* file;
  unsigned long fileLen;

  // Open file
  file = fopen(file_name, "rb");
  if (!file) {
    ET_LOG(Error, "Unable to open file %s\n", file_name);
    return Error::NotSupported;
  }

  // Get file length
  fseek(file, 0, SEEK_END);
  fileLen = ftell(file);
  fseek(file, 0, SEEK_SET);

  // Allocate memory
  auto ptr = std::shared_ptr<char>(
      new char[fileLen + 1], std::default_delete<char[]>());
  if (!ptr) {
    ET_LOG(Error, "Unable to allocate memory to read file %s\n", file_name);
    fclose(file);
    return Error::NotSupported;
  }

  // Read file contents into buffer
  fread(ptr.get(), fileLen, 1, file);
  fclose(file);

  *file_data = ptr;
  *file_length = fileLen;
  return Error::Ok;
}

__ET_DEPRECATED std::shared_ptr<char> read_file_content(const char* name) {
  std::shared_ptr<char> file_data;
  size_t file_length;
  Error status = read_file_content(name, &file_data, &file_length);
  if (status == Error::Ok) {
    return file_data;
  } else {
    return nullptr;
  }
}

#if defined(ET_MMAP_SUPPORTED)
Error mmap_file_content(
    const char* file_name,
    std::shared_ptr<char>* file_data,
    size_t* file_length) {
  ET_CHECK_OR_RETURN_ERROR(
      file_data != nullptr,
      InvalidArgument,
      "file_data pointer must not be null.");
  ET_CHECK_OR_RETURN_ERROR(
      file_length != nullptr,
      InvalidArgument,
      "file_length pointer must not be null.");
  int fd = open(file_name, O_RDONLY);
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0,
      NotFound,
      "Erro while opening file %s: error is: %s",
      file_name,
      strerror(errno));
  struct stat statbuf {};
  ET_CHECK_OR_RETURN_ERROR(
      fstat(fd, &statbuf) == 0,
      Internal,
      "Could not query size of the opened file %s",
      file_name);
  void* ptr = mmap(
      nullptr, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, /* offset */ 0);
  close(fd);
  ET_CHECK_OR_RETURN_ERROR(ptr != nullptr, Internal, "mmap returned nullptr");
  // Note that it is ok to call munmap on a range that does not contain
  // any mapped pages:
  // "It is not an error if the indicated range does not contain any mapped
  // pages."
  // ^^ from https://linux.die.net/man/2/munmap.
  // Note that it does not talk about partially range that is not mapped.
  auto deleter = [statbuf](char* ptr) { munmap(ptr, statbuf.st_size); };
  std::shared_ptr<char> shared_ptr(reinterpret_cast<char*>(ptr), deleter);
  *file_length = statbuf.st_size;
  *file_data = shared_ptr;
  return Error::Ok;
}
#endif

} // namespace util
} // namespace executor
} // namespace torch
