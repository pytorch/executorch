#include <executorch/util/memory_utils.h>

#if defined(ET_MMAP_SUPPORTED)
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <executorch/core/Error.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/log.h>

namespace torch {
namespace executor {
namespace util {
void mark_memory_as_unused(void* ptr, const size_t nbytes) {
  int ps = sysconf(_SC_PAGESIZE);
  // Page mask will help zero out lower N bits of the address pointed by ptr.
  // For 4k page size N = 12
  uintptr_t page_mask = ~(ps - 1);
  void* orig_ptr = ptr;
  // Following will generate page size aligned address
  // If the address is not aligned, it will align it to the next page.
  void* ptr_to_free = reinterpret_cast<void*>(
      reinterpret_cast<size_t>((reinterpret_cast<void*>(
          reinterpret_cast<ptrdiff_t>(orig_ptr) + (ps - 1)))) &
      page_mask);
  // Since we align the address, when not aligned, to the start of
  // the next page, we must subtract the bytes occupied in the current
  // page. These are the bytes starting at address ptr.
  size_t bytes_to_subtract = reinterpret_cast<size_t>(ptr_to_free) -
      reinterpret_cast<size_t>(orig_ptr);
  size_t bytes_to_free = 0;
  if (nbytes > bytes_to_subtract) {
    bytes_to_free = nbytes - bytes_to_subtract;
  }
  // Only request freeing multiple of page size
  // Meaning subtract bytes that partially occupy the last page.
  bytes_to_free = bytes_to_free & page_mask;

  auto status = munlock(ptr_to_free, bytes_to_free);
  if (status != 0) {
    ET_LOG(Info, "Failed to unlock, returned error is %s", strerror(errno));
  }
  status = madvise(ptr_to_free, bytes_to_free, MADV_DONTNEED);
  if (status != 0) {
    ET_LOG(Info, "Failed to madvise, returned error is %s", strerror(errno));
  }
}

} // namespace util
} // namespace executor
} // namespace torch
#endif
