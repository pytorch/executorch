#pragma once
// TODO(T144120904): Remove this file once delegate blobs can be freed directly

#include <executorch/util/system.h>

#if defined(ET_MMAP_SUPPORTED)
#include <cstdlib>
namespace torch {
namespace executor {
namespace util {
/**
 * Advise kernel to free region of memory pointer by ptr
 * via calling madvise(MADV_DONTNEED)
 *
 * @param[in] ptr: pointer to mmapped memory
 * @param[in] bytes: Number of bytes to free starting at ptr
 */
void mark_memory_as_unused(void* ptr, const size_t nbytes);

} // namespace util
} // namespace executor
} // namespace torch
#endif
