// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <executorch/util/memory_utils.h>

#include <gtest/gtest.h>

#if defined(ET_MMAP_SUPPORTED)
using namespace ::testing;
static constexpr size_t kB = 1024;
TEST(MemoryUtilsTest, FreeUnmappedMemory) {
  size_t size = 20 * kB;
  {
    std::vector<char> some_memory(size);
    // free random memory. should not die even when no mmaped.
    torch::executor::util::mark_memory_as_unused(some_memory.data(), size);
  }

  {
    // subsequently when original vector is freed, it should not crash.
    std::vector<char> some_memory(size);
    size_t offset = 2000;
    // free random memory off of offset. should not die even when no mmaped.
    size_t size_to_free = size - offset;
    torch::executor::util::mark_memory_as_unused(
        some_memory.data() + offset, size_to_free);
  }
}
// Would like to add another test that actually write a bunch
// of bytes to a file, reads it via mmap_file_content and try to free
// it. But this require file I/O in a test.
#endif
