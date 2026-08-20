/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>

#if defined(EXECUTORCH_SIZE_TEST_NO_OS_LINK)
#include <errno.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

static uint8_t method_allocator_pool[1024];
static uint8_t activation_pool[512];

#if defined(EXECUTORCH_SIZE_TEST_NO_OS_LINK)
#define ET_WEAK_SYSCALL __attribute__((weak))

extern "C" {

// The Zephyr size test links directly with arm-zephyr-eabi, outside Zephyr's
// normal application link flow. This binary is link/size-only and is never run;
// these stubs only satisfy libc hooks pulled in by FileDataLoader and
// profiling.
#ifdef stderr
#undef stderr
#endif
#if !defined(__GLIBC__)
extern FILE* const stderr ET_WEAK_SYSCALL = nullptr;
#endif

ET_WEAK_SYSCALL int close(int fd) {
  (void)fd;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL int _close(int fd) {
  return close(fd);
}

ET_WEAK_SYSCALL int fstat(int fd, struct stat* st) {
  (void)fd;
  st->st_mode = S_IFCHR;
  return 0;
}

ET_WEAK_SYSCALL int _fstat(int fd, struct stat* st) {
  return fstat(fd, st);
}

ET_WEAK_SYSCALL int gettimeofday(void* tv, void* tz) {
  (void)tv;
  (void)tz;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL int getentropy(void* buffer, size_t length) {
  (void)buffer;
  (void)length;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL int _getentropy(void* buffer, size_t length) {
  return getentropy(buffer, length);
}

ET_WEAK_SYSCALL int isatty(int fd) {
  (void)fd;
  return 1;
}

ET_WEAK_SYSCALL int _isatty(int fd) {
  return isatty(fd);
}

ET_WEAK_SYSCALL off_t lseek(int fd, off_t offset, int whence) {
  (void)fd;
  (void)offset;
  (void)whence;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL off_t _lseek(int fd, off_t offset, int whence) {
  return lseek(fd, offset, whence);
}

ET_WEAK_SYSCALL int open(const char* path, int flags, ...) {
  (void)path;
  (void)flags;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL int _open(const char* path, int flags, ...) {
  (void)path;
  (void)flags;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL ssize_t read(int fd, void* buf, size_t count) {
  (void)fd;
  (void)buf;
  (void)count;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL ssize_t _read(int fd, void* buf, size_t count) {
  return read(fd, buf, count);
}

ET_WEAK_SYSCALL ssize_t write(int fd, const void* buf, size_t count) {
  (void)fd;
  (void)buf;
  (void)count;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL ssize_t _write(int fd, const void* buf, size_t count) {
  return write(fd, buf, count);
}

ET_WEAK_SYSCALL void* _sbrk(ptrdiff_t increment) {
  (void)increment;
  errno = ENOMEM;
  return (void*)-1;
}

ET_WEAK_SYSCALL void* sbrk(ptrdiff_t increment) {
  return _sbrk(increment);
}

ET_WEAK_SYSCALL int getpid(void) {
  return 1;
}

ET_WEAK_SYSCALL int _getpid(void) {
  return getpid();
}

ET_WEAK_SYSCALL int kill(int pid, int sig) {
  (void)pid;
  (void)sig;
  errno = ENOSYS;
  return -1;
}

ET_WEAK_SYSCALL int _kill(int pid, int sig) {
  return kill(pid, sig);
}

ET_WEAK_SYSCALL void _exit(int status) {
  (void)status;
  __builtin_trap();
  for (;;) {
  }
}

} // extern "C"

#undef ET_WEAK_SYSCALL
#endif

int main(int argc, char** argv) {
  runtime_init();

  ET_CHECK_MSG(argc == 2, "Expected model file argument.");

  MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);
  method_allocator.enable_profiling("method allocator");

  Span<uint8_t> memory_planned_buffers[1]{
      {activation_pool, sizeof(activation_pool)}};
  HierarchicalAllocator planned_memory({memory_planned_buffers, 1});

  MemoryManager memory_manager(&method_allocator, &planned_memory);

  Result<FileDataLoader> loader = FileDataLoader::from(argv[1]);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      static_cast<uint32_t>(loader.error()));

  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("de-serialize model");
  const auto program = Program::load(&loader.get());
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK_MSG(
      program.ok(),
      "Program::load() failed: 0x%" PRIx32,
      static_cast<uint32_t>(program.error()));
  ET_LOG(Info, "Program file %s loaded.", argv[1]);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Loading method %s", method_name);

  prof_tok = EXECUTORCH_BEGIN_PROF("load model");
  Result<Method> method = program->load_method(method_name, &memory_manager);
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(method.ok());

  ET_LOG(Info, "Method loaded.");

  // Prepare for inputs
  // It assumes the input is one tensor.
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Tensor::SizesType sizes[] = {6};
  Tensor::DimOrderType dim_order[] = {0};
  TensorImpl impl(ScalarType::Float, 1, sizes, data, dim_order);
  Tensor t(&impl);
  Error set_input_error = method->set_input(t, 0);
  ET_CHECK(set_input_error == Error::Ok);

  ET_LOG(Info, "Inputs prepared.");

  prof_tok = EXECUTORCH_BEGIN_PROF("run model");
  Error status = method->execute();
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(status == Error::Ok);
  ET_LOG(Info, "Model executed successfully.");

  // print output
  auto output_list =
      method_allocator.allocateList<EValue>(method->outputs_size());

  status = method->get_outputs(output_list, method->outputs_size());
  ET_CHECK(status == Error::Ok);

  // It assumes the outputs are all tensors.
  for (const auto i : c10::irange(method->outputs_size())) {
    auto output_tensor = output_list[i].toTensor();
    ET_UNUSED auto data_output = output_tensor.const_data_ptr<float>();
    for (ET_UNUSED const auto j : c10::irange(output_tensor.numel())) {
      ET_LOG(Info, "%f", data_output[j]);
    }
  }
  prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
  if (prof_result.num_bytes != 0) {
    FILE* ptr = fopen("prof_result.bin", "w+");
    fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
    fclose(ptr);
  }

  return 0;
}
