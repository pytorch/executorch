/*
 * Copyright 2026 The ExecuTorch Authors.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Baremetal runner for qemu-system-riscv64 -machine virt + semihosting. Loads
// a .bpte embedded into the ELF and emits "TEST: BundleIO index[N]
// Test_result: PASS|FAIL" via ET_LOG so examples/riscv/run.sh's grep can
// detect success without a host filesystem.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <executorch/devtools/bundled_program/bundled_program.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include "semihosting.h"

extern "C" const uint8_t model_pte[];
extern "C" const size_t model_pte_len;

using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

// Pools are sized for the largest model we currently test (llama2 / yolo26)
// rather than per-model; the .bss grows but freestanding picolibc never
// allocates from it so the cost is just a bigger ELF. Bumping these requires
// matching headroom in riscv_virt.ld's RAM region and qemu's -m flag.
alignas(16) uint8_t method_allocator_pool[1u << 23]; //   8 MiB
alignas(16) uint8_t temp_allocator_pool[1u << 22]; //   4 MiB
alignas(16) uint8_t planned_memory_pool[1u << 26]; //  64 MiB

constexpr size_t kMaxPlannedBuffers = 8;
constexpr double kRtol = 0.01;
constexpr double kAtol = 0.01;

} // namespace

extern "C" [[noreturn]] void baremetal_exit(int status) {
  executorch::riscv::baremetal::semihost_exit(status);
}

// picolibc's abort()/raise() resolve _exit; with our own start.S we don't
// link its crt0, so reroute it to the semihosting trap.
extern "C" [[noreturn]] void _exit(int status) {
  executorch::riscv::baremetal::semihost_exit(status);
}

// libstdc++'s <random> drags std::random_device → getentropy/read. The portable
// rand kernels are never invoked at runtime for our bundled-IO tests, so a
// failing stub is enough to satisfy the link.
extern "C" int getentropy(void*, size_t) {
  return -1;
}
extern "C" long read(int, void*, size_t) {
  return -1;
}

// Virtual destructors emit deleting variants that reference operator delete
// even when we never new/delete. Stubs satisfy the linker; never called.
void operator delete(void*) noexcept {}
void operator delete(void*, size_t) noexcept {}
void operator delete[](void*) noexcept {}
void operator delete[](void*, size_t) noexcept {}

// op_rand / op_native_dropout / op_randn from portable_kernels reference
// std::random_device::_M_{init,getval,fini}, whose only definitions live in
// libstdc++.a's medlow-built random.o (won't relocate at 0x80000000). The
// bundled-IO smoke tests never invoke those ops, so satisfy the linker with
// no-op trampolines under the Itanium-mangled names.
asm(R"(
    .globl _ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
    .type  _ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, @function
_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE:
    ret

    .globl _ZNSt13random_device9_M_getvalEv
    .type  _ZNSt13random_device9_M_getvalEv, @function
_ZNSt13random_device9_M_getvalEv:
    li     a0, 0
    ret

    .globl _ZNSt13random_device7_M_finiEv
    .type  _ZNSt13random_device7_M_finiEv, @function
_ZNSt13random_device7_M_finiEv:
    ret
)");

// Route ET_LOG through semihosting. Messages aren't null-terminated; copy and
// append \n\0 before forwarding to SYS_WRITE0.
extern "C" void et_pal_emit_log_message(
    et_timestamp_t,
    et_pal_log_level_t,
    const char*,
    const char*,
    size_t,
    const char* message,
    size_t length) {
  // The bundle doesn't expose a testset count, so we probe past the end and
  // rely on InvalidArgument to terminate the loop. The accompanying ET_LOG
  // ("testset_idx N is out of range ...") is benign noise — suppress it so
  // run.sh's PASS/FAIL grep stays clean.
  static const char kOorPrefix[] = "testset_idx ";
  if (length >= sizeof(kOorPrefix) - 1 &&
      std::memcmp(message, kOorPrefix, sizeof(kOorPrefix) - 1) == 0) {
    return;
  }
  char buf[512];
  size_t n = length < sizeof(buf) - 2 ? length : sizeof(buf) - 2;
  std::memcpy(buf, message, n);
  buf[n] = '\n';
  buf[n + 1] = '\0';
  executorch::riscv::baremetal::semihost_write0(buf);
}

extern "C" void et_pal_init(void) {}
extern "C" [[noreturn]] void et_pal_abort(void) {
  executorch::riscv::baremetal::semihost_exit(1);
}
extern "C" et_timestamp_t et_pal_current_ticks(void) {
  return 0;
}
extern "C" et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  return {1, 1};
}
extern "C" void* et_pal_allocate(size_t) {
  return nullptr;
}
extern "C" void et_pal_free(void*) {}

int main() {
  executorch::runtime::runtime_init();

  const void* program_data = nullptr;
  size_t program_size = 0;
  Error status = executorch::bundled_program::get_program_data(
      const_cast<uint8_t*>(model_pte),
      model_pte_len,
      &program_data,
      &program_size);
  if (status != Error::Ok) {
    ET_LOG(
        Error, "get_program_data failed: 0x%x", static_cast<unsigned>(status));
    return 1;
  }

  BufferDataLoader loader(program_data, program_size);
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(
        Error,
        "Program::load failed: 0x%x",
        static_cast<unsigned>(program.error()));
    return 1;
  }

  // The harness always exports a single "forward" method. Skipping the
  // Result<const char*> deref of program->get_method_name(0) sidesteps a
  // codegen wedge we hit under -mcmodel=medany + picolibc.
  const char* method_name = "forward";
  ET_LOG(Info, "Using method %s", method_name);

  Result<MethodMeta> method_meta = program->method_meta(method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Error,
        "method_meta failed: 0x%x",
        static_cast<unsigned>(method_meta.error()));
    return 1;
  }

  MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);
  MemoryAllocator temp_allocator(
      sizeof(temp_allocator_pool), temp_allocator_pool);

  // One span per planned buffer, bumped through a single .bss arena so we
  // don't need a heap. kMaxPlannedBuffers / pool size both grow with bigger
  // models; failures here are loud rather than silent.
  Span<uint8_t> planned_spans[kMaxPlannedBuffers];
  size_t num_planned = method_meta->num_memory_planned_buffers();
  if (num_planned > kMaxPlannedBuffers) {
    ET_LOG(
        Error,
        "num_planned=%zu exceeds kMaxPlannedBuffers=%zu",
        num_planned,
        kMaxPlannedBuffers);
    return 1;
  }
  size_t offset = 0;
  for (size_t id = 0; id < num_planned; ++id) {
    size_t sz =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    sz = (sz + 15u) & ~15u;
    if (offset + sz > sizeof(planned_memory_pool)) {
      ET_LOG(
          Error,
          "planned buffer %zu (size %zu) overflows pool (%zu/%zu)",
          id,
          sz,
          offset,
          sizeof(planned_memory_pool));
      return 1;
    }
    planned_spans[id] = Span<uint8_t>(planned_memory_pool + offset, sz);
    offset += sz;
  }
  HierarchicalAllocator planned_memory(
      Span<Span<uint8_t>>(planned_spans, num_planned));
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method = program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    ET_LOG(
        Error,
        "load_method failed: 0x%x",
        static_cast<unsigned>(method.error()));
    return 1;
  }

  // load_bundled_input returns InvalidArgument past the last testset; that's
  // how we detect the loop terminator (the bundle has no public count API).
  int rc = 0;
  for (size_t testset_idx = 0;; ++testset_idx) {
    Error load = executorch::bundled_program::load_bundled_input(
        *method, const_cast<uint8_t*>(model_pte), testset_idx);
    if (load != Error::Ok) {
      if (testset_idx == 0) {
        ET_LOG(
            Error,
            "load_bundled_input failed for testset 0: 0x%x",
            static_cast<unsigned>(load));
        rc = 1;
      }
      break;
    }
    Error exec = method->execute();
    if (exec != Error::Ok) {
      ET_LOG(
          Error,
          "execute failed for testset %zu: 0x%x",
          testset_idx,
          static_cast<unsigned>(exec));
      ET_LOG(Error, "TEST: BundleIO index[%zu] Test_result: FAIL", testset_idx);
      rc = 1;
      continue;
    }
    Error verify = executorch::bundled_program::verify_method_outputs(
        *method, const_cast<uint8_t*>(model_pte), testset_idx, kRtol, kAtol);
    if (verify == Error::Ok) {
      ET_LOG(Info, "TEST: BundleIO index[%zu] Test_result: PASS", testset_idx);
    } else {
      ET_LOG(
          Error,
          "verify_method_outputs failed for testset %zu: 0x%x",
          testset_idx,
          static_cast<unsigned>(verify));
      ET_LOG(Error, "TEST: BundleIO index[%zu] Test_result: FAIL", testset_idx);
      rc = 1;
    }
  }

  return rc;
}
