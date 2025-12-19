#pragma once

#include <cstdint>

// UBSan (Undefined Behavior Sanitizer) macros
#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
  __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ \
  __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_pointer_overflow__ \
  __attribute__((no_sanitize("pointer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#define __ubsan_ignore_float_cast_overflow__ \
  __attribute__((no_sanitize("float-cast-overflow")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_pointer_overflow__
#define __ubsan_ignore_function__
#define __ubsan_ignore_float_cast_overflow__
#endif

// STANDALONE_LIKELY/STANDALONE_UNLIKELY
//
// These macros provide parentheses, so you can use these macros as:
//
//    if STANDALONE_LIKELY(some_expr) {
//      ...
//    }
//
// NB: static_cast to boolean is mandatory in C++, because __builtin_expect
// takes a long argument, which means you may trigger the wrong conversion
// without it.
//
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define STANDALONE_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define STANDALONE_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define STANDALONE_LIKELY(expr) (expr)
#define STANDALONE_UNLIKELY(expr) (expr)
#endif

// On nvcc, STANDALONE_UNLIKELY thwarts missing return statement analysis.  In
// cases where the unlikely expression may be a constant, use this macro to
// ensure return statement analysis keeps working (at the cost of not getting
// the likely/unlikely annotation on nvcc).
// https://github.com/pytorch/pytorch/issues/21418
//
// Currently, this is only used in the error reporting macros below.  If you
// want to use it more generally, move me to Macros.h
//
// TODO: Brian Vaughan observed that we might be able to get this to work on
// nvcc by writing some sort of C++ overload that distinguishes constexpr inputs
// from non-constexpr.  Since there isn't any evidence that losing
// STANDALONE_UNLIKELY in nvcc is causing us perf problems, this is not yet
// implemented, but this might be an interesting piece of C++ code for an
// intrepid bootcamper to write.
#if defined(__CUDACC__)
#define STANDALONE_UNLIKELY_OR_CONST(e) e
#else
#define STANDALONE_UNLIKELY_OR_CONST(e) STANDALONE_UNLIKELY(e)
#endif

#define STANDALONE_STRINGIZE_IMPL(x) #x
#define STANDALONE_STRINGIZE(x) STANDALONE_STRINGIZE_IMPL(x)

#define STANDALONE_CONCATENATE_IMPL(s1, s2) s1##s2
#define STANDALONE_CONCATENATE(s1, s2) STANDALONE_CONCATENATE_IMPL(s1, s2)

/**
 * STANDALONE_ANONYMOUS_VARIABLE(str) introduces a new identifier which starts
 * with str and ends with a unique number.
 */
#ifdef __COUNTER__
#define STANDALONE_UID __COUNTER__
#define STANDALONE_ANONYMOUS_VARIABLE(str) \
  STANDALONE_CONCATENATE(str, __COUNTER__)
#else
#define STANDALONE_UID __LINE__
#define STANDALONE_ANONYMOUS_VARIABLE(str) STANDALONE_CONCATENATE(str, __LINE__)
#endif

// Private helper macro for workaround MSVC misexpansion of nested macro
// invocations involving __VA_ARGS__.  See
// https://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
#define STANDALONE_EXPAND_MSVC_WORKAROUND(x) x

/// STANDALONE_NOINLINE - Functions whose declaration is annotated with this
/// will not be inlined.
#ifdef __GNUC__
#define STANDALONE_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define STANDALONE_NOINLINE __declspec(noinline)
#else
#define STANDALONE_NOINLINE
#endif

#if defined(_MSC_VER)
#define STANDALONE_ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define STANDALONE_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define STANDALONE_ALWAYS_INLINE inline
#endif

// Unlike STANDALONE_ALWAYS_INLINE, STANDALONE_ALWAYS_INLINE_ATTRIBUTE can be
// used on a lambda.
#if defined(_MSC_VER)
// MSVC 14.39 is reasonably recent and doesn't like
// [[msvc::forceinline]] on a lambda, so don't try to use it.
#define STANDALONE_ALWAYS_INLINE_ATTRIBUTE
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define STANDALONE_ALWAYS_INLINE_ATTRIBUTE __attribute__((__always_inline__))
#else
#define STANDALONE_ALWAYS_INLINE_ATTRIBUTE
#endif

#if defined(_MSC_VER)
#define STANDALONE_ATTR_VISIBILITY_HIDDEN
#elif defined(__GNUC__)
#define STANDALONE_ATTR_VISIBILITY_HIDDEN \
  __attribute__((__visibility__("hidden")))
#else
#define STANDALONE_ATTR_VISIBILITY_HIDDEN
#endif

#define STANDALONE_ERASE \
  STANDALONE_ALWAYS_INLINE STANDALONE_ATTR_VISIBILITY_HIDDEN

#include <cstdint>

#ifdef __HIPCC__
// Unlike CUDA, HIP requires a HIP header to be included for __host__ to work.
// We do this #include here so that STANDALONE_HOST_DEVICE and friends will Just
// Work. See https://github.com/ROCm/hip/issues/441
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
// Designates functions callable from the host (CPU) and the device (GPU)
#define STANDALONE_HOST_DEVICE __host__ __device__
#define STANDALONE_DEVICE __device__
#define STANDALONE_HOST __host__
// constants from
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing
// architecture (7.5), 1536 for Geforce Ampere (8.6)/Jetson Orin (8.7), and
// 2048 for all other architectures. You'll get warnings if you exceed these
// constants. Hence, the following macros adjust the input values from the user
// to resolve potential warnings.
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence,
//       STANDALONE_MAX_THREADS_PER_BLOCK and STANDALONE_MIN_BLOCKS_PER_SM are
//       kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your
// performance tuning on a modern GPU. Instead, you should write
// __launch_bounds__(STANDALONE_MAX_THREADS_PER_BLOCK(a),
// STANDALONE_MIN_BLOCKS_PER_SM(a, b)), which will also properly respect limits
// on old architectures.
#define STANDALONE_MAX_THREADS_PER_BLOCK(val)    \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define STANDALONE_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm) \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /       \
           (threads_per_block))))
// STANDALONE_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define STANDALONE_LAUNCH_BOUNDS_0 \
  __launch_bounds__(               \
      256, 4) // default launch bounds that should give good occupancy
              // and versatility across all architectures.
#define STANDALONE_LAUNCH_BOUNDS_1(max_threads_per_block) \
  __launch_bounds__((STANDALONE_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define STANDALONE_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__(                                                         \
      (STANDALONE_MAX_THREADS_PER_BLOCK((max_threads_per_block))),           \
      (STANDALONE_MIN_BLOCKS_PER_SM(                                         \
          (max_threads_per_block), (min_blocks_per_sm))))
#else
#define STANDALONE_HOST_DEVICE
#define STANDALONE_HOST
#define STANDALONE_DEVICE
#endif

#define _STANDALONE_PRAGMA__(string) _Pragma(#string)
#define _STANDALONE_PRAGMA_(string) _STANDALONE_PRAGMA__(string)

#ifdef __clang__
#define STANDALONE_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
#define STANDALONE_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#define STANDALONE_CLANG_DIAGNOSTIC_IGNORE(flag) \
  _STANDALONE_PRAGMA_(clang diagnostic ignored flag)
#define STANDALONE_CLANG_HAS_WARNING(flag) __has_warning(flag)
#else
#define STANDALONE_CLANG_DIAGNOSTIC_PUSH()
#define STANDALONE_CLANG_DIAGNOSTIC_POP()
#define STANDALONE_CLANG_DIAGNOSTIC_IGNORE(flag)
#define STANDALONE_CLANG_HAS_WARNING(flag) 0
#endif
