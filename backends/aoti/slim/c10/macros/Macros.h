#pragma once

// SlimTensor Macros Header
//
// This header bridges between SlimTensor's STANDALONE_* macro conventions and
// ExecuTorch's C10_* macros from portable_type. It includes the base macros
// from portable_type and provides STANDALONE_* aliases for backward
// compatibility.

#include <c10/macros/Macros.h>

// =============================================================================
// STANDALONE_* to C10_* macro mappings
// =============================================================================
// These mappings allow SlimTensor code to use STANDALONE_* macros while
// actually using the underlying C10_* implementations from portable_type.

// Host/Device macros
#define STANDALONE_HOST_DEVICE C10_HOST_DEVICE
#define STANDALONE_DEVICE C10_DEVICE
#define STANDALONE_HOST C10_HOST

// Compiler hint macros
#define STANDALONE_LIKELY C10_LIKELY
#define STANDALONE_UNLIKELY C10_UNLIKELY
#define STANDALONE_UNLIKELY_OR_CONST C10_UNLIKELY

// String/concatenation macros
#define STANDALONE_STRINGIZE_IMPL C10_STRINGIZE_IMPL
#define STANDALONE_STRINGIZE C10_STRINGIZE
#define STANDALONE_CONCATENATE_IMPL C10_CONCATENATE_IMPL
#define STANDALONE_CONCATENATE C10_CONCATENATE

// Anonymous variable macros
#define STANDALONE_UID C10_UID
#define STANDALONE_ANONYMOUS_VARIABLE C10_ANONYMOUS_VARIABLE

// MSVC workaround
#define STANDALONE_EXPAND_MSVC_WORKAROUND C10_MACRO_EXPAND

// Inline/visibility macros
#define STANDALONE_NOINLINE C10_NOINLINE
#define STANDALONE_ALWAYS_INLINE C10_ALWAYS_INLINE
#define STANDALONE_ALWAYS_INLINE_ATTRIBUTE C10_ALWAYS_INLINE_ATTRIBUTE
#define STANDALONE_ATTR_VISIBILITY_HIDDEN C10_ATTR_VISIBILITY_HIDDEN
#define STANDALONE_ERASE C10_ERASE

// Clang diagnostic macros
#define STANDALONE_CLANG_DIAGNOSTIC_PUSH C10_CLANG_DIAGNOSTIC_PUSH
#define STANDALONE_CLANG_DIAGNOSTIC_POP C10_CLANG_DIAGNOSTIC_POP
#define STANDALONE_CLANG_DIAGNOSTIC_IGNORE C10_CLANG_DIAGNOSTIC_IGNORE
#define STANDALONE_CLANG_HAS_WARNING C10_CLANG_HAS_WARNING

// CUDA launch bounds (these are identical between STANDALONE and C10)
#ifdef __CUDACC__
#define STANDALONE_MAX_THREADS_PER_BLOCK C10_MAX_THREADS_PER_BLOCK
#define STANDALONE_MIN_BLOCKS_PER_SM C10_MIN_BLOCKS_PER_SM
#define STANDALONE_LAUNCH_BOUNDS_0 C10_LAUNCH_BOUNDS_0
#define STANDALONE_LAUNCH_BOUNDS_1 C10_LAUNCH_BOUNDS_1
#define STANDALONE_LAUNCH_BOUNDS_2 C10_LAUNCH_BOUNDS_2
#endif
