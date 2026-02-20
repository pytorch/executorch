/* Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef UBSAN_RUNTIME_PREFIX
#define UBSAN_RUNTIME_PREFIX "[UBSAN] "
#endif

typedef struct {
  const char* filename;
  uint32_t line;
  uint32_t column;
} __ubsan_source_location;

typedef struct {
  uint16_t type_kind;
  uint16_t type_info;
  char type_name[];
} __ubsan_type_descriptor;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* type;
} __ubsan_overflow_data;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* lhs_type;
  const __ubsan_type_descriptor* rhs_type;
} __ubsan_shift_out_of_bounds_data;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* array_type;
  const __ubsan_type_descriptor* index_type;
} __ubsan_out_of_bounds_data;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* type;
  uint8_t log_alignment;
  uint8_t type_check_kind;
} __ubsan_type_mismatch_data_v1;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* type;
} __ubsan_vla_bound_data;

typedef struct {
  __ubsan_source_location location;
  __ubsan_source_location attr_location;
} __ubsan_nonnull_return_data_v1;

typedef struct {
  __ubsan_source_location location;
  __ubsan_source_location attr_location;
  uint8_t arg_index;
} __ubsan_nullability_arg_data;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* from_type;
  const __ubsan_type_descriptor* to_type;
} __ubsan_float_cast_overflow_data;

typedef struct {
  __ubsan_source_location location;
  const __ubsan_type_descriptor* type;
} __ubsan_invalid_value_data;

typedef struct {
  __ubsan_source_location location;
  __ubsan_source_location attr_location;
  uint32_t arg_index;
} __ubsan_nonnull_arg_data;

typedef struct {
  __ubsan_source_location location;
} __ubsan_pointer_overflow_data;

typedef struct {
  __ubsan_source_location location;
  __ubsan_source_location assumption_location;
  uint64_t alignment;
  uint8_t type_check_kind;
} __ubsan_alignment_assumption_data;

static const char* ubsan_get_type_name(const __ubsan_type_descriptor* type) {
  if (!type) {
    return "<unknown>";
  }
  return type->type_name;
}

static const char* ubsan_type_check_kind_string(uint8_t kind) {
  switch (kind) {
    case 0:
      return "load of";
    case 1:
      return "store to";
    case 2:
      return "reference binding to";
    case 3:
      return "member access within";
    case 4:
      return "member call on";
    case 5:
      return "constructor call for";
    case 6:
      return "downcast of";
    case 7:
      return "downcast of";
    case 8:
      return "upcast of";
    case 9:
      return "cast to virtual base of";
    default:
      return "use of";
  }
}

static uintptr_t ubsan_ptr_value(const void* ptr) {
  return (uintptr_t)ptr;
}

static void ubsan_abort(void) {
#if defined(__GNUC__)
  __builtin_trap();
#else
  abort();
#endif
  while (1) {
  }
}

static void ubsan_print_location(const __ubsan_source_location* loc) {
  if (!loc || !loc->filename) {
    printf(UBSAN_RUNTIME_PREFIX "unknown location: ");
    return;
  }
  printf(UBSAN_RUNTIME_PREFIX "%s:%u:%u: ", loc->filename, loc->line,
         loc->column);
}

static void ubsan_report_with_message(const __ubsan_source_location* loc,
                                      const char* message) {
  ubsan_print_location(loc);
  printf("%s\n", message);
  fflush(stdout);
  ubsan_abort();
}

static void ubsan_report_overflow(const __ubsan_overflow_data* data,
                                  const char* op,
                                  uintptr_t lhs,
                                  uintptr_t rhs) {
  const char* type_name = ubsan_get_type_name(data->type);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "%s on type '%s' (lhs=0x%08" PRIxPTR ", rhs=0x%08" PRIxPTR ")",
      op,
      type_name,
      lhs,
      rhs);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_add_overflow(__ubsan_overflow_data* data, void* lhs,
                                 void* rhs) {
  ubsan_report_overflow(
      data,
      "addition overflow",
      ubsan_ptr_value(lhs),
      ubsan_ptr_value(rhs));
}

void __ubsan_handle_sub_overflow(__ubsan_overflow_data* data, void* lhs,
                                 void* rhs) {
  ubsan_report_overflow(
      data,
      "subtraction overflow",
      ubsan_ptr_value(lhs),
      ubsan_ptr_value(rhs));
}

void __ubsan_handle_mul_overflow(__ubsan_overflow_data* data, void* lhs,
                                 void* rhs) {
  ubsan_report_overflow(
      data,
      "multiplication overflow",
      ubsan_ptr_value(lhs),
      ubsan_ptr_value(rhs));
}

void __ubsan_handle_negate_overflow(__ubsan_overflow_data* data, void* value) {
  ubsan_report_overflow(
      data,
      "negation overflow",
      ubsan_ptr_value(value),
      0);
}

void __ubsan_handle_divrem_overflow(__ubsan_overflow_data* data, void* lhs,
                                    void* rhs) {
  ubsan_report_overflow(
      data,
      "division remainder overflow",
      ubsan_ptr_value(lhs),
      ubsan_ptr_value(rhs));
}

void __ubsan_handle_shift_out_of_bounds(__ubsan_shift_out_of_bounds_data* data,
                                        void* lhs, void* rhs) {
  const char* lhs_type = ubsan_get_type_name(data->lhs_type);
  const char* rhs_type = ubsan_get_type_name(data->rhs_type);
  uintptr_t lhs_val = ubsan_ptr_value(lhs);
  uintptr_t rhs_val = ubsan_ptr_value(rhs);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "shift out of bounds (lhs=0x%08" PRIxPTR " of type '%s', rhs=0x%08" PRIxPTR
      " of type '%s')",
      lhs_val,
      lhs_type,
      rhs_val,
      rhs_type);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_out_of_bounds(__ubsan_out_of_bounds_data* data,
                                  void* index) {
  uintptr_t idx_val = ubsan_ptr_value(index);
  const char* idx_type = ubsan_get_type_name(data->index_type);
  const char* array_type = ubsan_get_type_name(data->array_type);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "index out of bounds (index=0x%08" PRIxPTR " of type '%s' on array '%s')",
      idx_val,
      idx_type,
      array_type);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_type_mismatch_v1(__ubsan_type_mismatch_data_v1* data,
                                     void* ptr) {
  uintptr_t address = (uintptr_t)ptr;
  size_t alignment =
      (data->log_alignment < (sizeof(size_t) * 8))
      ? ((size_t)1 << data->log_alignment)
      : 0;
  const char* type_name = ubsan_get_type_name(data->type);
  const char* check_desc = ubsan_type_check_kind_string(data->type_check_kind);

  char message[256];
  if (address == 0) {
    snprintf(
        message,
        sizeof(message),
        "%s null pointer of type '%s'",
        check_desc,
        type_name);
  } else if (alignment && (address & (alignment - 1))) {
    snprintf(
        message,
        sizeof(message),
        "%s misaligned address 0x%08" PRIxPTR " for type '%s' (alignment %zu)",
        check_desc,
        address,
        type_name,
        alignment);
  } else {
    snprintf(
        message,
        sizeof(message),
        "%s address 0x%08" PRIxPTR " with insufficient alignment for type '%s'",
        check_desc,
        address,
        type_name);
  }

  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_vla_bound_not_positive(__ubsan_vla_bound_data* data,
                                           void* bound) {
  uintptr_t bound_val = ubsan_ptr_value(bound);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "variable length array bound (%" PRIuPTR ") is not positive",
      (uintptr_t)bound_val);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_load_invalid_value(__ubsan_invalid_value_data* data,
                                       void* pointer) {
  uintptr_t addr = ubsan_ptr_value(pointer);
  const char* type_name = ubsan_get_type_name(data->type);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "load of invalid value at 0x%08" PRIxPTR " for type '%s'",
      addr,
      type_name);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_nonnull_return_v1(__ubsan_nonnull_return_data_v1* data,
                                      __ubsan_source_location* where) {
  (void)where; // Some toolchains leave this null; attr_location is reliable.
  char message[256];
  if (data->attr_location.filename) {
    snprintf(
        message,
        sizeof(message),
        "null pointer returned from function marked 'returns_nonnull' "
        "(attribute at %s:%u:%u)",
        data->attr_location.filename,
        data->attr_location.line,
        data->attr_location.column);
  } else {
    snprintf(
        message,
        sizeof(message),
        "null pointer returned from function marked 'returns_nonnull'");
  }
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_nullability_return_v1(
    __ubsan_nonnull_return_data_v1* data, __ubsan_source_location* where) {
  (void)where; // Some toolchains leave this null; attr_location is reliable.
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "null returned from non-null return (attribute at %s:%u:%u)",
      data->attr_location.filename ? data->attr_location.filename : "<unknown>",
      data->attr_location.line,
      data->attr_location.column);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_nullability_arg_v1(__ubsan_nullability_arg_data* data,
                                       __ubsan_source_location* where) {
  (void)where; // Some toolchains leave this null; attr_location is reliable.
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "null passed to non-null argument #%u (attribute at %s:%u:%u)",
      data->arg_index,
      data->attr_location.filename ? data->attr_location.filename : "<unknown>",
      data->attr_location.line,
      data->attr_location.column);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_nonnull_arg(__ubsan_nonnull_arg_data* data) {
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "null pointer passed to argument marked 'nonnull' (argument #%u, attribute at %s:%u:%u)",
      data->arg_index,
      data->attr_location.filename ? data->attr_location.filename : "<unknown>",
      data->attr_location.line,
      data->attr_location.column);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_float_cast_overflow(
    __ubsan_float_cast_overflow_data* data, void* from) {
  uintptr_t raw = ubsan_ptr_value(from);
  const char* from_type = ubsan_get_type_name(data->from_type);
  const char* to_type = ubsan_get_type_name(data->to_type);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "floating point cast overflow (value bits=0x%08" PRIxPTR
      ", from '%s' to '%s')",
      raw,
      from_type,
      to_type);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_pointer_overflow(__ubsan_pointer_overflow_data* data,
                                     void* base, void* result) {
  uintptr_t base_val = ubsan_ptr_value(base);
  uintptr_t result_val = ubsan_ptr_value(result);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "pointer overflow (base=0x%08" PRIxPTR ", result=0x%08" PRIxPTR ")",
      base_val,
      result_val);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_alignment_assumption(
    __ubsan_alignment_assumption_data* data, void* pointer,
    void* alignment, void* offset) {
  uintptr_t ptr_val = ubsan_ptr_value(pointer);
  uintptr_t align_val = ubsan_ptr_value(alignment);
  uintptr_t offset_val = ubsan_ptr_value(offset);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "alignment assumption violated (ptr=0x%08" PRIxPTR ", alignment=%" PRIuPTR
      ", offset=%" PRIuPTR ", required alignment=%" PRIu64 ")",
      ptr_val,
      align_val,
      offset_val,
      (unsigned long long)data->alignment);
  ubsan_report_with_message(&data->location, message);
}

void __ubsan_handle_builtin_unreachable(__ubsan_source_location* location) {
  ubsan_report_with_message(location, "execution reached an unreachable point");
}

void __ubsan_handle_missing_return(__ubsan_source_location* location) {
  ubsan_report_with_message(location,
                            "control reached end of void function without "
                            "returning");
}

void __ubsan_handle_invalid_builtin(__ubsan_source_location* location) {
  ubsan_report_with_message(location, "invalid builtin usage");
}

void __ubsan_handle_cfi_check_fail(__ubsan_source_location* location,
                                   void* data, void* vtable) {
  uintptr_t type_hash = ubsan_ptr_value(data);
  uintptr_t vtable_ptr = ubsan_ptr_value(vtable);
  char message[256];
  snprintf(
      message,
      sizeof(message),
      "control-flow integrity check failed (type hash=0x%08" PRIxPTR
      ", vtable=0x%08" PRIxPTR ")",
      type_hash,
      vtable_ptr);
  ubsan_report_with_message(location, message);
}

void __ubsan_handle_cfi_check_fail_abort(__ubsan_source_location* location,
                                         void* data, void* vtable) {
  __ubsan_handle_cfi_check_fail(location, data, vtable);
}

void __ubsan_handle_dynamic_type_cache_miss(void* data, void* ptr) {
  uintptr_t type_hash = ubsan_ptr_value(data);
  uintptr_t object_ptr = ubsan_ptr_value(ptr);
  printf(
      UBSAN_RUNTIME_PREFIX
      "dynamic type cache miss (type hash=0x%08" PRIxPTR ", object=0x%08" PRIxPTR
      ")\n",
      type_hash,
      object_ptr);
  fflush(stdout);
  ubsan_abort();
}

void __ubsan_on_error(void) {
  printf(UBSAN_RUNTIME_PREFIX "runtime error detected\n");
  fflush(stdout);
  ubsan_abort();
}
